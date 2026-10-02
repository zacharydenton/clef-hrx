//! HRX graph execution. Tensor leases govern scratch reuse while
//! byte-range access declarations order reuse inside a recorded graph.
use crate::Result;
use anyhow::{Context, ensure};
use hrx::{
    Buffer, Constants, GraphExec, Kernel, Stream,
    inference::ModelContext,
    loom::{Compiler, Specialization},
};
use std::{
    collections::HashMap,
    rc::{Rc, Weak},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum DType {
    Bf16 = 0,
    F32 = 1,
    U32 = 2,
}
impl DType {
    pub fn bytes(self) -> usize {
        if self == Self::Bf16 { 2 } else { 4 }
    }
}
#[derive(Clone)]
pub(crate) struct Tensor {
    pub slot: usize,
    pub offset: usize,
    pub rows: usize,
    pub cols: usize,
    pub dtype: DType,
    _lease: Rc<()>,
}
impl Tensor {
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
    pub fn reshape(&self, rows: usize, cols: usize) -> Self {
        assert_eq!(rows * cols, self.len());
        Self {
            rows,
            cols,
            ..self.clone()
        }
    }
    pub fn slice_rows(&self, start: usize, rows: usize) -> Self {
        assert!(start + rows <= self.rows);
        Self {
            offset: self.offset + start * self.cols * self.dtype.bytes(),
            rows,
            ..self.clone()
        }
    }
}
struct Allocation {
    buffer: Buffer,
    lease: Weak<()>,
    permanent: bool,
    host_input: bool,
}
struct Binding {
    slot: usize,
    offset: usize,
    bytes: usize,
    access: hrx::Access,
}
struct Command {
    #[cfg(feature = "bench-internals")]
    label: String,
    kernel: Kernel,
    constants: Constants,
    grid: [u32; 3],
    bindings: Vec<Binding>,
}
pub(crate) struct Engine {
    stream: Stream,
    compiler: Compiler,
    allocations: Vec<Allocation>,
    kernels: HashMap<String, Kernel>,
    commands: Vec<Command>,
    pub weights: HashMap<String, Tensor>,
    pub embedding_rows: HashMap<String, crate::embedding::EmbeddingRows>,
    pub retained: Vec<Tensor>,
    pub uploaded: u64,
    traces: Vec<(String, Tensor)>,
}
impl Engine {
    /// Specializes checked-in Loom sources using numeric configuration only.
    pub fn dispatch(
        &mut self,
        source: &str,
        work: usize,
        inputs: &[(&str, &Tensor)],
        output: &Tensor,
        parameters: &[(&str, String)],
    ) -> Result<()> {
        let mut spec = Specialization::new("clef_op");
        for (name, value) in [
            ("work", work),
            ("groups", work.div_ceil(64)),
            ("out_len", output.len()),
            ("out_dtype", output.dtype as usize),
        ] {
            spec.set_config(format!("clef.{name}"), value.to_string());
        }
        for (name, tensor) in inputs {
            spec.set_config(format!("clef.{name}_len"), tensor.len().to_string());
            spec.set_config(
                format!("clef.{name}_dtype"),
                (tensor.dtype as usize).to_string(),
            );
        }
        for (name, value) in parameters {
            spec.set_config(format!("clef.{name}"), value.clone());
        }
        self.emit_special(
            source,
            spec,
            Constants::new(),
            [work.div_ceil(64) as u32, 1, 1],
            &inputs.iter().map(|(_, tensor)| *tensor).collect::<Vec<_>>(),
            output,
        )
    }
    pub fn new(context: &ModelContext) -> Result<Self> {
        let gpu = context.runtime().gpu()?;
        let mut stream = hrx::Device::open(gpu.index())?.stream()?;
        if let Some(budget) = context.runtime().memory_budget() {
            stream = stream.with_memory_budget(budget.clone());
        }
        ensure!(
            stream.target().as_str() == "gfx1151",
            "CLEF currently requires gfx1151"
        );
        let compiler = Compiler::for_stream(None, &stream)?;
        Ok(Self {
            stream,
            compiler,
            allocations: Vec::new(),
            kernels: HashMap::new(),
            commands: Vec::new(),
            weights: HashMap::new(),
            embedding_rows: HashMap::new(),
            retained: Vec::new(),
            uploaded: 0,
            traces: Vec::new(),
        })
    }
    pub fn alloc(&mut self, rows: usize, cols: usize, dtype: DType) -> Result<Tensor> {
        let bytes = rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(dtype.bytes()))
            .context("tensor size overflow")?;
        ensure!(
            bytes > 0 && rows * cols <= u32::MAX as usize,
            "invalid tensor extent"
        );
        let slot = self
            .allocations
            .iter()
            .enumerate()
            .filter(|(_, a)| {
                !a.permanent
                    && !a.host_input
                    && a.lease.strong_count() == 0
                    && a.buffer.bytes() >= bytes
            })
            .min_by_key(|(_, a)| a.buffer.bytes())
            .map(|(i, _)| i);
        let lease = Rc::new(());
        let slot = if let Some(i) = slot {
            self.allocations[i].lease = Rc::downgrade(&lease);
            i
        } else {
            let i = self.allocations.len();
            crate::memory::before_allocation(bytes)?;
            self.allocations.push(Allocation {
                buffer: self.stream.allocate(bytes)?,
                lease: Rc::downgrade(&lease),
                permanent: false,
                host_input: false,
            });
            i
        };
        Ok(Tensor {
            slot,
            offset: 0,
            rows,
            cols,
            dtype,
            _lease: lease,
        })
    }
    pub fn upload(&mut self, t: &Tensor, bytes: &[u8]) -> Result<()> {
        ensure!(
            bytes.len() == t.len() * t.dtype.bytes(),
            "upload shape mismatch"
        );
        const CHUNK: usize = 16 << 20;
        for (i, chunk) in bytes.chunks(CHUNK).enumerate() {
            let view = self.allocations[t.slot]
                .buffer
                .binding()
                .slice(t.offset + i * CHUNK, chunk.len())?;
            self.stream.upload(view, chunk)?;
            // Bound staging even during a full 55 GB checkpoint load.
            self.stream.synchronize()?;
        }
        self.uploaded += bytes.len() as u64;
        Ok(())
    }
    pub fn input(
        &mut self,
        rows: usize,
        cols: usize,
        dtype: DType,
        bytes: &[u8],
    ) -> Result<Tensor> {
        let size = rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(dtype.bytes()))
            .context("input size overflow")?;
        ensure!(size > 0 && size == bytes.len(), "invalid input extent");
        // Host uploads precede the complete graph. Never alias a host input
        // with earlier scratch, even if its Rust lease has already expired.
        let old = self
            .allocations
            .iter()
            .enumerate()
            .filter(|(_, a)| {
                a.host_input && a.lease.strong_count() == 0 && a.buffer.bytes() >= size
            })
            .min_by_key(|(_, a)| a.buffer.bytes())
            .map(|(i, _)| i);
        let lease = Rc::new(());
        let slot = if let Some(i) = old {
            self.allocations[i].lease = Rc::downgrade(&lease);
            i
        } else {
            let i = self.allocations.len();
            crate::memory::before_allocation(size)?;
            self.allocations.push(Allocation {
                buffer: self.stream.allocate(size)?,
                lease: Rc::downgrade(&lease),
                permanent: false,
                host_input: true,
            });
            i
        };
        let t = Tensor {
            slot,
            offset: 0,
            rows,
            cols,
            dtype,
            _lease: lease,
        };
        self.upload(&t, bytes)?;
        self.retained.push(t.clone());
        Ok(t)
    }
    pub fn weight(&mut self, name: String, rows: usize, cols: usize, bytes: &[u8]) -> Result<()> {
        let t = self.alloc(rows, cols, DType::Bf16)?;
        self.upload(&t, bytes)?;
        self.allocations[t.slot].permanent = true;
        self.weights.insert(name, t);
        Ok(())
    }
    pub fn w(&self, name: &str) -> Result<Tensor> {
        self.weights
            .get(name)
            .cloned()
            .with_context(|| format!("weight {name}"))
    }
    pub fn emit_special(
        &mut self,
        source: &str,
        spec: Specialization,
        constants: Constants,
        grid: [u32; 3],
        inputs: &[&Tensor],
        output: &Tensor,
    ) -> Result<()> {
        let key = format!("{}{:?}", source, spec.configuration());
        let kernel = if let Some(k) = self.kernels.get(&key) {
            k.clone()
        } else {
            let artifact = self
                .compiler
                .module(source)
                .compile(&spec)
                .context("compiling CLEF kernel")?;
            // Safety: sources are crate-owned; all binding extents are recorded below.
            let k = unsafe { self.stream.load_artifact(&artifact)? };
            if self.kernels.len() >= 512 {
                self.kernels.clear();
            }
            self.kernels.insert(key, k.clone());
            k
        };
        let mut bindings: Vec<_> = inputs
            .iter()
            .map(|t| Binding {
                slot: t.slot,
                offset: t.offset,
                bytes: t.len() * t.dtype.bytes(),
                access: hrx::Access::Read,
            })
            .collect();
        bindings.push(Binding {
            slot: output.slot,
            offset: output.offset,
            bytes: output.len() * output.dtype.bytes(),
            access: hrx::Access::Write,
        });
        self.commands.push(Command {
            #[cfg(feature = "bench-internals")]
            label: source
                .lines()
                .find_map(|line| {
                    line.strip_prefix("// ").and_then(|line| {
                        line.split_once(". Shapes").map(|(name, _)| name.to_owned())
                    })
                })
                .unwrap_or_else(|| "gemm_bf16".into()),
            kernel,
            constants,
            grid,
            bindings,
        });
        Ok(())
    }
    pub fn finish(&mut self) -> Result<GraphExec> {
        let mut graph = self.stream.access_graph()?;
        for command in &self.commands {
            let views = command
                .bindings
                .iter()
                .map(|b| {
                    Ok(self.allocations[b.slot]
                        .buffer
                        .binding()
                        .slice(b.offset, b.bytes)?
                        .access(b.access))
                })
                .collect::<Result<Vec<_>>>()?;
            // Safety: shape-specific kernels only read/write the declared exact spans.
            unsafe {
                graph.dispatch(
                    &command.kernel,
                    command.grid,
                    command.kernel.info().workgroup_size,
                    &command.constants,
                    &views,
                )?;
            }
        }
        let executable = graph.finish()?;
        self.commands.clear();
        Ok(executable)
    }
    #[cfg(feature = "bench-internals")]
    pub fn finish_profiled(&mut self) -> Result<GraphExec> {
        // Profiling serializes every dispatch. It diagnoses kernel cost but
        // must not replace the ordinary access graph in timing benchmarks.
        let mut graph = self.stream.owned_graph()?;
        let mut after = Vec::new();
        let mut labels = Vec::new();
        for command in &self.commands {
            let views = command
                .bindings
                .iter()
                .map(|b| {
                    self.allocations[b.slot]
                        .buffer
                        .binding()
                        .slice(b.offset, b.bytes)
                })
                .collect::<hrx::Result<Vec<_>>>()?;
            let node = unsafe {
                graph.dispatch(
                    &after,
                    &command.kernel,
                    command.grid,
                    command.kernel.info().workgroup_size,
                    &command.constants,
                    &views,
                )?
            };
            after.clear();
            after.push(node);
            labels.push(command.label.clone());
        }
        let graph = graph.finish_profiled(&labels)?;
        self.commands.clear();
        Ok(graph)
    }
    #[cfg(feature = "bench-internals")]
    pub fn profile(&mut self, graph: &mut GraphExec) -> Result<hrx::fabric::DeviceProfile> {
        Ok(self.stream.launch_profiled(graph)?)
    }
    pub fn run(&mut self, graph: &mut GraphExec) -> Result<()> {
        self.stream.launch(graph)?;
        self.stream.synchronize()?;
        Ok(())
    }
    pub fn read_f32(&mut self, t: &Tensor) -> Result<Vec<f32>> {
        let mut data = vec![0; t.len() * t.dtype.bytes()];
        self.stream.read_blocking(
            self.allocations[t.slot]
                .buffer
                .binding()
                .slice(t.offset, data.len())?,
            &mut data,
        )?;
        Ok(if t.dtype == DType::Bf16 {
            data.chunks_exact(2)
                .map(|v| half::bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f32())
                .collect()
        } else {
            data.chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect()
        })
    }
    pub fn reset_plan(&mut self) -> Result<()> {
        self.stream.synchronize()?;
        self.commands.clear();
        self.retained.clear();
        self.traces.clear();
        Ok(())
    }
    pub fn trace(&mut self, name: impl Into<String>, tensor: &Tensor) {
        if std::env::var_os("CLEF_TRACE_DIR").is_some() {
            self.traces.push((name.into(), tensor.clone()));
        }
    }
    pub fn dump_traces(&mut self) -> Result<()> {
        if let Some(path) = std::env::var_os("CLEF_TRACE_DIR") {
            let root = std::path::PathBuf::from(path);
            std::fs::create_dir_all(&root)?;
            let traces = self.traces.clone();
            let mut metadata = serde_json::Map::new();
            for (name, t) in traces {
                let data = self.read_f32(&t)?;
                std::fs::write(
                    root.join(format!("{name}.f32")),
                    bytemuck::cast_slice(&data),
                )?;
                metadata.insert(name, serde_json::json!([t.rows, t.cols]));
            }
            std::fs::write(
                root.join("shapes.json"),
                serde_json::to_vec_pretty(&metadata)?,
            )?;
        }
        Ok(())
    }
    pub fn fence(&mut self) -> hrx::Result<()> {
        self.stream.synchronize()
    }
    pub fn allocated_bytes(&self) -> usize {
        self.allocations.iter().map(|a| a.buffer.bytes()).sum()
    }
}
impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn engine() -> Result<Engine> {
        Engine::new(&ModelContext::new(Default::default())?)
    }
    fn input(e: &mut Engine, r: usize, c: usize, v: &[f32]) -> Result<Tensor> {
        let b: Vec<_> = v.iter().map(|v| half::bf16::from_f32(*v)).collect();
        e.input(r, c, DType::Bf16, bytemuck::cast_slice(&b))
    }
    fn finish(e: &mut Engine, t: &Tensor) -> Result<Vec<f32>> {
        let mut g = e.finish()?;
        e.run(&mut g)?;
        e.read_f32(t)
    }
    #[test]
    #[ignore = "requires gfx1151 and HRX native bundle"]
    fn operators() -> Result<()> {
        let mut e = engine()?;
        let x = input(&mut e, 2, 4, &[1., 2., 3., 4., -1., -2., 0., 1.])?;
        let w = input(&mut e, 1, 4, &[1.; 4])?;
        let b = input(&mut e, 1, 4, &[0.; 4])?;
        let norm = e.norm(&x, &w, Some(&b), 1e-5, false)?;
        let activation = e.unary(&x, "gelu")?;
        let tanh = e.unary(&x, "gelu_tanh")?;
        let rhs = input(
            &mut e,
            3,
            4,
            &[1., 0., 0., 1., 0., 1., 1., 0., 1., 1., 1., 1.],
        )?;
        let y = e.linear(&x, &rhs, None)?;
        let mut graph = e.finish()?;
        e.run(&mut graph)?;
        let actual = e.read_f32(&y)?;
        assert_eq!(actual, vec![5., 5., 10., 0., -2., -2.]);
        let norm = e.read_f32(&norm)?;
        assert!((norm[0] + 1.3416).abs() < 0.01);
        let a = e.read_f32(&activation)?;
        assert!((a[0] - 0.84134).abs() < 0.005);
        assert!((a[4] + 0.15865).abs() < 0.003);
        let a = e.read_f32(&tanh)?;
        assert!((a[0] - 0.84119).abs() < 0.005);
        Ok(())
    }
    #[test]
    #[ignore = "requires gfx1151 and HRX native bundle"]
    fn attention_rope_and_scratch() -> Result<()> {
        let mut e = engine()?;
        for d in [64, 72, 256] {
            let values: Vec<f32> = (0..3 * d).map(|i| ((i % 17) as f32 - 8.) / 16.).collect();
            let q = input(&mut e, 3, d, &values)?;
            let k = input(&mut e, 3, d, &values)?;
            let v = input(&mut e, 3, d, &vec![0.5; 3 * d])?;
            let a = e.attention(&q, &k, &v, 1, 1, true)?;
            let result = finish(&mut e, &a)?;
            assert!(result.iter().all(|v| (*v - 0.5).abs() < 1e-5));
            e.reset_plan()?;
        }
        let x = input(&mut e, 2, 64, &vec![1.; 128])?;
        let mut cs = vec![];
        for _ in 0..2 {
            cs.extend(vec![1.; 64]);
            cs.extend(vec![0.; 64]);
        }
        let cs = input(&mut e, 2, 128, &cs)?;
        let r = e.rope(&x, 1, 64, &cs)?;
        assert_eq!(finish(&mut e, &r)?, vec![1.; 128]);
        // A later host input must not overwrite an earlier temporary's backing.
        e.reset_plan()?;
        let z = e.unary(&x, "sigmoid")?;
        let first_slot = z.slot;
        drop(z);
        let later = input(&mut e, 2, 64, &vec![0.25; 128])?;
        assert_ne!(first_slot, later.slot);
        Ok(())
    }
    #[test]
    #[ignore = "requires gfx1151 and HRX native bundle"]
    fn delta_zero_and_reset() -> Result<()> {
        let mut e = engine()?;
        let qkv = input(&mut e, 3, 10240, &vec![0.; 3 * 10240])?;
        let a = input(&mut e, 3, 48, &vec![0.; 3 * 48])?;
        let b = a.clone();
        let al = input(&mut e, 1, 48, &[0.; 48])?;
        let dt = al.clone();
        let out = e.delta(&qkv, &a, &b, &al, &dt)?;
        let mut graph = e.finish()?;
        for _ in 0..2 {
            e.run(&mut graph)?;
            assert!(e.read_f32(&out)?.iter().all(|v| *v == 0.));
        }
        Ok(())
    }
}

#[cfg(test)]
mod differential {
    use super::*;
    use hrx::artifacts::safetensors::FileView;
    fn fixture(e: &mut Engine, f: &FileView, name: &str) -> Result<Tensor> {
        let t = f.get(name)?;
        let rows = if t.shape.len() > 1 { t.shape[0] } else { 1 };
        let cols = t.bytes.len() / 2 / rows;
        e.input(rows, cols, DType::Bf16, t.bytes)
    }
    fn compare(e: &mut Engine, t: &Tensor, f: &FileView, name: &str, tol: f32) -> Result<()> {
        let expected = f
            .get(name)?
            .bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect::<Vec<_>>();
        let mut graph = e.finish()?;
        e.run(&mut graph)?;
        let actual = e.read_f32(t)?;
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().chain(&expected).all(|v| v.is_finite()));
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max);
        eprintln!("{name}: max absolute error {error}");
        assert!(error <= tol, "{name}: {error} > {tol}");
        Ok(())
    }
    #[test]
    #[ignore = "requires gfx1151; compares captured upstream operator fixtures"]
    fn pytorch_nonzero() -> Result<()> {
        let mut e = Engine::new(&ModelContext::new(Default::default())?)?;
        let f = FileView::read("tests/fixtures/operators.safetensors")?;
        let x = fixture(&mut e, &f, "gemm_x")?;
        let w = fixture(&mut e, &f, "gemm_w")?;
        let y = e.linear(&x, &w, None)?;
        compare(&mut e, &y, &f, "gemm_y", 0.008)?;
        e.reset_plan()?;
        let q = fixture(&mut e, &f, "attn_q")?;
        let k = fixture(&mut e, &f, "attn_k")?;
        let v = fixture(&mut e, &f, "attn_v")?;
        let y = e.attention(&q, &k, &v, 24, 4, true)?;
        compare(&mut e, &y, &f, "attn_y", 0.008)?;
        e.reset_plan()?;
        let qkv = fixture(&mut e, &f, "delta_qkv")?;
        let a = fixture(&mut e, &f, "delta_a")?;
        let b = fixture(&mut e, &f, "delta_b")?;
        let al = fixture(&mut e, &f, "delta_al")?;
        let dt = fixture(&mut e, &f, "delta_dt")?;
        let y = e.delta(&qkv, &a, &b, &al, &dt)?;
        compare(&mut e, &y, &f, "delta_y", 0.002)?;
        e.reset_plan()?;
        let y = e.delta_chunked(&qkv, &a, &b, &al, &dt)?;
        compare(&mut e, &y, &f, "delta_y", 0.002)?;
        Ok(())
    }
}
