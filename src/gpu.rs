//! HRX graph execution. Tensor leases govern scratch reuse while
//! byte-range access declarations order reuse inside a recorded graph.
use crate::Result;
use anyhow::{Context, ensure};
use hrx::{
    Buffer, Constants, GraphExec, Kernel, Stream,
    inference::ModelContext,
    loom::{Compiler, Specialization},
};
use sha2::{Digest, Sha256};
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
    pub(crate) fn same_view(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self._lease, &other._lease)
            && self.slot == other.slot
            && self.offset == other.offset
            && self.rows == other.rows
            && self.cols == other.cols
            && self.dtype == other.dtype
    }
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
struct CompiledKernel {
    kernel: Kernel,
    grids: HashMap<Vec<u64>, [u32; 3]>,
}
pub(crate) struct Engine {
    #[cfg(feature = "bench-internals")]
    pub reference_kernels: crate::benchmarking::ReferenceKernels,
    pub quantized: crate::quantized::QuantizedProjections,
    #[cfg(feature = "bench-internals")]
    pub prepared_delta_max: usize,
    #[cfg(feature = "bench-internals")]
    pub prepared_delta_v8_max: usize,
    stream: Stream,
    compiler: Compiler,
    compile_reports: Option<std::path::PathBuf>,
    allocations: Vec<Allocation>,
    kernels: HashMap<String, CompiledKernel>,
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
            &[],
            &inputs.iter().map(|(_, tensor)| *tensor).collect::<Vec<_>>(),
            output,
        )
    }
    pub fn new(context: &ModelContext) -> Result<Self> {
        let stream = context.runtime().stream(None)?;
        ensure!(
            stream.target().as_str() == "gfx1151",
            "CLEF currently requires gfx1151"
        );
        let compiler = Compiler::for_stream(None, &stream)?;
        #[cfg(feature = "bench-internals")]
        if let Some(path) = std::env::var_os("CLEF_BENCH_RUNTIME_RECORD") {
            let path = std::path::PathBuf::from(path);
            if !path.exists() {
                std::fs::write(
                    path,
                    serde_json::to_vec_pretty(&serde_json::json!({
                        "compiler_identity": compiler.identity(),
                        "compiler_library": compiler.library_path(),
                        "target": compiler.target().as_str(),
                    }))?,
                )?;
            }
        }
        let compile_reports =
            std::env::var_os("CLEF_COMPILE_REPORT_DIR").map(std::path::PathBuf::from);
        if let Some(path) = &compile_reports {
            std::fs::create_dir_all(path)?;
        }
        Ok(Self {
            #[cfg(feature = "bench-internals")]
            reference_kernels: Default::default(),
            #[cfg(feature = "bench-internals")]
            quantized: crate::quantized::QuantizedProjections::from_env()?,
            #[cfg(not(feature = "bench-internals"))]
            quantized: Default::default(),
            #[cfg(feature = "bench-internals")]
            prepared_delta_max: std::env::var("CLEF_EXPERIMENT_DELTA_PREPARED_MAX")
                .ok()
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(0),
            #[cfg(feature = "bench-internals")]
            prepared_delta_v8_max: std::env::var("CLEF_EXPERIMENT_DELTA_PREPARED_V8_MAX")
                .ok()
                .map(|value| value.parse::<usize>())
                .transpose()?
                .unwrap_or(0),
            stream,
            compiler,
            compile_reports,
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
        let view = self.allocations[t.slot]
            .buffer
            .binding()
            .slice(t.offset, bytes.len())?;
        // HRX bounds reusable staging and splits large transfers internally.
        self.stream.upload(view, bytes)?;
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
        let elements = rows.checked_mul(cols).context("weight size overflow")?;
        ensure!(
            elements > 0
                && elements <= u32::MAX as usize
                && elements.checked_mul(DType::Bf16.bytes()) == Some(bytes.len()),
            "invalid weight extent"
        );
        crate::memory::before_allocation(bytes.len())?;
        // Immutable weights can initialize their final allocation directly.
        // No zero fill, temporary upload buffer, or per-chunk fence is needed.
        let buffer = self.stream.allocate_from(bytes)?;
        let lease = Rc::new(());
        let t = Tensor {
            slot: self.allocations.len(),
            offset: 0,
            rows,
            cols,
            dtype: DType::Bf16,
            _lease: lease.clone(),
        };
        self.allocations.push(Allocation {
            buffer,
            lease: Rc::downgrade(&lease),
            permanent: true,
            host_input: false,
        });
        self.uploaded += bytes.len() as u64;
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
        mut spec: Specialization,
        constants: Constants,
        workload: &[u64],
        inputs: &[&Tensor],
        output: &Tensor,
    ) -> Result<()> {
        let key = format!("{}{:?}", source, spec.configuration());
        if !self.kernels.contains_key(&key) {
            if self.compile_reports.is_some() {
                spec.set_report(hrx::loom::ReportMode::Details);
            }
            let artifact = self
                .compiler
                .module(source)
                .compile(&spec)
                .context("compiling CLEF kernel")?;
            if let Some(directory) = &self.compile_reports {
                let identity = format!("{:x}", Sha256::digest(key.as_bytes()));
                let report = serde_json::json!({
                    "source_sha256":format!("{:x}", Sha256::digest(source.as_bytes())),
                    "configuration":spec.configuration(),
                    "artifact":artifact.path(),
                    "compiler":artifact.compiler_identity(),
                    "target":artifact.target(),
                    "report":artifact.report(),
                });
                std::fs::write(
                    directory.join(format!("{identity}.json")),
                    serde_json::to_vec_pretty(&report)?,
                )?;
            }
            // Safety: callers supply trusted kernels matching these binding extents.
            // Production sources are crate-owned; experimental sources are bench-only.
            let k = unsafe { self.stream.load_artifact(&artifact)? };
            if self.kernels.len() >= 512 {
                self.kernels.clear();
            }
            self.kernels.insert(
                key.clone(),
                CompiledKernel {
                    kernel: k,
                    grids: HashMap::new(),
                },
            );
        }
        let compiled = self.kernels.get_mut(&key).unwrap();
        let grid = if let Some(grid) = compiled.grids.get(workload) {
            *grid
        } else {
            let grid = compiled.kernel.launch_config(workload)?.workgroup_count;
            if compiled.grids.len() >= 128 {
                compiled.grids.clear();
            }
            compiled.grids.insert(workload.to_vec(), grid);
            grid
        };
        let kernel = compiled.kernel.clone();
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
                .unwrap_or_else(|| {
                    format!(
                        "{} {}x{}x{}",
                        spec.symbol(),
                        output.rows,
                        inputs[0].cols,
                        output.cols
                    )
                }),
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
        self.quantized.clear_activation();
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
        self.quantized.clear_activation();
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
            data.as_chunks::<2>()
                .0
                .iter()
                .map(|v| half::bf16::from_bits(u16::from_le_bytes(*v)).to_f32())
                .collect()
        } else {
            data.as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect()
        })
    }
    pub fn reset_plan(&mut self) -> Result<()> {
        self.stream.synchronize()?;
        self.quantized.clear_activation();
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
    #[ignore = "requires gfx1151; checks shared-context accounting for initialized weights"]
    fn initialized_weights_share_context_budget() -> Result<()> {
        let manager = hrx::residency::ResidencyManager::new(16)?;
        let budget = manager.budget();
        let context = ModelContext::new(hrx::execution::RuntimeOptions {
            memory_budget: Some(budget.clone()),
            ..Default::default()
        })?;
        let mut e = Engine::new(&context)?;
        e.weight("first".into(), 1, 8, &[0; 16])?;
        assert_eq!(budget.reserved_bytes(), 16);
        assert!(e.weight("second".into(), 1, 1, &[0; 2]).is_err());
        assert_eq!(budget.reserved_bytes(), 16);
        drop(e);
        assert_eq!(budget.reserved_bytes(), 0);
        Ok(())
    }
    #[test]
    #[ignore = "requires gfx1151; checks direct weights, large uploads and subview offsets"]
    fn weight_and_input_transfers_preserve_bytes() -> Result<()> {
        let mut e = engine()?;
        let cols = (32 << 20) + 129;
        let data: Vec<_> = (0..cols * 2).map(|i| (i * 37 + 11) as u8).collect();
        e.weight("test".into(), 1, cols, &data)?;
        let weight = e.w("test")?;
        let mut actual = vec![0; data.len()];
        e.stream
            .read_blocking(e.allocations[weight.slot].buffer.binding(), &mut actual)?;
        assert_eq!(actual, data);
        let host = e.input(1, cols, DType::Bf16, &data)?;
        e.stream
            .read_blocking(e.allocations[host.slot].buffer.binding(), &mut actual)?;
        assert_eq!(actual, data);
        assert_eq!(e.uploaded, (data.len() * 2) as u64);

        let small = e.input(3, 4, DType::Bf16, &[0x33; 24])?;
        let row = small.slice_rows(1, 1);
        e.upload(&row, &[0x77; 8])?;
        let mut actual = [0; 24];
        e.stream
            .read_blocking(e.allocations[small.slot].buffer.binding(), &mut actual)?;
        assert_eq!(
            actual,
            [vec![0x33; 8], vec![0x77; 8], vec![0x33; 8]].concat()[..]
        );
        assert!(e.weight("invalid".into(), usize::MAX, 2, &[]).is_err());
        assert!(e.weight("invalid".into(), 1, 2, &[0; 3]).is_err());
        assert!(e.weight("invalid".into(), 0, 2, &[]).is_err());
        Ok(())
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
    #[ignore = "requires gfx1151; validates packed WMMA attention and tails"]
    fn tiled_attention_tails_and_replay() -> Result<()> {
        let mut e = engine()?;
        for (rows, keys, heads, kv, causal) in [
            (16, 16, 1, 1, true),
            (64, 64, 1, 1, true),
            (65, 79, 6, 2, false),
            (127, 127, 24, 4, true),
            (235, 235, 24, 4, true),
            (620, 620, 24, 4, true),
            (4031, 4031, 24, 4, true),
            (4032, 4032, 24, 4, true),
        ] {
            let values = |n, factor| {
                (0..n)
                    .map(|i| ((i * factor % 251) as f32 - 125.) / 64.)
                    .collect::<Vec<_>>()
            };
            let q = input(&mut e, rows, heads * 256, &values(rows * heads * 256, 17))?;
            let k = input(&mut e, keys, kv * 256, &values(keys * kv * 256, 31))?;
            let v = input(&mut e, keys, kv * 256, &values(keys * kv * 256, 43))?;
            let packed = e.transpose_attention_v(&v)?;
            let mut pack_graph = e.finish()?;
            e.run(&mut pack_graph)?;
            let pack_values = e.read_f32(&packed)?;
            let v_values = e.read_f32(&v)?;
            for col in 0..v.cols {
                for row in 0..packed.cols {
                    let expected = if row < v.rows {
                        v_values[row * v.cols + col]
                    } else {
                        0.
                    };
                    assert_eq!(
                        pack_values[col * packed.cols + row],
                        expected,
                        "transpose at {row},{col}"
                    );
                }
            }
            let old = e.attention_impl::<false>(&q, &k, &v, heads, kv, causal)?;
            let mut reference = e.finish()?;
            let new = e.attention_tiled(&q, &k, &v, heads, kv, causal)?;
            let mut candidate = e.finish()?;
            e.run(&mut reference)?;
            let expected = e.read_f32(&old)?;
            e.run(&mut candidate)?;
            let actual = e.read_f32(&new)?;
            assert!(actual.iter().chain(&expected).all(|x| x.is_finite()));
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "tiled attention {rows}x{keys}, heads={heads}/{kv}, causal={causal}: max error={error}"
            );
            if rows <= 127 {
                let qf = e.read_f32(&q)?;
                let kf = e.read_f32(&k)?;
                for row in [0, rows / 2, rows - 1] {
                    for head in [0, heads - 1] {
                        let kh = head / (heads / kv);
                        let scores: Vec<f64> = (0..if causal { row + 1 } else { keys })
                            .map(|t| {
                                (0..256)
                                    .map(|d| {
                                        f64::from(qf[row * heads * 256 + head * 256 + d])
                                            * f64::from(kf[t * kv * 256 + kh * 256 + d])
                                    })
                                    .sum::<f64>()
                                    / 16.
                            })
                            .collect();
                        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let denom = scores.iter().map(|s| (s - max).exp()).sum::<f64>();
                        for col in [0, 127, 255] {
                            let oracle = scores
                                .iter()
                                .enumerate()
                                .map(|(t, s)| {
                                    (s - max).exp()
                                        * f64::from(v_values[t * kv * 256 + kh * 256 + col])
                                })
                                .sum::<f64>()
                                / denom;
                            let index = row * heads * 256 + head * 256 + col;
                            assert!(
                                (f64::from(actual[index]) - oracle).abs() <= 0.008,
                                "FP64 attention mismatch at {row}/{head}/{col}"
                            );
                        }
                    }
                }
            }
            assert!(error <= 0.008, "packed attention error {error}");
            e.run(&mut candidate)?;
            assert_eq!(e.read_f32(&new)?, actual, "attention replay changed output");
            e.reset_plan()?;
        }
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
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::bf16::from_bits(u16::from_le_bytes(*b)).to_f32())
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
        for wide in [false, true] {
            let y = if wide {
                e.linear_impl::<true>(&x, &w, None)?
            } else {
                e.linear_impl::<false>(&x, &w, None)?
            };
            compare(&mut e, &y, &f, "gemm_y", 0.008)?;
        }
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
