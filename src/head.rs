use crate::{
    Result,
    encoding::EncodedRecord,
    gpu::{DType, Engine, Tensor},
};

impl Engine {
    pub fn copy_into(&mut self, x: &Tensor, out: &Tensor) -> Result<()> {
        anyhow::ensure!(x.len() == out.len() && x.dtype == out.dtype, "copy shape");
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/copy.loom")
            ),
            out.len(),
            &[("x", x)],
            out,
            &[],
        )
    }
    pub fn cat_rows(&mut self, inputs: &[Tensor]) -> Result<Tensor> {
        anyhow::ensure!(
            !inputs.is_empty()
                && inputs
                    .iter()
                    .all(|t| t.cols == inputs[0].cols && t.dtype == inputs[0].dtype),
            "concat dimensions"
        );
        let out = self.alloc(
            inputs.iter().map(|t| t.rows).sum(),
            inputs[0].cols,
            inputs[0].dtype,
        )?;
        let mut offset = 0;
        for x in inputs {
            self.copy_into(x, &out.slice_rows(offset, x.rows))?;
            offset += x.rows;
        }
        Ok(out)
    }
    fn mha(&mut self, q: &Tensor, memory: &Tensor, name: &str) -> Result<Tensor> {
        let w = self.w(&format!("{name}.in_proj_weight"))?;
        let b = self.w(&format!("{name}.in_proj_bias"))?.reshape(3072, 1);
        let qw = w.slice_rows(0, 1024);
        let kw = w.slice_rows(1024, 1024);
        let vw = w.slice_rows(2048, 1024);
        let qb = b.slice_rows(0, 1024).reshape(1, 1024);
        let kb = b.slice_rows(1024, 1024).reshape(1, 1024);
        let vb = b.slice_rows(2048, 1024).reshape(1, 1024);
        let q = self.linear(q, &qw, Some(&qb))?;
        let k = self.linear(memory, &kw, Some(&kb))?;
        let v = self.linear(memory, &vw, Some(&vb))?;
        let a = self.attention(&q, &k, &v, 16, 16, false)?;
        self.project(&a, &format!("{name}.out_proj"), true)
    }
    fn residual_features(&mut self, fields: &Tensor, options: &Tensor) -> Result<Tensor> {
        let width = fields.cols;
        let out = self.alloc(fields.rows, width * 4, DType::Bf16)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/residual_features.loom")
            ),
            out.len(),
            &[("f", fields), ("o", options)],
            &out,
            &[
                ("stride", (width * 4).to_string()),
                ("width", (width).to_string()),
            ],
        )?;
        Ok(out)
    }
    fn score(
        &mut self,
        anchors: &Tensor,
        lexical: &Tensor,
        fields: &Tensor,
        options: &Tensor,
        residual: &Tensor,
    ) -> Result<Tensor> {
        let ps = self.w("head.prior_logit_scale")?;
        let js = self.w("head.joint_logit_scale")?;
        let gate = self.w("head.residual_gate")?;
        let out = self.alloc(fields.rows, 1, DType::F32)?;
        self.dispatch(
            concat!(
                include_str!("../kernels/common.loom"),
                include_str!("../kernels/score.loom")
            ),
            out.len(),
            &[
                ("anchors", anchors),
                ("lexical", lexical),
                ("fields", fields),
                ("options", options),
                ("residual", residual),
                ("ps", &ps),
                ("js", &js),
                ("gate", &gate),
            ],
            &out,
            &[],
        )?;
        Ok(out)
    }
}

pub(crate) fn forward(e: &mut Engine, hidden: &Tensor, record: &EncodedRecord) -> Result<Tensor> {
    let hidden = e.named_norm(hidden, "head.hidden_norm", true, 1e-5)?;
    let memory = e.project(&hidden, "head.memory_projection", false)?;
    let global = hidden.slice_rows(hidden.rows - 1, 1);
    let qspans: Vec<_> = record.questions.iter().map(|q| q.question_span).collect();
    let ospans: Vec<_> = record
        .questions
        .iter()
        .flat_map(|q| q.option_spans.iter().copied())
        .collect();
    let questions = e.pool_spans(&hidden, &qspans)?;
    let contexts = e.pool_spans(&hidden, &ospans)?;
    let mut lexical_ids = Vec::new();
    let mut lexical_spans = Vec::new();
    for [start, end] in &ospans {
        let s = lexical_ids.len();
        lexical_ids.extend_from_slice(&record.input_ids[*start..*end]);
        lexical_spans.push([s, lexical_ids.len()]);
    }
    let ids = e.input(
        lexical_ids.len(),
        1,
        DType::U32,
        bytemuck::cast_slice(&lexical_ids),
    )?;
    let embeddings = e.w("lm_head.weight")?;
    let lexical_tokens = e.gather(&embeddings, &ids)?;
    let lexical = e.pool_spans(&lexical_tokens, &lexical_spans)?;
    drop(lexical_tokens);
    let repeats: Vec<u32> = record
        .questions
        .iter()
        .enumerate()
        .flat_map(|(i, q)| std::iter::repeat_n(i as u32, q.option_ids.len()))
        .collect();
    let repeated_ids = e.input(repeats.len(), 1, DType::U32, bytemuck::cast_slice(&repeats))?;
    let qp = e.project(&questions, "head.option_question_projection", false)?;
    let qp = e.gather(&qp, &repeated_ids)?;
    let cp = e.project(&contexts, "head.option_context_projection", false)?;
    let lp = e.project(&lexical, "head.option_lexical_projection", false)?;
    let mut options = e.binary(&cp, &lp, "addf")?;
    options = e.binary(&options, &qp, "addf")?;
    for i in 0..2 {
        let p = format!("head.evidence_layers.{i}");
        let q = e.named_norm(&options, &format!("{p}.query_norm"), true, 1e-5)?;
        let m = e.named_norm(&memory, &format!("{p}.memory_norm"), true, 1e-5)?;
        let a = e.mha(&q, &m, &format!("{p}.attention"))?;
        options = e.binary(&options, &a, "addf")?;
        let f = e.named_norm(&options, &format!("{p}.feedforward_norm"), true, 1e-5)?;
        let f = e.project(&f, &format!("{p}.feedforward.0"), true)?;
        let f = e.unary(&f, "gelu")?;
        let f = e.project(&f, &format!("{p}.feedforward.3"), true)?;
        options = e.binary(&options, &f, "addf")?;
    }
    let base = e.project(&questions, "head.question_projection", false)?;
    let mut summaries = Vec::new();
    let mut off = 0;
    for (i, q) in record.questions.iter().enumerate() {
        let o = options.slice_rows(off, q.option_ids.len());
        off += q.option_ids.len();
        summaries.push(e.attention(&base.slice_rows(i, 1), &o, &o, 1, 1, false)?);
    }
    let summaries = e.cat_rows(&summaries)?;
    let summaries = e.named_norm(&summaries, "head.option_summary_norm", true, 1e-5)?;
    let gp = e.project(&global, "head.global_projection", false)?;
    let types: Vec<_> = record
        .questions
        .iter()
        .map(|q| q.question_type.id())
        .collect();
    let type_ids = e.input(types.len(), 1, DType::U32, bytemuck::cast_slice(&types))?;
    let type_w = e.w("head.type_embedding.weight")?;
    let tp = e.gather(&type_w, &type_ids)?;
    let mut fields = e.binary(&base, &summaries, "addf")?;
    fields = e.binary(&fields, &gp, "addf")?;
    fields = e.binary(&fields, &tp, "addf")?;
    for i in 0..4 {
        let p = format!("head.layers.{i}");
        let n = e.named_norm(&fields, &format!("{p}.norm1"), true, 1e-5)?;
        let a = e.mha(&n, &n, &format!("{p}.self_attn"))?;
        fields = e.binary(&fields, &a, "addf")?;
        let n = e.named_norm(&fields, &format!("{p}.norm2"), true, 1e-5)?;
        let a = e.mha(&n, &memory, &format!("{p}.multihead_attn"))?;
        fields = e.binary(&fields, &a, "addf")?;
        let n = e.named_norm(&fields, &format!("{p}.norm3"), true, 1e-5)?;
        let f = e.project(&n, &format!("{p}.linear1"), true)?;
        let f = e.unary(&f, "gelu")?;
        let f = e.project(&f, &format!("{p}.linear2"), true)?;
        fields = e.binary(&fields, &f, "addf")?;
    }
    let fields = e.named_norm(&fields, "head.field_norm", true, 1e-5)?;
    let fields = e.gather(&fields, &repeated_ids)?;
    let options = e.named_norm(&options, "head.option_norm", true, 1e-5)?;
    let features = e.residual_features(&fields, &options)?;
    let residual = e.project(&features, "head.residual_scorer.0", true)?;
    let residual = e.unary(&residual, "gelu")?;
    let residual = e.project(&residual, "head.residual_scorer.3", true)?;
    let anchors = e.binary(&questions, &global, "addf")?;
    let anchors = e.gather(&anchors, &repeated_ids)?;
    e.score(&anchors, &lexical, &fields, &options, &residual)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hrx::artifacts::safetensors::FileView;
    #[test]
    #[ignore = "requires gfx1151, trained head in HF cache, and head fixtures"]
    fn trained_head_parity() -> Result<()> {
        let mut e = Engine::new(&hrx::inference::ModelContext::new(Default::default())?)?;
        let source = crate::checkpoint::Source {
            offline: true,
            ..Default::default()
        };
        let weights = FileView::read(source.resolve("joint_head.safetensors")?)?;
        for name in weights.names() {
            let t = weights.get(name)?;
            let rows = t.shape.first().copied().unwrap_or(1);
            let cols = t.shape.iter().skip(1).product();
            e.weight(format!("head.{name}"), rows, cols, t.bytes)?;
        }
        let fixture = FileView::read("tests/fixtures/head.safetensors")?;
        e.weight(
            "lm_head.weight".into(),
            32,
            5120,
            fixture.get("embeddings")?.bytes,
        )?;
        let hidden = e.input(24, 5120, DType::Bf16, fixture.get("hidden")?.bytes)?;
        let data: serde_json::Value =
            serde_json::from_slice(&std::fs::read("tests/fixtures/head.json")?)?;
        let questions = data["questions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                let mut v = v.clone();
                v["question_type"] = serde_json::json!(
                    ["noul", "choice", "score"][v["question_type"].as_u64().unwrap() as usize]
                );
                serde_json::from_value(v).unwrap()
            })
            .collect();
        let record = EncodedRecord {
            input_ids: (0..24).collect(),
            questions,
            position_ids: (0..24).map(|v| [v; 3]).collect(),
            media: Default::default(),
        };
        let output = forward(&mut e, &hidden, &record)?;
        let mut graph = e.finish()?;
        e.run(&mut graph)?;
        let actual = e.read_f32(&output)?;
        let expected: Vec<_> = fixture
            .get("logits")?
            .bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        eprintln!("head actual {actual:?}; reference {expected:?}");
        let mut off = 0;
        for q in record.questions {
            let n = q.option_ids.len();
            let a = crate::schema::softmax(&actual[off..off + n])?;
            let b = crate::schema::softmax(&expected[off..off + n])?;
            assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() <= 0.003));
            off += n;
        }
        Ok(())
    }
}
