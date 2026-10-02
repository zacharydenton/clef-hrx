use crate::{
    EncodeOptions, Encoder, Request,
    encoding::{canonical_json, render},
    media::{MediaOptions, VideoFrames, prepare, sample_indices, smart_resize},
    schema::{Question, answer, softmax},
};
use indexmap::IndexMap;
use serde_json::{Value, json};

#[test]
fn canonical_json_and_option_order() {
    let v: Value =
        serde_json::from_str(r#"{"z":[1e-6,1e-4,1e16,-0.0],"å":"😀","a":{"z":2,"a":1}}"#).unwrap();
    assert_eq!(
        canonical_json(&v),
        "{\"a\":{\"a\":1,\"z\":2},\"z\":[1e-06,0.0001,1e+16,-0.0],\"å\":\"😀\"}"
    );
    assert_eq!(render(&json!("raw\nstate")), "raw\nstate");
    let q: Question =
        serde_json::from_value(json!({"type":"choice","criteria":{"z":"last","a":"first"}}))
            .unwrap();
    assert_eq!(
        q.options()
            .unwrap()
            .iter()
            .map(|v| v.0.as_str())
            .collect::<Vec<_>>(),
        ["a", "z"]
    );
    let p: IndexMap<_, _> = [("a".into(), 0.5), ("z".into(), 0.5)].into_iter().collect();
    assert_eq!(answer(&q, &p).unwrap()["choice"], "z");
}
#[test]
fn answers_and_invalid_inputs() {
    let q: Question =
        serde_json::from_value(json!({"type":"score","criteria":["low","medium","high"]})).unwrap();
    let p: IndexMap<_, _> = [("0".into(), 0.25), ("1".into(), 0.25), ("2".into(), 0.5)]
        .into_iter()
        .collect();
    let a = answer(&q, &p).unwrap();
    assert_eq!(a["score"], 1.25);
    assert_eq!(a["confidence"], 0.5);
    assert!(softmax(&[f32::NAN]).is_err());
    assert!(softmax(&[]).is_err());
    assert_eq!(softmax(&[1000., 1000.]).unwrap(), vec![0.5, 0.5]);
    for bad in [
        json!({"type":"choice","criteria":[]}),
        json!({"type":"score","criteria":{}}),
        json!({"type":"choice","criteria":{}}),
    ] {
        assert!(
            serde_json::from_value::<Question>(bad)
                .unwrap()
                .options()
                .is_err()
        );
    }
    assert!(serde_json::from_value::<Request>(json!({"model":"clef","questions":{}})).is_err());
}
#[test]
fn media_layout_temporal_padding_and_positions() {
    let im = image::RgbImage::from_fn(32, 32, |x, y| image::Rgb([x as u8, y as u8, 255]));
    let options = MediaOptions {
        min_pixels: Some(1024),
        max_pixels: Some(8192),
        do_sample_frames: Some(false),
        ..Default::default()
    };
    let video = VideoFrames {
        frames: vec![im.clone(); 3],
        fps: 2.,
    };
    let mut media = prepare(&[im], &[video], &options, 100).unwrap();
    assert_eq!(media.items[0].grid, [1, 2, 2]);
    assert_eq!(media.items[1].grid, [2, 2, 2]);
    assert_eq!(media.items[1].timestamps, vec![0.25, 1.]);
    assert_eq!(
        &media.items[1].patches[0..1536 * 4],
        &media.items[1].patches[1536 * 4..]
    );
    media
        .bind_tokens(
            &[
                248053, 248056, 248054, 100, 248053, 248057, 248054, 100, 248053, 248057, 248054,
            ],
            5,
        )
        .unwrap();
    let p = media.position_ids(20).unwrap();
    assert_eq!(p.len(), 20);
    assert_eq!(p[6], [6, 6, 6]);
    assert_eq!(
        sample_indices(10, 10., &MediaOptions::default()).unwrap(),
        [0, 3, 6, 9]
    );
    assert!(smart_resize(1, 1000, 1, 1024, 2048, false).is_err());
    assert!(
        prepare(
            &[image::RgbImage::new(100, 100)],
            &[],
            &MediaOptions::default(),
            1
        )
        .is_err()
    );
}
#[test]
#[ignore = "requires the pinned tokenizer in the shared HF cache"]
fn release_encoding_parity() {
    let encoder = Encoder::load(
        crate::checkpoint::Source {
            offline: true,
            ..Default::default()
        }
        .resolve("tokenizer.json")
        .unwrap(),
    )
    .unwrap();
    let fixtures: Vec<Value> =
        serde_json::from_str(include_str!("../tests/fixtures/encoding.json")).unwrap();
    for f in fixtures {
        let req: Request = serde_json::from_value(f["request"].clone()).unwrap();
        let options = EncodeOptions {
            max_length: 512,
            max_state_tokens: f["max_state_tokens"].as_u64().map(|v| v as usize),
        };
        let actual = encoder.encode_record(&req, options).unwrap();
        assert_eq!(
            serde_json::to_value(&actual.input_ids).unwrap(),
            f["input_ids"]
        );
        for (a, b) in actual
            .questions
            .iter()
            .zip(f["questions"].as_array().unwrap())
        {
            assert_eq!(json!(a.question_span), b["question_span"]);
            assert_eq!(json!(a.option_spans), b["option_spans"]);
            assert_eq!(json!(a.option_ids), b["option_ids"]);
            assert_eq!(json!(a.question_type.id()), b["question_type"]);
        }
    }
    let req: Request = serde_json::from_str(include_str!("../examples/invoice.json")).unwrap();
    assert!(
        encoder
            .encode_record(
                &req,
                EncodeOptions {
                    max_length: 1,
                    max_state_tokens: None
                }
            )
            .is_err()
    );
}

#[test]
fn processor_patch_parity() {
    use hrx::artifacts::safetensors::FileView;
    let f = FileView::read("tests/fixtures/media.safetensors").unwrap();
    for (h, w) in [(32, 32), (73, 117), (513, 777)] {
        let im = image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([
                ((x * 7 + y * 3) % 256) as u8,
                ((x * 11 + y * 5) % 256) as u8,
                ((x + y * 13) % 256) as u8,
            ])
        });
        let options = MediaOptions {
            min_pixels: Some(1024),
            max_pixels: Some(262144),
            ..Default::default()
        };
        let media = prepare(&[im], &[], &options, 16384).unwrap();
        let expected = f.get(&format!("image_{h}_{w}")).unwrap();
        let expected: Vec<_> = expected
            .bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let actual = &media.items[0].patches;
        assert_eq!(actual.len(), expected.len());
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max);
        let differing = actual
            .iter()
            .zip(&expected)
            .filter(|(a, b)| (*a - *b).abs() > 1e-5)
            .count();
        eprintln!(
            "processor {h}x{w}: max error {error}, differing {differing}/{}",
            actual.len()
        );
        assert!(error <= 1e-6);
    }
}

#[test]
fn local_checkpoint_validation_and_offline_failures() {
    let root = tempfile::tempdir().unwrap();
    let source = crate::checkpoint::Source {
        directory: Some(root.path().into()),
        offline: true,
    };
    assert!(source.resolve("../outside").is_err());
    assert!(source.resolve("config.json").is_err());
    for (name, bytes) in [
        ("config.json", include_str!("../reference/config.json")),
        (
            "joint_head_config.json",
            include_str!("../reference/joint_head_config.json"),
        ),
        (
            "processor_config.json",
            include_str!("../reference/processor_config.json"),
        ),
        (
            "model.safetensors.index.json",
            include_str!("../reference/model.safetensors.index.json"),
        ),
    ] {
        std::fs::write(root.path().join(name), bytes).unwrap();
    }
    let info = source.inspect().unwrap();
    assert_eq!(info.backbone_layers, 64);
    assert_eq!(info.weight_bytes, 54969569768);
    assert_eq!(info.resident_weight_bytes, 49883976168);
    let mut config: Value = serde_json::from_str(include_str!("../reference/config.json")).unwrap();
    config["text_config"]["hidden_size"] = json!(42);
    std::fs::write(
        root.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    assert!(source.inspect().is_err());
}

#[test]
#[ignore = "requires the pinned tokenizer in the shared HF cache"]
fn multimodal_token_and_rope_parity() {
    let encoder = Encoder::load(
        crate::checkpoint::Source {
            offline: true,
            ..Default::default()
        }
        .resolve("tokenizer.json")
        .unwrap(),
    )
    .unwrap();
    let fixtures: Vec<Value> =
        serde_json::from_slice(&std::fs::read("tests/fixtures/multimodal_encoding.json").unwrap())
            .unwrap();
    for f in fixtures {
        let request: Request = serde_json::from_value(f["request"].clone()).unwrap();
        let images = vec![
            image::RgbImage::from_pixel(64, 32, image::Rgb([12, 30, 240]));
            f["images"].as_u64().unwrap() as usize
        ];
        let videos = vec![
            VideoFrames {
                frames: (0..3)
                    .map(|i| image::RgbImage::from_pixel(
                        64,
                        32,
                        image::Rgb([i * 50, 0, 255 - i * 50])
                    ))
                    .collect(),
                fps: 2.
            };
            f["videos"].as_u64().unwrap() as usize
        ];
        let media = prepare(
            &images,
            &videos,
            &MediaOptions {
                min_pixels: Some(1024),
                max_pixels: Some(8192),
                do_sample_frames: Some(false),
                ..Default::default()
            },
            16384,
        )
        .unwrap();
        let encoded = encoder
            .encode_with_media(&request, EncodeOptions::default(), media)
            .unwrap();
        assert_eq!(json!(encoded.input_ids), f["input_ids"]);
        assert_eq!(json!(encoded.position_ids), f["position_ids"]);
    }
}

#[test]
#[ignore = "requires cached tokenizer and ffmpeg; does not load model weights"]
fn corpus_encoding_parity() -> crate::Result<()> {
    let source = crate::checkpoint::Source {
        offline: true,
        ..Default::default()
    };
    let encoder = Encoder::load(source.resolve("tokenizer.json")?)?;
    for case in 0..8 {
        let request: Request = serde_json::from_slice(&std::fs::read(format!(
            "tests/fixtures/corpus/{case}.json"
        ))?)?;
        let reference: Value = serde_json::from_slice(&std::fs::read(format!(
            "tests/fixtures/corpus/{case}.reference.json"
        ))?)?;
        let encoded = encoder.encode_record(
            &request,
            EncodeOptions {
                max_length: 1536,
                max_state_tokens: None,
            },
        )?;
        assert_eq!(
            json!(encoded.input_ids),
            reference["input_ids"],
            "case {case}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires the pinned checkpoint in the shared HF cache; reads only selected embedding rows"]
fn cached_embedding_row_parity() -> crate::Result<()> {
    let source = crate::checkpoint::Source {
        offline: true,
        ..Default::default()
    };
    // Cached checkpoint files remain immutable for the lifetime of these mappings.
    let checkpoint = unsafe { crate::checkpoint::Checkpoint::open(&source)? };
    let ids = [248319, 0, 151643, 1, 248319];
    let mut cases = Vec::new();
    for name in crate::embedding::TABLES {
        let tensor = checkpoint.tensor(name)?;
        let row_bytes = tensor.shape[1] * 2;
        let expected: Vec<u8> = ids
            .iter()
            .flat_map(|&id| {
                let start = id as usize * row_bytes;
                tensor.bytes[start..start + row_bytes].iter().copied()
            })
            .collect();
        cases.push((checkpoint.embedding_rows(name)?, expected));
    }
    drop(checkpoint);
    // Row readers remain usable after the loader releases every mapping.
    for (rows, expected) in cases {
        assert_eq!(rows.read(&ids)?, expected);
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151, cached full checkpoint, ffmpeg, and at least 50 GiB available RAM; run alone"]
fn full_checkpoint_corpus() -> crate::Result<()> {
    use crate::{ClefModel, LoadOptions, checkpoint::Source};
    let mut model = ClefModel::load(LoadOptions {
        source: Source {
            directory: std::env::var_os("CLEF_MODEL_DIR").map(Into::into),
            offline: true,
        },
        memory_budget_bytes: 60usize << 30,
        encoding: EncodeOptions {
            max_length: 1536,
            max_state_tokens: None,
        },
        ..Default::default()
    })?;
    eprintln!("checkpoint loaded; starting full corpus");
    let mut first = None;
    let mut failures = Vec::new();
    for case in [0, 1, 2, 3, 4, 5, 6, 7, 0, 0] {
        eprintln!("case {case}: starting");
        let request: Request = serde_json::from_slice(&std::fs::read(format!(
            "tests/fixtures/corpus/{case}.json"
        ))?)?;
        let reference: Value = serde_json::from_slice(&std::fs::read(format!(
            "tests/fixtures/corpus/{case}.reference.json"
        ))?)?;
        let encoded = model.encode_record(&request)?;
        assert_eq!(
            json!(encoded.input_ids),
            reference["input_ids"],
            "case {case}: tokens"
        );
        let prediction = model.infer(&request)?;
        assert_eq!(
            prediction.questions.len(),
            reference["questions"].as_array().unwrap().len()
        );
        let mut maximum = 0.0f32;
        for (actual, expected) in prediction
            .questions
            .iter()
            .zip(reference["questions"].as_array().unwrap())
        {
            assert_eq!(json!(actual.question_id), expected["question_id"]);
            assert_eq!(json!(actual.option_ids), expected["option_ids"]);
            let probabilities: Vec<f32> =
                serde_json::from_value(expected["probabilities"].clone())?;
            assert_eq!(actual.probabilities.len(), probabilities.len());
            for (a, b) in actual.probabilities.iter().zip(&probabilities) {
                assert!(a.is_finite());
                maximum = maximum.max((a - b).abs());
            }
            let winner = |values: &[f32]| {
                values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap()
                    .0
            };
            if winner(&actual.probabilities) != winner(&probabilities) {
                failures.push(format!(
                    "case {case}: question {} winner",
                    actual.question_id
                ));
            }
            if actual
                .probabilities
                .iter()
                .zip(&probabilities)
                .any(|(a, b)| (a - b).abs() > 0.003)
            {
                eprintln!(
                    "case {case}: question {}, actual {:?}, reference {:?}",
                    actual.question_id, actual.probabilities, probabilities
                );
            }
        }
        if maximum > 0.003 {
            failures.push(format!("case {case}: probability error {maximum}"));
        }
        eprintln!(
            "case {case}: tokens={}, max_probability_error={maximum:.8}, inference_ms={:.2}, allocated_GiB={:.2}",
            prediction.input_tokens,
            prediction.timings.inference_ms,
            prediction.timings.allocated_bytes as f64 / (1u64 << 30) as f64
        );
        // A changed-input replay and an identical-input replay must reset state.
        if case == 0 {
            let actual = serde_json::to_value(&prediction.questions)?;
            if let Some(expected) = &first {
                assert_eq!(&actual, expected);
            } else {
                first = Some(actual);
            }
        }
    }
    assert!(
        failures.is_empty(),
        "full corpus failures: {}",
        failures.join("; ")
    );
    Ok(())
}
