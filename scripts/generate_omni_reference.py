#!/usr/bin/env python3
"""Generate native Omni fixtures using Cloudflare's pinned release as the oracle.

Python is used for verification only; the Rust CLI does not invoke this script.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REVISION = "0db1cd2607d76a7bdb2a382f659e7b313079f84b"
RELEASE_SHA256 = "21d05ebb8cbea26a26af65eca3d16cf4b69bf80a320e2d5b203e86f0d632a670"


def release_module(directory):
    path = directory / "joint_schema_model.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != RELEASE_SHA256:
        raise ValueError("release code differs from pinned Omni revision")
    spec = importlib.util.spec_from_file_location("omni_release", path)
    release = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = release
    spec.loader.exec_module(release)
    return release


def operator_fixtures(output):
    import torch
    import torch.nn.functional as F
    x = torch.tensor([(i * 7 % 47 - 23) / 32 for i in range(7 * 9 * 3)], dtype=torch.bfloat16).reshape(1,7,9,3)
    w = torch.tensor([(i * 11 % 19 - 9) / 32 for i in range(4 * 3 * 3 * 3)], dtype=torch.bfloat16).reshape(4,3,3,3)
    b = torch.tensor([0.125,-0.25,0.375,0.], dtype=torch.bfloat16)
    y = F.gelu(F.conv2d(x.permute(0,3,1,2), w, b, stride=2, padding=1)).permute(0,2,3,1)
    result = {"x":x.float().flatten().tolist(),"w":w.float().flatten().tolist(),"b":b.float().tolist(),"y":y.float().flatten().tolist()}
    (output / "conv.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    q = torch.tensor([(i * 17 % 67 - 33) / 32 for i in range(7*4*128)],dtype=torch.bfloat16).reshape(7,4,128)
    k = torch.tensor([(i * 11 % 43 - 21) / 32 for i in range(7*2*128)],dtype=torch.bfloat16).reshape(7,2,128)
    v = torch.tensor([(i * 13 % 59 - 29) / 32 for i in range(7*2*128)],dtype=torch.bfloat16).reshape(7,2,128)
    result = {name:t.float().flatten().tolist() for name,t in [("q",q),("k",k),("v",v)]}
    for causal in [False, True]:
        y=F.scaled_dot_product_attention(q.permute(1,0,2)[None],k.permute(1,0,2)[None],v.permute(1,0,2)[None],is_causal=causal,enable_gqa=True)
        result[str(causal)]=y[0].permute(1,0,2).float().flatten().tolist()
    (output / "attention.json").write_text(json.dumps(result,separators=(",", ":"))+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
        Qwen3OmniMoeThinkerTextRMSNorm,
        Qwen3OmniMoeThinkerTextRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    release = release_module(args.model_dir)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    cfg = AutoConfig.from_pretrained(args.model_dir, local_files_only=True).thinker_config
    with torch.device("meta"):
        thinker = Qwen3OmniMoeThinkerForConditionalGeneration(cfg)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"model": "Cloudflare/clef-omni", "revision": REVISION,
                "release_code_sha256": RELEASE_SHA256, "torch": torch.__version__,
                "transformers": transformers.__version__, "device": "cpu"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Multiple clip lengths catch shared padding, STFT tails and CNN token counts.
    clips = [np.array([(((i * 17) % 101) - 50) / 256 for i in range(n)], dtype=np.float32)
             for n in [4801, 16321]]
    features = processor.feature_extractor(clips, sampling_rate=16000, padding=True,
                                           truncation=False, return_attention_mask=True,
                                           return_tensors="pt")
    mel = []
    for i, n in enumerate(features["attention_mask"].sum(-1).tolist()):
        mel.append({"samples": len(clips[i]), "frames": n,
                    "mel": features["input_features"][i, :, :n].flatten().tolist()})
    (args.output / "mel.json").write_text(json.dumps(mel, separators=(",", ":")) + "\n")
    import wave
    import subprocess
    import struct
    audio_path = args.output / "tone.wav"
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"".join(struct.pack("<h", ((i * 17) % 101 - 50) * 128) for i in range(16321)))
    video_path = args.output / "colors-audio.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", "tests/fixtures/corpus/colors.mkv",
                    "-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                    "-c:a", "pcm_s16le", "-shortest", str(video_path)], check=True)
    # Real media paths plus audio arrays test release encoding and synchronized
    # positions independently of native media code. File-based audio comes below.
    requests = [
        {"model": "clef-omni", "state": "The invoice is overdue.", "questions": {
            "status": {"type": "choice", "criteria": {"paid": "Paid", "overdue": "Overdue"}},
            "late": {"type": "noul"}, "urgency": {"type": "score", "criteria": ["Low", "High"]}}},
        {"model": "clef-omni", "state": "Look at the picture.",
         "images": ["tests/fixtures/corpus/red.png"],
         "questions": {"red": {"type": "noul", "instructions": "Is the picture red?"}}},
        {"model": "clef-omni", "state": "Watch the clip.",
         "videos": ["tests/fixtures/corpus/colors.mkv"],
         "questions": {"changes": {"type": "noul"}}},
    ]
    requests.extend([
        {"model": "clef-omni", "state": "Listen to the recording.", "audio": [str(audio_path)],
         "questions": {"sound": {"type": "noul", "instructions": "Is there audible sound?"}}},
        {"model": "clef-omni", "state": "Watch and listen.", "videos": [str(video_path)],
         "questions": {"sound": {"type": "noul", "instructions": "Does the video have sound?"}}},
    ])
    for i, request in enumerate(requests):
        record = release.encode_record(processor.tokenizer, request, processor=processor)
        media = record.media or {}
        ids = torch.tensor([record.input_ids])
        position_ids, _ = thinker.get_rope_index(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            image_grid_thw=media.get("image_grid_thw"), video_grid_thw=media.get("video_grid_thw"),
            audio_seqlens=media.get("feature_attention_mask", torch.empty(0,0)).sum(-1),
            use_audio_in_video=media.get("use_audio_in_video", False),
            second_per_grids=media.get("video_second_per_grid"),
        )
        result = {"request": request, "input_ids": list(record.input_ids),
                  "position_ids": position_ids[:,0].T.tolist(),
                  "questions": [{"question_id":q.question_id,"question_span":q.question_span,
                                 "option_spans":q.option_spans,"option_ids":q.option_ids} for q in record.questions]}
        (args.output / f"encoding-{i}.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    # Small independent RMSNorm/RoPE fixtures from the exact upstream classes.
    x = torch.tensor([((i * 13) % 79 - 39) / 32 for i in range(3*128)], dtype=torch.bfloat16).reshape(3,128)
    norm = Qwen3OmniMoeThinkerTextRMSNorm(128).to(torch.bfloat16)
    norm.weight.data.copy_(torch.tensor([(i % 11 + 1) / 8 for i in range(128)],dtype=torch.bfloat16))
    normalized = norm(x).detach()
    rope = Qwen3OmniMoeThinkerTextRotaryEmbedding(cfg.text_config)
    positions = torch.tensor([[[0,11,31]],[[0,17,37]],[[0,23,41]]])
    cos, sin = rope(x,positions)
    rotated,_ = apply_rotary_pos_emb(normalized.reshape(1,1,3,128), normalized.reshape(1,1,3,128),cos,sin)
    ops = {"x":x.float().flatten().tolist(),"norm_weight":norm.weight.float().tolist(),
           "normalized":normalized.float().flatten().tolist(),"positions":positions[:,0].T.tolist(),
           "rotated":rotated.float().flatten().tolist()}
    (args.output / "operators.json").write_text(json.dumps(ops, separators=(",", ":")) + "\n")
    operator_fixtures(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
