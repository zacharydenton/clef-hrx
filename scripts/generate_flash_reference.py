#!/usr/bin/env python3
"""Generate independent corpus fixtures with the pinned CLEF Flash release code.

Use pristine weights (never HRX tiled shards). Dependencies are documented in
 docs/clef-flash.md. Runs sequentially, without an HRX GPU process.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import transformers
    import importlib.metadata

    path = args.model_dir / "joint_schema_model.py"
    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
    ):
        parser.error("release code does not match the pinned checkpoint")
    spec = importlib.util.spec_from_file_location("flash_release", path)
    release = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = release
    spec.loader.exec_module(release)
    model, processor = release.load_release_model(
        args.model_dir, attn_implementation="sdpa"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "model": "Cloudflare/clef-flash",
                "revision": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
                "release_code_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "attention": "sdpa",
                "dtype": "bfloat16",
                "fla": importlib.metadata.version("fla-core"),
                "video_decoder": "pyav",
            },
            indent=2,
        )
        + "\n"
    )
    with torch.inference_mode():
        for case in range(8):
            request = json.loads(Path(f"tests/fixtures/corpus/{case}.json").read_text())
            if request.get("videos"):
                import av
                import numpy as np
                from transformers.video_utils import VideoMetadata

                videos, metadata = [], []
                for filename in request["videos"]:
                    with av.open(filename) as container:
                        stream = container.streams.video[0]
                        fps = float(stream.average_rate)
                        frames = [
                            frame.to_ndarray(format="rgb24")
                            for frame in container.decode(stream)
                        ]
                    videos.append(np.stack(frames))
                    metadata.append(
                        VideoMetadata(
                            total_num_frames=len(frames),
                            fps=fps,
                            duration=len(frames) / fps,
                            video_backend="pyav",
                            frames_indices=list(range(len(frames))),
                        )
                    )
                request["videos"] = videos
                request.setdefault("media_kwargs", {})["video_metadata"] = metadata
            record = release.encode_record(
                processor.tokenizer, request, max_length=1536, processor=processor
            )
            logits = model(
                release.collate_records(
                    [record], processor.tokenizer.pad_token_id, torch.device("cuda")
                )
            )[0]
            result = {
                "input_ids": list(record.input_ids),
                "questions": [
                    {
                        "question_id": q.question_id,
                        "option_ids": list(q.option_ids),
                        "logits": l.float().tolist(),
                        "probabilities": l.float().softmax(-1).tolist(),
                    }
                    for q, l in zip(record.questions, logits)
                ],
            }
            (args.output / f"{case}.reference.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(f"case {case}: {len(record.input_ids)} tokens", flush=True)


if __name__ == "__main__":
    main()
