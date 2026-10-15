#!/usr/bin/env python3
"""Run Cloudflare's unmodified Omni forward with one backbone block resident.

Only checkpoint loading is changed: hooks materialize a module before forward
and release its weights afterward. This is a correctness oracle, not a speed
baseline. It avoids loading the entire 59 GiB thinker at once.
"""
import argparse
import gc
import json
import time
from pathlib import Path
from generate_omni_reference import release_module, REVISION, RELEASE_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("requests", type=Path, nargs="+")
    args = parser.parse_args()
    import torch
    import transformers
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
        Qwen3OmniMoeThinkerTextRotaryEmbedding,
        Qwen3OmniMoeVisionRotaryEmbedding,
        SinusoidsPositionEmbedding,
    )

    release = release_module(args.model_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    cfg = AutoConfig.from_pretrained(args.model_dir, local_files_only=True).thinker_config
    cfg._attn_implementation = "sdpa"
    for sub in [cfg.text_config, cfg.vision_config, cfg.audio_config]:
        sub._attn_implementation = "sdpa"
    with torch.device("meta"):
        thinker = Qwen3OmniMoeThinkerForConditionalGeneration(cfg)
        head = release.JointSchemaHead(**json.loads((args.model_dir / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(args.model_dir / "joint_head.safetensors"), strict=True, assign=True)
    head.to(device=device, dtype=torch.bfloat16)
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(key):
        with safe_open(args.model_dir / index[key], framework="pt", device="cpu") as shard:
            return shard.get_tensor(key)

    def load(module, prefix):
        names = [k for k in index if k.startswith(prefix + ".")]
        state = {k[len(prefix)+1:]: tensor(k) for k in names}
        # Transformers 5.10 groups legacy per-expert checkpoint matrices.
        if "mlp.experts.gate_up_proj" in module.state_dict():
            def stack(projection):
                return torch.stack([state.pop(f"mlp.experts.{i}.{projection}_proj.weight") for i in range(128)])
            gate, up = stack("gate"), stack("up")
            state["mlp.experts.gate_up_proj"] = torch.cat((gate, up), dim=1)
            del gate, up
            state["mlp.experts.down_proj"] = stack("down")
        module.load_state_dict(state, strict=True, assign=True)
        del state
        if prefix == "thinker.audio_tower":
            module.positional_embedding = SinusoidsPositionEmbedding(cfg.audio_config.max_source_positions, cfg.audio_config.d_model)
        if prefix == "thinker.visual":
            module.rotary_pos_emb = Qwen3OmniMoeVisionRotaryEmbedding(36)
        module.to(device=device, dtype=torch.bfloat16)
        # Vision sin/cos and audio positional tables are formed in FP32 upstream.
        if prefix == "thinker.visual":
            module.rotary_pos_emb = Qwen3OmniMoeVisionRotaryEmbedding(36).to(device)
        if prefix == "thinker.audio_tower":
            module.positional_embedding = SinusoidsPositionEmbedding(cfg.audio_config.max_source_positions, cfg.audio_config.d_model).to(device)

    load(thinker.model.embed_tokens, "thinker.model.embed_tokens")
    load(thinker.lm_head, "thinker.lm_head")
    load(thinker.model.norm, "thinker.model.norm")
    thinker.model.rotary_emb = Qwen3OmniMoeThinkerTextRotaryEmbedding(cfg.text_config, device=device)
    hooks = []
    modules = [(thinker.audio_tower,"thinker.audio_tower"),(thinker.visual,"thinker.visual")]
    modules += [(layer,f"thinker.model.layers.{i}") for i,layer in enumerate(thinker.model.layers)]
    for module,prefix in modules:
        def before(module, inputs, prefix=prefix):
            load(module,prefix)
        def after(module, inputs, output, prefix=prefix):
            if device.type == "cuda": torch.cuda.synchronize(device)
            module.to_empty(device="meta")
            return output
        hooks += [module.register_forward_pre_hook(before),module.register_forward_hook(after)]
    model = release.ClefModel(thinker,head).eval()
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    manifest = {"model":"Cloudflare/clef-omni","revision":REVISION,"release_code_sha256":RELEASE_SHA256,
                "torch":torch.__version__,"transformers":transformers.__version__,"device":str(device),
                "dtype":"bfloat16","attention":"sdpa","loading":"one block at a time; unchanged release forward"}
    (args.output / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    with torch.inference_mode():
        for i,path in enumerate(args.requests):
            request=json.loads(path.read_text())
            if "request" in request: request=request["request"]
            record=release.encode_record(processor.tokenizer,request,max_length=args.max_length,processor=processor)
            batch=release.collate_records([record],processor.tokenizer.pad_token_id,device)
            start=time.monotonic()
            logits=model(batch)[0]
            if device.type == "cuda": torch.cuda.synchronize(device)
            output={"request":request,"input_ids":list(record.input_ids),"elapsed_seconds_including_streaming":time.monotonic()-start,
                    "questions":[{"question_id":q.question_id,"option_ids":list(q.option_ids),"logits":v.float().tolist(),
                                  "probabilities":v.float().softmax(-1).tolist()} for q,v in zip(record.questions,logits)]}
            (args.output / f"{i}.json").write_text(json.dumps(output,indent=2)+"\n")
            print(f"{path}: {len(record.input_ids)} tokens, {output['elapsed_seconds_including_streaming']:.2f}s",flush=True)
            del batch,logits
            gc.collect()


if __name__ == "__main__":
    main()
