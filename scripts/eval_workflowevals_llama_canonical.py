#!/usr/bin/env python3
"""Text-only WorkflowEvals adapter with canonical structured-state serialization.

Reuse the frozen llama runner, including its memory guards and scoring contract.
Fingerprint both entry points so old pilots and checkpoints cannot be resumed
under the changed input transport. Media requests are outside this adapter's
scope.
"""
import hashlib
import json
from pathlib import Path

import eval_workflowevals_llama as frozen


def canonical_request(request):
    """Match HRX's structured-state text without C++ float formatting."""
    state = request['state']
    if isinstance(state, str):
        return request
    return {**request, 'state': json.dumps(state, ensure_ascii=False, sort_keys=True,
                                         separators=(',', ':'), allow_nan=False)}


def adapter_digest(base_source, canonical_source):
    sources = {'frozen': hashlib.sha256(base_source).hexdigest(),
               'canonical': hashlib.sha256(canonical_source).hexdigest()}
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def main():
    base_path = Path(frozen.__file__).resolve()
    canonical_source = Path(__file__).read_bytes()
    digest = adapter_digest(base_path.read_bytes(), canonical_source)
    original_client, original_hash, original_write = frozen.LlamaClient, frozen.sha256, frozen.write_json

    class CanonicalClient(original_client):
        def request(self, request):
            return super().request(canonical_request(request))

    def hash_source(path):
        return digest if Path(path).resolve() == base_path else original_hash(path)

    def write_record(path, value):
        original_write(path, value)
        if Path(path).name == 'manifest.json':
            # The base runner already snapshots itself as runner.py.
            Path(path).with_name('canonical-runner.py').write_bytes(canonical_source)

    frozen.LlamaClient, frozen.sha256, frozen.write_json = CanonicalClient, hash_source, write_record
    try:
        return frozen.main()
    finally:
        frozen.LlamaClient, frozen.sha256, frozen.write_json = original_client, original_hash, original_write


if __name__ == '__main__':
    raise SystemExit(main())
