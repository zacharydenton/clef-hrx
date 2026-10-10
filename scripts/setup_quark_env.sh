#!/usr/bin/env bash
# Run only after approval of the Quark skill's environment-change checkpoint.
set -euo pipefail

quark_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
quark_env="$quark_root/.venv-quark"
quark_python=/home/zach/.local/share/uv/python/cpython-3.13.11-linux-x86_64-gnu/bin/python3.13

if [[ -e "$quark_env" ]]; then
    echo "Refusing to overwrite an existing environment: $quark_env" >&2
    exit 1
fi

uv venv --python "$quark_python" "$quark_env"
uv pip install --python "$quark_env/bin/python" \
    --index-url https://download.pytorch.org/whl/rocm7.2 \
    'torch==2.14.0+rocm7.2'
uv pip install --python "$quark_env/bin/python" \
    --index https://download.pytorch.org/whl/rocm7.2 \
    --index-strategy unsafe-best-match \
    --requirements "$quark_root/scripts/quark-requirements.txt"
uv pip freeze --python "$quark_env/bin/python" > "$quark_env/requirements.lock.txt"
"$quark_env/bin/quark-cli" torch-llm-ptq --help
