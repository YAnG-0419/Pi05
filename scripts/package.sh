#!/usr/bin/env bash
set -euo pipefail
DP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - "$DP_ROOT" <<'PY'
from pathlib import Path
import tarfile
import sys
root = Path(sys.argv[1])
out = root / 'dist/dp_infer-deploy.tar.gz'
out.parent.mkdir(exist_ok=True)
names = ['README.md', 'UPSTREAM.md', 'requirements-model.txt', 'run.sh',
         'configs', 'dp_deploy', 'vendor', 'scripts', 'tests']
def selected(info):
    if '__pycache__' in Path(info.name).parts or info.name.endswith(('.pyc', '.pyo')):
        return None
    return info
with tarfile.open(out, 'w:gz') as archive:
    for name in names:
        archive.add(root / name, arcname='dp_infer/' + name, filter=selected)
print(out)
PY
