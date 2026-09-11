"""Record artifacts only after build_overlay.sh completes successfully."""

import json
from pathlib import Path

from .prepare_overlay import OUTPUT
from .qualification import digest


def main():
    path = OUTPUT / "source-manifest.json"
    manifest = json.loads(path.read_text())
    reference = Path(manifest["reference"])
    for name, expected in manifest["source_hashes"].items():
        if digest(reference / name) != expected:
            raise ValueError("Reference controller changed: " + name)
    target = OUTPUT / "src/franka_fr3_arm_controllers"
    library = OUTPUT / "install/franka_fr3_arm_controllers/lib/libfranka_fr3_arm_controllers.so"
    manifest.update(
        build_completed=True,
        reference_sources_unchanged=True,
        built_library=str(library),
        built_library_sha256=digest(library),
        overlay_source_hashes={str(p.relative_to(target)): digest(p) for p in target.rglob("*") if p.is_file()},
    )
    path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
