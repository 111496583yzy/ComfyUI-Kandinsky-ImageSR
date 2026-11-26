"""Download and verify the pinned public Kandinsky ImageSR weight bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent


def digest(path, spec):
    algo = spec["algo"]
    h = hashlib.new(algo)
    if algo == "sha1":
        h.update(f"blob {spec['size']}".encode() + bytes((0,)))
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_files(root, files):
    missing, mismatch = [], []
    for spec in files:
        part = PurePosixPath(spec["name"])
        if part.is_absolute() or ".." in part.parts:
            raise ValueError("Unexpected file name in the model manifest")
        dest = root.joinpath(*part.parts)
        if not dest.exists():
            missing.append(spec["name"])
        elif not dest.is_file() or dest.stat().st_size != spec["size"] or digest(dest, spec) != spec["hash"]:
            mismatch.append(spec["name"])
    return missing, mismatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, help="Override the local ComfyUI models directory")
    parser.add_argument("--verify-only", action="store_true", help="Check existing files without downloading")
    opts = parser.parse_args()
    config = json.loads((HERE / "deployment.json").read_text(encoding="utf-8"))
    manifest = json.loads((HERE / "source-manifest.json").read_text(encoding="utf-8"))
    location = Path(config["model_directory"]).expanduser()
    if not location.is_absolute():
        models = opts.models_dir
        if models is None:
            if HERE.parent.name != "custom_nodes":
                parser.error("Place this plugin in ComfyUI/custom_nodes or specify --models-dir")
            models = HERE.parent.parent / "models"
        location = models.expanduser() / location
    location = location.resolve()
    files = manifest["files"]
    missing, mismatch = inspect_files(location, files)
    if mismatch:
        raise SystemExit("Existing files do not match the pinned release; not overwriting: " + ", ".join(mismatch))
    if missing and opts.verify_only:
        raise SystemExit("Missing files: " + ", ".join(missing))
    if missing:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=manifest["repo"], revision=manifest["revision"],
            allow_patterns=missing, local_dir=location, token=False,
        )
        selected = [spec for spec in files if spec["name"] in missing]
        still_missing, mismatch = inspect_files(location, selected)
        if still_missing or mismatch:
            raise SystemExit("Model verification failed: " + ", ".join(still_missing + mismatch))
    print(f"Model files verified: {location}")


if __name__ == "__main__":
    main()
