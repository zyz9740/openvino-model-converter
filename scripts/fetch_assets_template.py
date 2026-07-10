"""Template for per-export scripts/fetch_assets.py.

Copy this file into each export directory's scripts/ folder and fill in
ASSETS for that specific model. The skill's Section 6 (GitHub-ready
cleanup) requires every file excluded by .gitignore to be reproducible by
running this script.

Design rules (do not relax without reading SKILL.md Section 6 first):
- Manifest is hardcoded here, not scraped at runtime. The manifest IS the
  audit trail.
- Every asset has a SHA256. Hash mismatch is a hard fail -- the URL may
  have been replaced and silently using the new file destroys reproducibility.
- Each asset's dest_path is its EXACT final relative location under the
  export directory, so the user does not have to mv anything afterwards.
- Re-running with everything already present is a no-op (skip on hash match).
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

EXPORT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Asset:
    url: str
    dest_path: str          # relative to EXPORT_ROOT
    sha256: str             # lowercase hex
    size_bytes: int
    description: str
    alternate_urls: tuple[str, ...] = ()  # fallback mirrors


# Fill this in for each export. Examples shown -- delete and replace.
ASSETS: list[Asset] = [
    # Example: pretrained weights from HuggingFace via mirror
    # Asset(
    #     url="https://hf-mirror.com/<owner>/<model>/resolve/main/pytorch_model.bin",
    #     alternate_urls=("https://huggingface.co/<owner>/<model>/resolve/main/pytorch_model.bin",),
    #     dest_path="<model_name>/checkpoints/pytorch_model.bin",
    #     sha256="0000000000000000000000000000000000000000000000000000000000000000",
    #     size_bytes=512_000_000,
    #     description="Pretrained PyTorch weights, used by converter/convert.py",
    # ),
    # Example: large test input from the model's release page
    # Asset(
    #     url="https://github.com/<owner>/<repo>/releases/download/v1.0/sample_pointcloud.bin",
    #     dest_path="demo/sample_pointcloud.bin",
    #     sha256="0000000000000000000000000000000000000000000000000000000000000000",
    #     size_bytes=120_000_000,
    #     description="Sample LiDAR scan used by demo/infer_demo.py",
    # ),
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        while True:
            buf = resp.read(1 << 20)
            if not buf:
                break
            out.write(buf)
    tmp.replace(dest)


def fetch_one(asset: Asset) -> str:
    dest = EXPORT_ROOT / asset.dest_path
    if dest.exists() and sha256_file(dest) == asset.sha256:
        return "skip (already present)"

    last_err: Exception | None = None
    for url in (asset.url, *asset.alternate_urls):
        try:
            download(url, dest)
            actual = sha256_file(dest)
            if actual != asset.sha256:
                dest.unlink(missing_ok=True)
                raise RuntimeError(
                    f"sha256 mismatch from {url}: got {actual}, expected {asset.sha256}"
                )
            return f"ok ({url})"
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"all sources failed for {asset.dest_path}: {last_err}")


def main() -> int:
    if not ASSETS:
        print("ASSETS list is empty -- fill in the manifest before running.", file=sys.stderr)
        return 2

    failed = 0
    for a in ASSETS:
        print(f"[{a.dest_path}] ({a.size_bytes/1e6:.1f} MB) {a.description}")
        try:
            status = fetch_one(a)
            print(f"  -> {status}")
        except Exception as e:
            print(f"  -> FAIL: {e}", file=sys.stderr)
            failed += 1

    total = len(ASSETS)
    print(f"\nDone: {total - failed}/{total} assets ready under {EXPORT_ROOT}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
