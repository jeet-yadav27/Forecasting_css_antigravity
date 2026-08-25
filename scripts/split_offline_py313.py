"""
Split packages-py313 wheels into 3 size-balanced folders and zip them
for a GitHub Release (browser-friendly downloads).

Run from project root:
    python scripts/split_offline_py313.py
"""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages-py313"
OUT_ZIPS = ROOT / "_release_assets"
N_PARTS = 3
PART_NAMES = [f"part{i}" for i in range(1, N_PARTS + 1)]


def _wheels() -> list[Path]:
    files = [
        p for p in SRC.iterdir()
        if p.is_file() and p.suffix.lower() in {".whl", ".tar.gz", ".zip"}
        and p.name.lower() != "readme.txt"
    ]
    # Also collect wheels already sitting in part folders (re-run)
    for part in PART_NAMES:
        d = SRC / part
        if d.is_dir():
            files.extend(
                p for p in d.iterdir()
                if p.is_file() and p.suffix.lower() in {".whl", ".tar.gz"}
            )
    # Unique by name (prefer already-in-part path last → we move from wherever)
    by_name: dict[str, Path] = {}
    for p in files:
        by_name[p.name] = p
    return list(by_name.values())


def _pack(files: list[Path]) -> list[list[Path]]:
    """Greedy: always add next-largest file to the currently smallest bin."""
    ordered = sorted(files, key=lambda p: p.stat().st_size, reverse=True)
    bins: list[list[Path]] = [[] for _ in range(N_PARTS)]
    sizes = [0] * N_PARTS
    for p in ordered:
        i = sizes.index(min(sizes))
        bins[i].append(p)
        sizes[i] += p.stat().st_size
    for b in bins:
        b.sort(key=lambda p: p.name.lower())
    return bins


def _mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def main() -> None:
    files = _wheels()
    if not files:
        raise SystemExit(f"No wheels found under {SRC}")

    bins = _pack(files)
    OUT_ZIPS.mkdir(exist_ok=True)

    # Move into part1 / part2 / part3
    for i, group in enumerate(bins):
        dest = SRC / PART_NAMES[i]
        dest.mkdir(exist_ok=True)
        lines = [
            f"# packages-py313 {PART_NAMES[i]}",
            f"# {len(group)} files, {_mb(sum(p.stat().st_size for p in group))}",
            "# Extract this zip into packages-py313\\ so this folder sits next to part2/part3.",
            "",
        ]
        for p in group:
            target = dest / p.name
            if p.resolve() != target.resolve():
                if target.exists() and p.resolve() != target.resolve():
                    target.unlink()
                shutil.move(str(p), str(target))
            lines.append(p.name)
        (dest / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Zip each part (store = wheels already compressed)
    zip_paths = []
    for name in PART_NAMES:
        folder = SRC / name
        zip_path = OUT_ZIPS / f"packages-py313-{name}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for p in sorted(folder.iterdir()):
                if p.is_file():
                    zf.write(p, arcname=f"{name}/{p.name}")
        zip_paths.append(zip_path)
        print(f"Wrote {zip_path.name}  {_mb(zip_path.stat().st_size)}")

    print("Folders:", ", ".join(str(SRC / n) for n in PART_NAMES))
    print("Upload these three zips as GitHub Release assets.")


if __name__ == "__main__":
    main()
