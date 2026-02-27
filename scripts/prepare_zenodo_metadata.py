"""Generate Zenodo metadata JSON for experiment artifacts.

Computes SHA256 checksums, records git commit provenance, and writes a
metadata file that upload_zenodo.py reads to drive the upload.

Usage:
    python scripts/prepare_zenodo_metadata.py \
        staging/*.tar.gz \
        --experiment-name my_experiment \
        --steps 2000 --seed-start 0 --seed-end 99 \
        --paper-binding tab:results=experiments/results.json \
        --zenodo-doi 10.5281/zenodo.XXXXXXX \
        --output docs/zenodo_metadata.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out or None


def _parse_binding(raw: str) -> dict[str, str]:
    if "=" not in raw:
        raise ValueError(f"invalid binding '{raw}': expected REF=PATH")
    ref, path = raw.split("=", 1)
    ref, path = ref.strip(), path.strip()
    if not ref or not path:
        raise ValueError(f"invalid binding '{raw}': expected REF=PATH")
    return {"paper_ref": ref, "file_path": path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--entrypoint", default=None)
    parser.add_argument("--paper-binding", action="append", default=[])
    parser.add_argument("--zenodo-doi", default=None)
    parser.add_argument("--output", type=Path, default=Path("zenodo_metadata.json"))
    args = parser.parse_args()

    artifacts = []
    for path in sorted(args.files):
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"not found: {path}")
        artifacts.append({
            "path": str(path),
            "size_bytes": resolved.stat().st_size,
            "sha256": _sha256(resolved),
        })

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "experiment_name": args.experiment_name,
        "artifact_source_commit": _detect_git_commit(),
        "entrypoint": args.entrypoint,
        # Never pass secrets via CLI — argv is safe here
        "metadata_generation_argv": list(sys.argv[1:]),
        "seed_range": {"start": args.seed_start, "end": args.seed_end},
        "steps": args.steps,
        "artifacts": artifacts,
        "paper_bindings": [_parse_binding(b) for b in args.paper_binding],
    }
    if args.zenodo_doi:
        payload["zenodo_doi"] = args.zenodo_doi

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Saved: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
