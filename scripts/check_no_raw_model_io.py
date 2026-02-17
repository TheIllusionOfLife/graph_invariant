"""Fail if tracked artifact logs contain raw LLM prompt/response payloads."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FORBIDDEN_KEYS = ("prompt", "llm_response", "extracted_code")


def main() -> int:
    tracked = subprocess.check_output(
        ["git", "ls-files", "artifacts/**/logs/events.jsonl"],
        text=True,
    ).splitlines()

    violations: list[str] = []
    for rel in tracked:
        path = Path(rel)
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                for idx, raw in enumerate(f, start=1):
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    payload = obj.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    for key in FORBIDDEN_KEYS:
                        if payload.get(key) is not None:
                            violations.append(f"{path}:{idx}: non-null {key}")
        except OSError as exc:
            violations.append(f"{path}: read error: {exc}")

    if violations:
        print("Raw model I/O detected in tracked artifact logs:")
        for item in violations[:100]:
            print(f"  - {item}")
        if len(violations) > 100:
            print(f"  ... and {len(violations) - 100} more")
        return 1

    print("No raw model I/O found in tracked artifact logs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
