"""Upload research artifacts to Zenodo via the REST API.

Reads artifact paths and checksums from a metadata JSON file, creates a
draft deposit, uploads each file via the bucket URL, sets metadata, and
optionally publishes.

Usage:
    # Draft (default — safe, review before publishing):
    python scripts/upload_zenodo.py --metadata zenodo_metadata.json

    # Publish (irreversible):
    python scripts/upload_zenodo.py --metadata zenodo_metadata.json --publish

    # New version of existing record:
    python scripts/upload_zenodo.py --metadata zenodo_metadata.json \
        --new-version RECORD_ID --publish

    # Edit metadata only (no re-upload):
    python scripts/upload_zenodo.py --edit RECORD_ID --title "New title" --publish

    # Fetch BibTeX:
    python scripts/upload_zenodo.py --fetch-bibtex RECORD_ID

    # Sandbox testing:
    python scripts/upload_zenodo.py --metadata zenodo_metadata.json --sandbox

Environment:
    ZENODO_TOKEN — personal access token with deposit:write and
                   deposit:actions scopes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import requests

ZENODO_API = "https://zenodo.org/api"
SANDBOX_API = "https://sandbox.zenodo.org/api"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_response(resp: requests.Response, context: str) -> None:
    if resp.status_code >= 400:
        print(f"ERROR [{context}]: {resp.status_code}", file=sys.stderr)
        try:
            print(json.dumps(resp.json(), indent=2), file=sys.stderr)
        except ValueError:
            print(resp.text[:500], file=sys.stderr)
        sys.exit(1)


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _json_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _parse_creator(raw: str) -> dict[str, str]:
    """Parse 'Last, First; Affiliation; 0000-0000-0000-0000'."""
    parts = [p.strip() for p in raw.split(";")]
    name = parts[0]
    if not name:
        raise ValueError(f"invalid creator: '{raw}'")
    entry: dict[str, str] = {"name": name}
    if len(parts) > 1 and parts[1]:
        entry["affiliation"] = parts[1]
    if len(parts) > 2 and parts[2]:
        entry["orcid"] = parts[2]
    return entry


# ---------------------------------------------------------------------------
# API operations
# ---------------------------------------------------------------------------


def create_deposit(base_url: str, token: str) -> dict:
    resp = requests.post(
        f"{base_url}/deposit/depositions",
        json={},
        headers=_json_headers(token),
        timeout=30,
    )
    _check_response(resp, "create deposit")
    data = resp.json()
    doi = data["metadata"]["prereserve_doi"]["doi"]
    print(f"Created deposit {data['id']} (DOI: {doi})", file=sys.stderr)
    return data


def create_new_version(base_url: str, token: str, record_id: int) -> dict:
    resp = requests.post(
        f"{base_url}/deposit/depositions/{record_id}/actions/newversion",
        headers=_auth_headers(token),
        timeout=30,
    )
    _check_response(resp, "new version")
    resp2 = requests.get(
        resp.json()["links"]["latest_draft"],
        headers=_auth_headers(token),
        timeout=15,
    )
    _check_response(resp2, "fetch new version draft")
    draft = resp2.json()
    print(f"New version draft {draft['id']}", file=sys.stderr)
    return draft


def edit_published(base_url: str, token: str, record_id: int) -> dict:
    resp = requests.post(
        f"{base_url}/deposit/depositions/{record_id}/actions/edit",
        headers=_auth_headers(token),
        timeout=30,
    )
    _check_response(resp, "edit published")
    print(f"Unlocked deposit {record_id} for editing.", file=sys.stderr)
    return resp.json()


def upload_file(bucket_url: str, token: str, path: Path) -> dict:
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Uploading {path.name} ({size_mb:.1f} MB) ...", file=sys.stderr, end="", flush=True)
    with open(path, "rb") as fp:
        resp = requests.put(
            f"{bucket_url}/{path.name}",
            data=fp,
            headers=_auth_headers(token),
            timeout=600,
        )
    _check_response(resp, f"upload {path.name}")
    data = resp.json()
    print(f" OK (checksum: {data.get('checksum', 'n/a')})", file=sys.stderr)
    return data


def delete_file(base_url: str, token: str, dep_id: int, file_id: str) -> None:
    resp = requests.delete(
        f"{base_url}/deposit/depositions/{dep_id}/files/{file_id}",
        headers=_auth_headers(token),
        timeout=15,
    )
    _check_response(resp, f"delete file {file_id}")


def set_metadata(base_url: str, token: str, dep_id: int, metadata: dict) -> dict:
    resp = requests.put(
        f"{base_url}/deposit/depositions/{dep_id}",
        json={"metadata": metadata},
        headers=_json_headers(token),
        timeout=30,
    )
    _check_response(resp, "set metadata")
    print("  Metadata updated.", file=sys.stderr)
    return resp.json()


def publish_deposit(base_url: str, token: str, dep_id: int) -> dict:
    resp = requests.post(
        f"{base_url}/deposit/depositions/{dep_id}/actions/publish",
        headers=_auth_headers(token),
        timeout=30,
    )
    _check_response(resp, "publish")
    data = resp.json()
    print(f"  Published! DOI: {data['doi']}", file=sys.stderr)
    return data


def fetch_bibtex(base_url: str, record_id: int) -> str:
    resp = requests.get(
        f"{base_url}/records/{record_id}",
        headers={"Accept": "application/x-bibtex"},
        timeout=15,
    )
    _check_response(resp, "fetch bibtex")
    return resp.text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--new-version", type=int, metavar="RECORD_ID")
    mode.add_argument("--edit", type=int, metavar="RECORD_ID")
    mode.add_argument("--fetch-bibtex", type=int, metavar="RECORD_ID")

    parser.add_argument("--metadata", type=Path, default=Path("zenodo_metadata.json"))
    parser.add_argument("--title", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--creator", action="append", default=[])
    parser.add_argument("--version", default=None)
    parser.add_argument("--keyword", action="append", default=[])
    parser.add_argument("--github-url", default=None)
    parser.add_argument("--conference-title", default=None)
    parser.add_argument("--conference-url", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument("--no-verify-checksums", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------


def _build_metadata(args: argparse.Namespace) -> dict:
    if args.creator:
        creators = [_parse_creator(c) for c in args.creator]
    else:
        creators = [{"name": "<authors>"}]
    meta: dict = {
        "title": args.title or "Research Dataset",
        "upload_type": "dataset",
        "description": args.description or "Research experiment data.",
        "creators": creators,
        "license": "MIT",
        "access_right": "open",
    }
    if args.version:
        meta["version"] = args.version
    if args.keyword:
        meta["keywords"] = args.keyword
    if args.github_url:
        meta["related_identifiers"] = [
            {
                "identifier": args.github_url,
                "relation": "isSupplementTo",
                "scheme": "url",
            }
        ]
    if args.conference_title:
        meta["conference_title"] = args.conference_title
    if args.conference_url:
        meta["conference_url"] = args.conference_url
    if args.language:
        meta["language"] = args.language
    return meta


def _load_and_verify(args: argparse.Namespace, meta: dict) -> list[Path]:
    artifacts = meta.get("artifacts", [])
    if not artifacts:
        print("ERROR: no artifacts in metadata.", file=sys.stderr)
        sys.exit(1)
    paths: list[Path] = []
    for entry in artifacts:
        p = Path(entry["path"])
        if not p.exists():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            sys.exit(1)
        if not args.no_verify_checksums:
            local = _sha256(p)
            expected = entry.get("sha256", "")
            if local != expected:
                print(f"ERROR: checksum mismatch: {p}", file=sys.stderr)
                sys.exit(1)
        paths.append(p)
    return paths


def main() -> int:
    args = parse_args()
    base_url = SANDBOX_API if args.sandbox else ZENODO_API

    if args.fetch_bibtex:
        print(fetch_bibtex(base_url, args.fetch_bibtex))
        return 0

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print(
            "ERROR: ZENODO_TOKEN not set.\n"
            "Create at https://zenodo.org/account/settings/applications/\n"
            "Scopes: deposit:write, deposit:actions",
            file=sys.stderr,
        )
        return 1

    env = "SANDBOX" if args.sandbox else "PRODUCTION"
    print(f"Target: {env} ({base_url})", file=sys.stderr)
    zenodo_meta = _build_metadata(args)

    if args.edit is not None:
        edit_published(base_url, token, args.edit)
        set_metadata(base_url, token, args.edit, zenodo_meta)
        if args.publish:
            result = publish_deposit(base_url, token, args.edit)
            print(f"\nDOI: {result['doi']}", file=sys.stderr)
        return 0

    with open(args.metadata) as f:
        meta = json.load(f)
    artifact_paths = _load_and_verify(args, meta)

    if args.new_version:
        draft = create_new_version(base_url, token, args.new_version)
        dep_id = draft["id"]
        bucket_url = draft["links"]["bucket"]
        for old in draft.get("files", []):
            delete_file(base_url, token, dep_id, old["id"])
    else:
        deposit = create_deposit(base_url, token)
        dep_id = deposit["id"]
        bucket_url = deposit["links"]["bucket"]

    print(f"Uploading {len(artifact_paths)} file(s) ...", file=sys.stderr)
    for path in artifact_paths:
        upload_file(bucket_url, token, path)

    set_metadata(base_url, token, dep_id, zenodo_meta)

    if args.publish:
        result = publish_deposit(base_url, token, dep_id)
        print(f"\nDOI: {result['doi']}", file=sys.stderr)
    else:
        web = base_url.replace("/api", "")
        print(f"\nDraft created: {web}/deposit/{dep_id}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
