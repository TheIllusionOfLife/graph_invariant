#!/usr/bin/env python3
"""Audit paper bibliography sources via live web checks.

This script parses `paper/references.bib`, validates DOI resolution using
`doi.org`, checks URL reachability, and writes a markdown audit report.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass(slots=True)
class BibEntry:
    key: str
    entry_type: str
    doi: str | None
    url: str | None


@dataclass(slots=True)
class AuditRow:
    key: str
    entry_type: str
    doi: str | None
    doi_status: str
    doi_title: str | None
    doi_resolver: str | None
    url: str | None
    url_status: str


ENTRY_HEADER_RE = re.compile(r"@(\w+)\s*\{\s*([^,]+),", re.IGNORECASE)
FIELD_RE = re.compile(r"^\s*([A-Za-z_]+)\s*=\s*[{\"](.+?)[}\"]\s*,?\s*$")


def _parse_bibtex(path: Path) -> list[BibEntry]:
    entries: list[BibEntry] = []
    entry_type: str | None = None
    key: str | None = None
    fields: dict[str, str] = {}
    depth = 0

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        if entry_type is None:
            m = ENTRY_HEADER_RE.match(line)
            if m:
                entry_type = m.group(1).strip().lower()
                key = m.group(2).strip()
                fields = {}
                depth = line.count("{") - line.count("}")
            continue

        depth += line.count("{") - line.count("}")
        m = FIELD_RE.match(raw_line)
        if m:
            fields[m.group(1).strip().lower()] = m.group(2).strip()
        if depth <= 0:
            entries.append(
                BibEntry(
                    key=key or "unknown",
                    entry_type=entry_type,
                    doi=fields.get("doi"),
                    url=fields.get("url"),
                )
            )
            entry_type = None
            key = None
            fields = {}
            depth = 0
    return entries


def _fetch_doi_metadata(doi: str, timeout_sec: float) -> tuple[str, str | None, str | None]:
    resolver = f"https://doi.org/{doi}"
    headers = {"Accept": "application/vnd.citationstyles.csl+json"}
    try:
        resp = requests.get(resolver, headers=headers, timeout=timeout_sec, allow_redirects=True)
        if resp.status_code != 200:
            return f"HTTP {resp.status_code}", None, resolver
        payload = resp.json()
        title = payload.get("title")
        if isinstance(title, list):
            title = title[0] if title else None
        if title is not None and not isinstance(title, str):
            title = str(title)
        return "OK", title, resolver
    except requests.RequestException as exc:
        return f"ERROR {type(exc).__name__}", None, resolver
    except ValueError:
        # Some resolvers may return non-JSON HTML pages even with Accept header.
        return "OK (non-JSON)", None, resolver


def _check_url(url: str, timeout_sec: float) -> str:
    try:
        head = requests.head(url, timeout=timeout_sec, allow_redirects=True)
        if head.status_code < 400:
            return f"OK ({head.status_code})"
        # fallback to GET in case HEAD is unsupported
        get = requests.get(url, timeout=timeout_sec, allow_redirects=True)
        return f"OK ({get.status_code})" if get.status_code < 400 else f"HTTP {get.status_code}"
    except requests.RequestException as exc:
        return f"ERROR {type(exc).__name__}"


def _audit_entries(entries: Iterable[BibEntry], timeout_sec: float) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for entry in entries:
        doi_status = "N/A"
        doi_title = None
        doi_resolver = None
        if entry.doi:
            doi_status, doi_title, doi_resolver = _fetch_doi_metadata(entry.doi, timeout_sec)
        url_status = "N/A"
        if entry.url:
            url_status = _check_url(entry.url, timeout_sec)
        rows.append(
            AuditRow(
                key=entry.key,
                entry_type=entry.entry_type,
                doi=entry.doi,
                doi_status=doi_status,
                doi_title=doi_title,
                doi_resolver=doi_resolver,
                url=entry.url,
                url_status=url_status,
            )
        )
    return rows


def _render_report(rows: list[AuditRow], bib_path: Path) -> str:
    total = len(rows)
    doi_ok = sum(1 for r in rows if r.doi and r.doi_status.startswith("OK"))
    url_ok = sum(1 for r in rows if r.url and r.url_status.startswith("OK"))
    lines = [
        "# Paper Source Audit",
        "",
        f"- Bib file: `{bib_path}`",
        f"- Entries audited: {total}",
        f"- DOI checks passed: {doi_ok}",
        f"- URL checks passed: {url_ok}",
        "",
        "| Key | Type | DOI | DOI Status | DOI Resolver | URL | URL Status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        doi = r.doi or ""
        resolver = r.doi_resolver or ""
        url = r.url or ""
        lines.append(
            f"| {r.key} | {r.entry_type} | {doi} | {r.doi_status} | "
            f"{resolver} | {url} | {r.url_status} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- DOI checks use `https://doi.org/<doi>` resolution with "
            "citation-style JSON accept header.",
            "- URL checks use `HEAD` then fallback `GET` on failure/non-2xx.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit paper bibliography sources via web checks.")
    parser.add_argument(
        "--bib",
        type=Path,
        default=Path("paper/references.bib"),
        help="Path to BibTeX file (default: paper/references.bib)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/paper_source_audit.md"),
        help="Path to markdown output (default: docs/paper_source_audit.md)",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds for each check (default: 10.0)",
    )
    args = parser.parse_args()

    entries = _parse_bibtex(args.bib)
    rows = _audit_entries(entries, timeout_sec=args.timeout_sec)
    report = _render_report(rows, bib_path=args.bib)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote audit report: {args.output}")


if __name__ == "__main__":
    main()
