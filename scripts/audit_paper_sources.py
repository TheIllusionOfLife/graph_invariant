#!/usr/bin/env python3
"""Audit paper bibliography sources via live web checks.

This script parses `paper/references.bib`, validates DOI resolution using
`doi.org`, checks URL reachability, and writes a markdown audit report.
"""

from __future__ import annotations

import argparse
import ipaddress
import re
import socket
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit

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

_FORBIDDEN_LOCAL_HOSTNAMES = {"localhost", "localhost.localdomain"}
_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}


def _read_braced_value(text: str, pos: int) -> tuple[str, int]:
    depth = 1
    i = pos + 1
    out: list[str] = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
            i += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out), i + 1
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out), i


def _read_quoted_value(text: str, pos: int) -> tuple[str, int]:
    i = pos + 1
    out: list[str] = []
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            out.append(text[i + 1])
            i += 2
            continue
        if ch == '"':
            return "".join(out), i + 1
        out.append(ch)
        i += 1
    return "".join(out), i


def _read_plain_value(text: str, pos: int) -> tuple[str, int]:
    i = pos
    while i < len(text) and text[i] not in ",\n\r":
        i += 1
    return text[pos:i], i


def _parse_entry_fields(entry_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    i = 0
    n = len(entry_text)
    while i < n:
        while i < n and entry_text[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        key_start = i
        while i < n and (entry_text[i].isalnum() or entry_text[i] in "_-"):
            i += 1
        key = entry_text[key_start:i].strip().lower()
        if not key:
            while i < n and entry_text[i] not in "\n\r":
                i += 1
            continue
        while i < n and entry_text[i].isspace():
            i += 1
        if i >= n or entry_text[i] != "=":
            while i < n and entry_text[i] not in ",\n\r":
                i += 1
            continue
        i += 1
        while i < n and entry_text[i].isspace():
            i += 1
        if i >= n:
            break
        if entry_text[i] == "{":
            value, i = _read_braced_value(entry_text, i)
        elif entry_text[i] == '"':
            value, i = _read_quoted_value(entry_text, i)
        else:
            value, i = _read_plain_value(entry_text, i)
        normalized = " ".join(value.split()).strip()
        fields[key] = normalized
        while i < n and entry_text[i].isspace():
            i += 1
        if i < n and entry_text[i] == ",":
            i += 1
    return fields


def _parse_bibtex(path: Path) -> list[BibEntry]:
    entries: list[BibEntry] = []
    text = path.read_text(encoding="utf-8")
    pos = 0
    while True:
        match = ENTRY_HEADER_RE.search(text, pos)
        if match is None:
            break
        entry_type = match.group(1).strip().lower()
        key = match.group(2).strip() or "unknown"
        i = match.end()
        depth = 1
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        entry_text = text[match.end() : max(match.end(), i - 1)]
        fields = _parse_entry_fields(entry_text)
        entries.append(
            BibEntry(
                key=key,
                entry_type=entry_type,
                doi=fields.get("doi"),
                url=fields.get("url"),
            )
        )
        pos = i
    return entries


def _forbidden_ip_reason(ip_addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    if ip_addr.is_private:
        return "private address"
    if ip_addr.is_loopback:
        return "loopback address"
    if ip_addr.is_link_local:
        return "link-local address"
    if ip_addr.is_multicast:
        return "multicast address"
    if ip_addr.is_reserved:
        return "reserved address"
    if ip_addr.is_unspecified:
        return "unspecified address"
    return None


def _resolve_host_ips(host: str) -> tuple[list[str], str | None]:
    try:
        ip_literal = ipaddress.ip_address(host)
        return [str(ip_literal)], None
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        return [], f"host resolution failed ({type(exc).__name__})"
    ips = sorted({info[4][0] for info in infos if info and info[4]})
    if not ips:
        return [], "host resolution returned no addresses"
    return ips, None


def _validate_public_http_url(raw_url: str) -> tuple[str | None, str | None]:
    url = raw_url.strip()
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None, "invalid URL"
    if parsed.scheme not in {"http", "https"}:
        return None, "URL scheme must be http/https"
    if parsed.username or parsed.password:
        return None, "URL must not include credentials"
    host = parsed.hostname
    if host is None:
        return None, "URL host is missing"
    lowered_host = host.lower()
    if lowered_host in _FORBIDDEN_LOCAL_HOSTNAMES:
        return None, "URL host resolves to local machine"
    try:
        _ = parsed.port
    except ValueError:
        return None, "invalid URL port"
    ips, resolve_error = _resolve_host_ips(host)
    if resolve_error is not None:
        return None, resolve_error
    for ip_text in ips:
        try:
            ip_addr = ipaddress.ip_address(ip_text)
        except ValueError:
            return None, f"invalid resolved IP {ip_text}"
        reason = _forbidden_ip_reason(ip_addr)
        if reason is not None:
            return None, f"URL points to forbidden {reason} ({ip_text})"
    return url, None


def _request_with_validated_redirects(
    method: str,
    start_url: str,
    timeout_sec: float,
    max_redirects: int = 5,
) -> tuple[requests.Response | None, str | None]:
    current = start_url
    for _ in range(max_redirects + 1):
        resp = requests.request(method, current, timeout=timeout_sec, allow_redirects=False)
        if resp.status_code not in _REDIRECT_STATUS_CODES:
            return resp, None
        location = resp.headers.get("Location")
        if not location:
            return resp, None
        next_url = urljoin(current, location)
        _, err = _validate_public_http_url(next_url)
        if err is not None:
            return None, f"BLOCKED redirect target: {err}"
        current = next_url
    return None, "ERROR TooManyRedirects"


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
    safe_url, validation_error = _validate_public_http_url(url)
    if validation_error is not None:
        return f"BLOCKED {validation_error}"
    assert safe_url is not None
    try:
        head, head_error = _request_with_validated_redirects(
            "HEAD",
            safe_url,
            timeout_sec=timeout_sec,
        )
        if head_error is not None:
            return head_error
        assert head is not None
        if head.status_code < 400:
            return f"OK ({head.status_code})"
        # fallback to GET in case HEAD is unsupported
        get, get_error = _request_with_validated_redirects(
            "GET",
            safe_url,
            timeout_sec=timeout_sec,
        )
        if get_error is not None:
            return get_error
        assert get is not None
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
