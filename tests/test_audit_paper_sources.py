import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def audit_module():
    module_path = Path(__file__).resolve().parent.parent / "scripts" / "audit_paper_sources.py"
    spec = importlib.util.spec_from_file_location("audit_paper_sources", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load audit_paper_sources module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_check_url_blocks_link_local_without_network(audit_module, monkeypatch):
    def _no_network(*_args, **_kwargs):
        raise AssertionError("network call should not be attempted for blocked URLs")

    monkeypatch.setattr(audit_module.requests, "request", _no_network)

    status = audit_module._check_url("http://169.254.169.254/latest/meta-data", timeout_sec=1.0)
    assert status.startswith("BLOCKED")


def test_parse_bibtex_multiline_fields(audit_module, tmp_path):
    bib = tmp_path / "sample.bib"
    bib.write_text(
        """
@article{foo2026,
  title = {A long title that spans
    multiple lines with {nested} braces},
  doi = {10.1000/example},
  url = {https://example.com/path?x=1&y=2}
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    entries = audit_module._parse_bibtex(bib)
    assert len(entries) == 1
    assert entries[0].key == "foo2026"
    assert entries[0].entry_type == "article"
    assert entries[0].doi == "10.1000/example"
    assert entries[0].url == "https://example.com/path?x=1&y=2"


def test_parse_bibtex_multiline_quoted_fields(audit_module, tmp_path):
    bib = tmp_path / "quoted.bib"
    bib.write_text(
        """
@inproceedings{bar2026,
  title = "Quoted title on
    two lines",
  url = "https://example.org/ok"
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    entries = audit_module._parse_bibtex(bib)
    assert len(entries) == 1
    assert entries[0].key == "bar2026"
    assert entries[0].entry_type == "inproceedings"
    assert entries[0].url == "https://example.org/ok"
