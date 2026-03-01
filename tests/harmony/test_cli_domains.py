"""Tests for CLI domain registration — ensures all domains are wired."""

from __future__ import annotations

import pytest


class TestCliDomainBuilders:
    """Verify harmony.cli._DOMAIN_BUILDERS covers all expected domains."""

    EXPECTED_DOMAINS = {
        "linear_algebra",
        "periodic_table",
        "astronomy",
        "physics",
        "materials",
        "wikidata_physics",
        "wikidata_materials",
    }

    def test_cli_has_all_domains(self):
        from harmony.cli import _DOMAIN_BUILDERS

        assert self.EXPECTED_DOMAINS == set(_DOMAIN_BUILDERS.keys())

    @pytest.mark.parametrize("domain", sorted(EXPECTED_DOMAINS))
    def test_cli_builder_callable(self, domain: str):
        """Each CLI domain builder is a callable."""
        from harmony.cli import _DOMAIN_BUILDERS

        assert callable(_DOMAIN_BUILDERS[domain])
