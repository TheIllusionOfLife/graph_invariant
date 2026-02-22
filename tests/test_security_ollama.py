import ipaddress
import socket
from urllib.parse import urlparse

import pytest

from graph_invariant.llm_ollama import validate_ollama_url


def test_validate_ollama_url_localhost_allow_remote_false():
    # Test localhost (allow_remote=False)
    # This might resolve to 127.0.0.1 or ::1
    url, headers = validate_ollama_url("http://localhost:11434", False)
    assert "127.0.0.1" in url or "::1" in url
    assert headers["Host"] == "localhost:11434"


def test_validate_ollama_url_ip_allow_remote_false():
    # Test IP literal (allow_remote=False)
    url, headers = validate_ollama_url("http://127.0.0.1:11434", False)
    assert "127.0.0.1" in url
    assert headers["Host"] == "127.0.0.1:11434"


def test_validate_ollama_url_forbidden_ip_allow_remote_false():
    # Test forbidden IP (allow_remote=False)
    # 1.1.1.1 is public, so not loopback.
    with pytest.raises(ValueError, match="ollama_url must target localhost"):
        validate_ollama_url("http://1.1.1.1:11434", False)


def test_validate_ollama_url_public_ip_allow_remote_true():
    # Test allow_remote=True with public IP
    # We use a dummy IP that parses as valid
    url, headers = validate_ollama_url("http://1.2.3.4:11434", True)
    assert "1.2.3.4" in url
    assert headers["Host"] == "1.2.3.4:11434"


def test_validate_ollama_url_metadata_ip_blocked():
    # Test allow_remote=True with metadata IP (blocked)
    # 169.254.169.254 is link-local
    with pytest.raises(ValueError, match="link-local"):
        validate_ollama_url("http://169.254.169.254/latest", True)


def test_validate_ollama_url_dns_resolution_http():
    # Test DNS rebinding prevention (HTTP)
    # We check that the returned URL contains an IP address, not the hostname
    try:
        # Assuming example.com resolves
        url, headers = validate_ollama_url("http://example.com", True)
        assert "example.com" not in url
        assert headers["Host"] == "example.com"

        # Parse the returned URL to verify netloc is an IP
        parsed = urlparse(url)
        # remove brackets for IPv6
        host = parsed.hostname
        ipaddress.ip_address(host)  # Should verify it's an IP
    except socket.gaierror:
        pytest.skip("No DNS resolution for example.com")


def test_validate_ollama_url_https_preserves_hostname():
    # Test HTTPS (Hostname preserved)
    try:
        url, headers = validate_ollama_url("https://example.com", True)
        assert "example.com" in url
        assert "Host" not in headers  # No override needed
    except socket.gaierror:
        pytest.skip("No DNS resolution for example.com")


def test_validate_ollama_url_ipv6_bracket():
    # Test IPv6 bracketed
    url, headers = validate_ollama_url("http://[::1]:11434", False)
    assert "[::1]" in url
    assert headers["Host"] == "[::1]:11434"


def test_validate_ollama_url_user_pass():
    # Test URL with user:pass
    # http://user:pass@localhost:11434
    url, headers = validate_ollama_url("http://user:pass@localhost:11434", False)
    assert "user:pass@" in url
    assert "127.0.0.1" in url or "::1" in url
    assert headers["Host"] == "localhost:11434"
