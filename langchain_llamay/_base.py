"""Where a llamay server is, when nobody says."""

import os

DEFAULT_BASE_URL = "http://localhost:11435"


def default_base_url() -> str:
    """LLAMAY_HOST, or llamay's own port on this machine.

    llamay listens on 11435, beside Ollama's 11434, so the two can run at once;
    pointing at 11434 by default would quietly talk to Ollama instead.
    """
    return os.environ.get("LLAMAY_HOST") or DEFAULT_BASE_URL
