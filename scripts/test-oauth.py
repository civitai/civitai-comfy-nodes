"""Manual end-to-end OAuth smoke test.

Runs the interactive PKCE login (opens/prints a browser URL), then exercises the
stored-token path by echoing through the real node. Confirms a registered OAuth
app works against production.

    .venv/bin/python scripts/test-oauth.py

Set CIVITAI_OAUTH_CLIENT_ID to test your own app; otherwise the baked-in default is used.
"""

import os

# Force the OAuth path — ignore any API token in the environment.
os.environ.pop("CIVITAI_API_TOKEN", None)

from civitai_comfy_nodes import oauth  # noqa: E402
from civitai_comfy_nodes.generated.misc import CivitaiEcho  # noqa: E402


def main() -> None:
    token = oauth.interactive_login()
    print(f"\n[ok] access token acquired: {token[:12]}…")
    print(f"[ok] tokens stored at {oauth.token_store_path()}")

    message, workflow_id, _raw = CivitaiEcho().run(message="oauth smoke test")
    print(f"[ok] echo round-trip via OAuth token: {message!r} (workflow {workflow_id})")


if __name__ == "__main__":
    main()
