"""Read the signed-in user's Buzz wallet balances from civitai.com.

The orchestrator has no consumer endpoint for balances; the site exposes them through its tRPC
`buzz.getBuzzAccount` query, which accepts the same Bearer token (API key or OAuth access token
with the BuzzRead scope) the pack already holds.
"""

from __future__ import annotations

import requests

from .catalog import USER_AGENT
from .config import BUZZ_ACCOUNTS
from .errors import CivitaiNodeError

CIVITAI_BUZZ_ACCOUNTS_URL = "https://civitai.com/api/trpc/buzz.getBuzzAccount"


def parse_buzz_accounts(payload: dict) -> dict[str, int]:
    """Pull `{blue, green, yellow}` out of the tRPC envelope (`result.data.json`). Missing wallets
    read as 0; anything non-numeric is a contract change and raises."""
    data = ((payload or {}).get("result") or {}).get("data") or {}
    accounts = data.get("json", data)
    if not isinstance(accounts, dict):
        raise CivitaiNodeError("Unexpected Buzz balance response from civitai.com")
    balances = {}
    for account in BUZZ_ACCOUNTS:
        value = accounts.get(account, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CivitaiNodeError(f"Unexpected Buzz balance for {account}: {value!r}")
        balances[account] = int(value)
    return balances


def fetch_buzz_accounts(token: str, *, timeout: int = 15) -> dict[str, int]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json", "Authorization": f"Bearer {token}"}
    response = requests.get(CIVITAI_BUZZ_ACCOUNTS_URL, headers=headers, timeout=timeout)
    if response.status_code in (401, 403):
        raise CivitaiNodeError("civitai.com rejected the stored credentials for reading Buzz balances")
    response.raise_for_status()
    return parse_buzz_accounts(response.json())
