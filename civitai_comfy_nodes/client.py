import random
import time
from pathlib import Path
from urllib import parse

import requests

from .config import ClientConfig
from .errors import CivitaiNodeError, http_error_message

RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
TERMINAL_STATUSES = {"succeeded", "failed", "expired", "canceled"}


class OrchestrationClient:
    def __init__(self, config: ClientConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {config.token}"
        # Older orchestrations type GET ?wait as bool and 400 on an integer; flipped off on first 400.
        self._get_wait_supported = True

    def _request(self, method: str, path: str, *, max_tries: int = 4, **kwargs) -> requests.Response:
        url = path if path.startswith("http") else f"{self.config.base_url}{path}"
        kwargs.setdefault("timeout", 120)
        last_response = None
        for attempt in range(max_tries):
            response = self.session.request(method, url, **kwargs)
            if response.status_code not in RETRYABLE_STATUSES:
                if response.status_code >= 400:
                    self._raise_api_error(response)
                return response
            last_response = response
            time.sleep(min(2**attempt, 15) + random.uniform(0, 1))
        self._raise_api_error(last_response)

    @staticmethod
    def _raise_api_error(response: requests.Response):
        error = CivitaiNodeError(http_error_message(response.status_code, response.text))
        error.status_code = response.status_code
        raise error

    def submit_steps(
        self,
        steps: list[dict],
        *,
        wait: int = 5,
        whatif: bool = False,
        tags: list[str] | None = None,
    ) -> dict:
        params: dict = {"wait": wait}
        if whatif:
            params["whatif"] = "true"
        if self.config.allow_mature_content:
            params["hideMatureContent"] = "false"
        body: dict = {"steps": steps}
        if tags:
            body["tags"] = list(tags)
        return self._request("POST", "/v2/consumer/workflows", params=params, json=body).json()

    def submit_workflow(
        self,
        step_type: str,
        input_payload: dict,
        *,
        wait: int = 5,
        whatif: bool = False,
        tags: list[str] | None = None,
    ) -> dict:
        return self.submit_steps(
            [{"$type": step_type, "input": input_payload}], wait=wait, whatif=whatif, tags=tags
        )

    def query_workflows(
        self,
        *,
        cursor: str | None = None,
        take: int = 60,
        exclude_failed: bool = True,
        hide_mature: bool | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """List the caller's own workflows newest-first (scoped to the token's user). Returns
        `{"next": <cursor|None>, "items": [workflow, ...]}`; pass `next` back as `cursor` to page.
        `tags` filters to workflows carrying ALL the given tags (the API ANDs them)."""
        params: dict = {"take": take}
        if cursor:
            params["cursor"] = cursor
        if exclude_failed:
            params["excludeFailed"] = "true"
        if hide_mature is not None:
            params["hideMatureContent"] = "true" if hide_mature else "false"
        if from_date:
            params["fromDate"] = from_date
        if to_date:
            params["toDate"] = to_date
        if tags:
            params["tags"] = list(tags)
        return self._request("GET", "/v2/consumer/workflows", params=params).json()

    def get_workflow(self, workflow_id: str, wait: int = 0) -> dict:
        """Fetch a workflow. `wait` (seconds) long-polls: the server holds the request until the
        workflow completes or `wait` elapses (then 202 with the current state). Falls back to a
        plain GET if the orchestration still types `wait` as a bool (400s on an integer)."""
        path = f"/v2/consumer/workflows/{workflow_id}"
        if wait and self._get_wait_supported:
            try:
                return self._request("GET", path, params={"wait": wait}, timeout=wait + 30).json()
            except CivitaiNodeError as e:
                if getattr(e, "status_code", None) != 400:
                    raise
                self._get_wait_supported = False
        return self._request("GET", path).json()

    def cancel_workflow(self, workflow_id: str) -> None:
        self._request("PUT", f"/v2/consumer/workflows/{workflow_id}", json={"status": "canceled"})

    def refresh_blob(self, blob_id: str) -> dict:
        return self._request("POST", f"/v2/consumer/blobs/{blob_id}/refresh").json()

    def download_blob(self, blob: dict) -> bytes:
        """Download blob content, refreshing the signed URL once if it has expired."""
        url = blob.get("url")
        if not url:
            raise CivitaiNodeError(
                f"Blob {blob.get('id', '?')} has no download URL (available={blob.get('available')})"
            )
        response = self.session.get(url, timeout=300)
        if response.status_code in (401, 403) and blob.get("id"):
            refreshed = self.refresh_blob(blob["id"])
            response = self.session.get(refreshed["url"], timeout=300)
        if response.status_code >= 400:
            raise CivitaiNodeError(f"Blob download failed ({response.status_code}) for blob {blob.get('id', '?')}")
        return response.content

    def upload_media(self, data: bytes, content_type: str) -> str:
        """Upload bytes via the presigned-blob endpoint; returns a URL usable as a recipe input."""
        presign = self._request("GET", "/v2/consumer/blobs/upload").json()
        upload_url = presign["uploadUrl"]
        response = self.session.post(upload_url, data=data, headers={"Content-Type": content_type}, timeout=300)
        if response.status_code >= 400:
            raise CivitaiNodeError(http_error_message(response.status_code, response.text))
        try:
            blob = response.json()
        except ValueError:
            blob = {}
        return blob.get("url") or upload_url.split("?")[0]

    def upload_blob_file(self, path: str | Path, content_type: str) -> dict:
        """Upload a local file through the consumer blob endpoint and return the blob metadata."""
        path = Path(path)
        size = path.stat().st_size
        with path.open("rb") as handle:
            response = self._request(
                "POST",
                "/v2/consumer/blobs",
                data=handle,
                headers={"Content-Type": content_type, "Content-Length": str(size)},
                timeout=300,
            )
        try:
            blob = response.json()
        except ValueError as e:
            raise CivitaiNodeError("Blob upload did not return JSON metadata") from e
        if not isinstance(blob, dict) or not (blob.get("id") or blob.get("url")):
            raise CivitaiNodeError("Blob upload did not return an id or URL")
        return blob

    @staticmethod
    def blob_air(blob: dict) -> str:
        blob_id = blob.get("id")
        if not blob_id and blob.get("url"):
            path = parse.urlparse(blob["url"]).path.rstrip("/")
            blob_id = parse.unquote(path.rsplit("/", 1)[-1])
        if not blob_id:
            raise CivitaiNodeError("Blob metadata does not contain a usable id")
        return f"urn:air:other:other:orchestrator:blob@{blob_id}"
