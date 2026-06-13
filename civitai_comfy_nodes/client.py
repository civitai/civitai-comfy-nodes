import random
import time

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

    def _request(self, method: str, path: str, *, max_tries: int = 4, **kwargs) -> requests.Response:
        url = path if path.startswith("http") else f"{self.config.base_url}{path}"
        kwargs.setdefault("timeout", 120)
        last_response = None
        for attempt in range(max_tries):
            response = self.session.request(method, url, **kwargs)
            if response.status_code not in RETRYABLE_STATUSES:
                if response.status_code >= 400:
                    raise CivitaiNodeError(http_error_message(response.status_code, response.text))
                return response
            last_response = response
            time.sleep(min(2**attempt, 15) + random.uniform(0, 1))
        raise CivitaiNodeError(http_error_message(last_response.status_code, last_response.text))

    def submit_workflow(self, step_type: str, input_payload: dict, *, wait: int = 5, whatif: bool = False) -> dict:
        params: dict = {"wait": wait}
        if whatif:
            params["whatif"] = "true"
        if self.config.allow_mature_content:
            params["hideMatureContent"] = "false"
        body = {"steps": [{"$type": step_type, "input": input_payload}]}
        return self._request("POST", "/v2/consumer/workflows", params=params, json=body).json()

    def get_workflow(self, workflow_id: str) -> dict:
        return self._request("GET", f"/v2/consumer/workflows/{workflow_id}").json()

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
