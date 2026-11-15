"""
RunPod REST API v2, the few calls Syntra needs (stdlib only).

Checked against docs.runpod.io on 2026-11-15: base https://api.runpod.io/v2,
`Authorization: Bearer <key>`, POST /pods with {name, image, gpu:{id,count},
cloud, disk, env (object), args (shell string), ports}, GET /pods/{id},
POST /pods/{id}/action {"action":"terminate"}; errors in RFC 9457 form
(title, status, detail). The v1 API (rest.runpod.io/v1) retires on
2026-11-15, so nothing here uses it.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

BASE = "https://api.runpod.io/v2"


class RunPodError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _explain(status: int | None, body: dict | str) -> str:
    detail = body.get("detail") or body.get("title") if isinstance(body, dict) else str(body)[:200]
    if status == 401:
        return "RunPod did not accept this API key. Create one in RunPod > Settings > API Keys"
    if status == 402 or (detail and "balance" in str(detail).lower()):
        return "RunPod needs credit first: add funds in RunPod > Billing (at least one hour of GPU time)"
    if status == 403:
        return f"RunPod refused this action for this key: {detail}"
    if status in (407,):
        return "A proxy blocks api.runpod.io on this network"
    if status is None and any(w in str(detail) for w in ("Tunnel", "403", "407", "proxy", "Proxy")):
        return "Could not reach RunPod: a proxy or firewall blocks api.runpod.io on this network"
    return f"RunPod answered HTTP {status}: {detail}" if status else f"Could not reach RunPod: check the internet connection ({detail})"


class RunPod:
    def __init__(self, api_key: str, base: str = BASE, opener=None):
        self.key = api_key
        self.base = base.rstrip("/")
        self._open = opener or urllib.request.urlopen

    def _call(self, method: str, path: str, body: dict | None = None, timeout: float = 30):
        req = urllib.request.Request(self.base + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                                              "Accept": "application/json", "User-Agent": "Syntra-AI"})
        try:
            with self._open(req, timeout=timeout) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read() or b"{}")
            except ValueError:
                body = {}
            raise RunPodError(_explain(e.code, body), e.code) from None
        except (urllib.error.URLError, OSError) as e:
            raise RunPodError(_explain(None, str(getattr(e, "reason", e)))) from None

    def check_key(self) -> int:
        """Number of pods on the account; raises RunPodError for a bad key."""
        return len(self._call("GET", "/pods").get("pods", []))

    def create_pod(self, *, name: str, image: str, gpu_id: str, cloud: str, disk_gb: int,
                   env: dict[str, str], args: str) -> dict:
        return self._call("POST", "/pods", {
            "name": name, "image": image, "gpu": {"id": gpu_id, "count": 1},
            "cloud": cloud, "disk": int(disk_gb), "env": env, "args": args, "ports": [],
        })

    def get_pod(self, pod_id: str) -> dict:
        return self._call("GET", f"/pods/{pod_id}")

    def terminate(self, pod_id: str) -> None:
        try:
            self._call("POST", f"/pods/{pod_id}/action", {"action": "terminate"})
        except RunPodError as e:
            if e.status != 404:          # already gone: that is the goal
                raise
