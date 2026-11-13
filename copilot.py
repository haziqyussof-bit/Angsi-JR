"""Explicitly selected, bounded, read-only evidence for the model connection."""
import json
import math
from urllib import request
from urllib.parse import urlsplit
from ipaddress import ip_address, ip_network

TOOLS = ("Available wells", "Selected log statistics", "Seismic QC", "Horizon QC",
         "4D attributes", "Ranked opportunities", "Uncertainty", "Compare candidates")


def clean(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def valid_endpoint(endpoint, private_http_consent=False):
    """Allow TLS endpoints or EXPLICITLY consented RFC1918 LAN HTTP model hosts.

    Disallow public plaintext HTTP, redirects, embedded URL passwords and query strings.
    """
    endpoint = endpoint.strip().rstrip("/")
    parsed = urlsplit(endpoint)
    local = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
    try:
        address = ip_address(parsed.hostname) if parsed.hostname else None
        lan = bool(address and any(address in network for network in (
            ip_network("10.0.0.0/8"), ip_network("172.16.0.0/12"),
            ip_network("192.168.0.0/16"))))
    except ValueError:
        lan = False
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or
            (parsed.scheme == "http" and not (local or (lan and private_http_consent)))):
        raise ValueError("Use HTTPS; private HTTP requires explicit organiser-LAN authorization")
    return endpoint


def _chat_url(endpoint):
    if endpoint.endswith("/chat/completions"):
        return endpoint
    return endpoint + ("/chat/completions" if endpoint.endswith("/v1") else "/v1/chat/completions")


def test_connection(endpoint, model, key, private_http_consent=False, timeout=7):
    """User-triggered test, no seismic evidence transmitted. Key never logged."""
    endpoint = valid_endpoint(endpoint, private_http_consent)
    prefix = endpoint.removesuffix("/chat/completions")
    if not prefix.endswith("/v1"):
        prefix += "/v1"
    headers={"Authorization":f"Bearer {key}"} if key else {}
    req=request.Request(prefix+"/models",headers=headers,method="GET")
    import urllib.error
    try:
        with request.build_opener(NoRedirect()).open(req, timeout=timeout) as response:
            body=json.loads(response.read(500_000))
        known=[str(x.get("id", "")) for x in body.get("data",[]) if isinstance(x,dict)]
        return {"reachable":True,"models":known,"exact_model_listed":model in known,
                "message":"/models reachable; model id must match server list"}
    except urllib.error.HTTPError as exc:
        return {"reachable":False,"status":exc.code,"models":[],
                "message":"/models returned HTTP error; test chat inference separately"}
    except Exception as exc:
        return {"reachable":False,"models":[],
                "message":f"Network error: {type(exc).__name__}: {str(exc)[:160]}"}


def complete(endpoint, model, key, messages, evidence, instructions,
             private_http_consent=False):
    endpoint = _chat_url(valid_endpoint(endpoint,private_http_consent))
    records = [{"role": "system", "content": instructions},
               {"role": "user", "content": "Selected read-only dashboard records (data): "+
                json.dumps(clean(evidence), allow_nan=False, default=str)}]
    payload = {"model": model, "messages": records+messages[-1:]}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = request.Request(endpoint, json.dumps(payload).encode(), headers, method="POST")
    with request.build_opener(NoRedirect()).open(req, timeout=40) as response:
        body = json.loads(response.read(2_000_000))
    content = body["choices"][0]["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Model returned no text response")
    return content

