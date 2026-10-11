"""Budget-capped client for Jev (typesafe/jev-1.13) on OpenRouter.

Spending ceiling (two layers):
  1. Client-side: every request reserves a conservative cost estimate BEFORE it is sent
     (input characters / 2 tokens, i.e. about twice the usual token count, at $0.042 per 1M tokens).
     A request whose reservation would push spent + reserved above CAP_USD is refused, and nothing
     is sent. After the response, the reservation is replaced by the cost OpenRouter reports
     (or by the estimate if it reports none). The ledger is written atomically to jev_ledger.json,
     so the ceiling survives kernel restarts.
  2. Server-side: check_key() reads OpenRouter's own usage counter for the key (GET /api/v1/key);
     run() stops if that counter, relative to its value at the start, reaches CAP_USD.
The API key is read from the OPENROUTER_API_KEY environment variable and is never printed or logged.
"""
import json, os, threading, time
import requests

MODEL = "typesafe/jev-1.13"
PRICE_PER_TOKEN = 0.042 / 1e6
CAP_USD = 3.00   # raised from 2.00 at the user's request (2026-10-09)
LEDGER = "jev_ledger.json"
URL_SYSTEMONE = "https://openrouter.ai/api/v1/systemone"
URL_DECISIONS = "https://openrouter.ai/api/alpha/decisions"
URL_KEY = "https://openrouter.ai/api/v1/key"
_lock = threading.Lock()


def _headers():
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        raise RuntimeError("OPENROUTER_API_KEY not set in this cell (declare the credential on the call)")
    return {"Authorization": f"Bearer {k}", "Content-Type": "application/json"}


def _load():
    if os.path.exists(LEDGER):
        return json.load(open(LEDGER))
    return {"cap_usd": CAP_USD, "spent_usd": 0.0, "reserved_usd": 0.0, "requests": 0, "refused": 0,
            "key_usage_at_start": None, "log": []}


def _save(L):
    tmp = LEDGER + ".tmp"
    json.dump(L, open(tmp, "w"), indent=1)
    os.replace(tmp, LEDGER)


OVERHEAD_TOKENS = 400

def estimate_usd(payload):
    """Conservative: 1 token per 2 characters of the serialized request, plus a fixed
    server-side prompt overhead. Test calls (2026-10-09) billed 367-420 input tokens for
    ~400-char requests, i.e. ~270 tokens of hidden overhead; we reserve 400."""
    return (len(json.dumps(payload)) / 2 + OVERHEAD_TOKENS) * PRICE_PER_TOKEN


def check_key():
    """OpenRouter's own record for this key: usage (USD) and limit, if any."""
    r = requests.get(URL_KEY, headers=_headers(), timeout=30)
    r.raise_for_status()
    d = r.json().get("data", r.json())
    return {k: d.get(k) for k in ("usage", "limit", "limit_remaining", "is_free_tier", "rate_limit")}


def call(payload, url=URL_SYSTEMONE, timeout=60):
    """Send one request under the ceiling. Returns (status_code, json_or_text, cost_usd, latency_s)."""
    est = estimate_usd(payload)
    with _lock:
        L = _load()
        if L["spent_usd"] + L["reserved_usd"] + est > L["cap_usd"]:
            L["refused"] += 1; _save(L)
            raise RuntimeError(f"budget ceiling: spent {L['spent_usd']:.4f} + reserved {L['reserved_usd']:.4f} "
                               f"+ this {est:.6f} > cap {L['cap_usd']:.2f} USD")
        L["reserved_usd"] += est; _save(L)
    t0 = time.time(); cost = est
    try:
        for attempt in range(4):                       # 1 try + up to 3 retries on transport errors
            try:
                r = requests.post(url, headers=_headers(), json=payload, timeout=timeout)
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt); continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** attempt + (5 if r.status_code == 429 else 0)); continue
            break
        lat = time.time() - t0
        try:
            body = r.json()
        except ValueError:
            body = r.text
        if isinstance(body, dict):
            u = body.get("usage") or {}
            if isinstance(u, dict) and u.get("cost") is not None:
                cost = float(u["cost"])
            elif isinstance(u, dict) and (u.get("prompt_tokens") or u.get("input_tokens")):
                cost = float(u.get("prompt_tokens") or u.get("input_tokens")) * PRICE_PER_TOKEN
        if r.status_code >= 400:
            cost = 0.0 if r.status_code in (400, 401, 402, 403, 404, 429) else cost
        return r.status_code, body, cost, lat
    finally:
        with _lock:
            L = _load()
            L["reserved_usd"] = max(0.0, L["reserved_usd"] - est)
            L["spent_usd"] += cost; L["requests"] += 1
            L["log"].append([round(time.time()), round(cost, 8)])
            L["log"] = L["log"][-200:]
            _save(L)


def status():
    L = _load()
    return {k: L[k] for k in ("cap_usd", "spent_usd", "reserved_usd", "requests", "refused", "key_usage_at_start")}


def start_session():
    """Record OpenRouter's usage counter for the key once, before the first request of a collection."""
    k = check_key()
    with _lock:
        L = _load()
        if L["key_usage_at_start"] is None:
            L["key_usage_at_start"] = float(k.get("usage") or 0.0)
            _save(L)
    return k


def server_spent():
    """USD charged to the key since start_session(), according to OpenRouter."""
    L = _load()
    if L["key_usage_at_start"] is None:
        raise RuntimeError("call start_session() before run()")
    return float(check_key().get("usage") or 0.0) - L["key_usage_at_start"]


def run(payloads, url=URL_SYSTEMONE, workers=8, check_every=200, on_result=None):
    """Send payloads (list of (item_id, payload)) with `workers` threads under BOTH ceilings.
    Layer 1: call() refuses any request that would exceed CAP_USD on the local ledger.
    Layer 2: before the first batch and after every `check_every` requests, OpenRouter's own usage
    counter for the key is read; the run stops if the amount charged since start_session() reaches
    CAP_USD minus a 2% margin. Returns (results dict item_id -> (status, body, cost, latency), stop_reason)."""
    from concurrent.futures import ThreadPoolExecutor
    start_session()
    results, stop_reason = {}, None
    margin = 0.98 * _load()["cap_usd"]
    for b0 in range(0, len(payloads), check_every):
        spent_srv = server_spent()
        if spent_srv >= margin:
            stop_reason = f"server-side ceiling: {spent_srv:.4f} USD charged since start >= {margin:.2f}"
            break
        batch = payloads[b0:b0 + check_every]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(call, p, url): iid for iid, p in batch}
            for f, iid in futs.items():
                try:
                    results[iid] = f.result()
                except RuntimeError as e:            # local ceiling reached
                    stop_reason = str(e)
                if on_result and iid in results:
                    on_result(iid, results[iid])
        if stop_reason:
            break
    return results, stop_reason
