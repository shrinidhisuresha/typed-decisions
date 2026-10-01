"""Minimal HTTP front end: POST a state and typed questions, get typed answers back.

    python -m typed_decisions.server --backend transformers --model Qwen/Qwen3.5-0.8B
    python -m typed_decisions.server --backend sglang --model Qwen/Qwen3.5-9B --url http://localhost:30000

Routes:
    POST /v1/ask   {"state": ..., "questions": {name: {type, instructions, criteria}}}
                   -> Result.to_dict()
    POST /v1/outcomes {"request_id": ..., "outcomes": {name: label}}   (needs --log)
                   what actually happened, joined to the ask later for distillation
    GET  /v1/heads distilled heads being served, with their promotion evidence
    GET  /v1/stats prefix-cache hit rate as the backend counts it
    GET  /healthz

Stdlib only, one request at a time: a local torch model is not safe to call
concurrently, and HTTP backends already batch inside a single ask().
"""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import os
import time
from collections import OrderedDict, deque

from .client import Calibration, Decider
from .routing import Router, load_heads
from .traffic import TrafficLog, _outcome_index, question_to_dict
from .types import question_from_dict

MAX_BODY = 1 << 20
RECENT_ASKS = 100_000


class _Access:
    """API keys, a per-caller requests-per-minute limit, and a queue cap -- the 401 / 429 /
    529 behaviour. All off unless configured."""

    def __init__(self, api_keys=None, rate_limit_rpm=None, max_queue=64):
        self.api_keys = set(api_keys or ())
        self.rpm = rate_limit_rpm
        self.max_queue = max_queue
        self._hits: dict[str, deque] = {}
        self._waiting = 0
        self._lock = threading.Lock()

    def caller(self, headers, address) -> str | None:
        """The authenticated caller, or None if keys are required and this one is not valid."""
        auth = headers.get("Authorization") or ""
        key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if self.api_keys:
            return key if key in self.api_keys else None
        return key or address

    def retry_after(self, caller: str) -> int | None:
        """Seconds to wait if `caller` is over its rate limit; records the hit otherwise."""
        if not self.rpm:
            return None
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(caller, deque())
            while hits and now - hits[0] >= 60.0:
                hits.popleft()
            if len(hits) >= self.rpm:
                return max(1, int(60.0 - (now - hits[0])) + 1)
            hits.append(now)
        return None

    def enter(self) -> bool:
        with self._lock:
            if self._waiting >= self.max_queue:
                return False
            self._waiting += 1
            return True

    def leave(self) -> None:
        with self._lock:
            self._waiting -= 1


def make_handler(client: Decider, model: str, log: TrafficLog | None = None,
                 heads=(), min_confidence: float = 0.0, shadow_heads=(),
                 api_keys=None, rate_limit_rpm=None, max_queue: int = 64):
    lock = threading.Lock()
    router = Router(client, heads, min_confidence, shadow_heads)
    access = _Access(api_keys, rate_limit_rpm, max_queue)
    # Questions of recent asks, so an outcome can be label-checked at write time.
    # An id older than this (or from before a restart) is accepted unchecked.
    recent: OrderedDict[str, dict] = OrderedDict()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep test output quiet
            pass

        def _send(self, status: int, payload: dict, headers: dict | None = None) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for name, value in (headers or {}).items():
                self.send_header(name, str(value))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, kind: str, message: str, headers=None) -> None:
            self._send(status, {"error": {"type": kind, "message": message}}, headers)

        def _admit(self) -> bool:
            """401 / 429 before any work. Health checks are exempt; everything under /v1 is not."""
            if not self.path.startswith("/v1/"):
                return True
            caller = access.caller(self.headers, self.client_address[0])
            if caller is None:
                self._error(401, "authentication_error", "missing or invalid API key")
                return False
            wait = access.retry_after(caller)
            if wait is not None:
                self._error(429, "rate_limit_error", "rate limit exceeded", {"Retry-After": wait})
                return False
            return True

        def do_GET(self):
            if self.path == "/healthz":
                return self._send(200, {"status": "ok", "model": model})
            if not self._admit():
                return
            if self.path == "/v1/heads":
                return self._send(200, {"min_confidence": router.min_confidence, "heads": [
                    {"family": h.family, "question": question_to_dict(h.question),
                     "base": h.base, "evaluation": h.evaluation}
                    for h in router.heads.values()],
                    "shadow": [{"family": h.family, "version": h.version}
                               for h in router.shadow_heads.values()]})
            if self.path == "/v1/stats":
                stats = getattr(client.backend, "stats", None)
                if stats is None:
                    return self._send(200, {"stats": None})
                return self._send(200, {"stats": {**asdict(stats), "hit_rate": stats.hit_rate}})
            self._send(404, {"error": "not found"})

        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                raise ValueError(f"body must be 1..{MAX_BODY} bytes")
            return json.loads(self.rfile.read(length))

        def do_POST(self):
            if not self._admit():
                return
            if self.path == "/v1/ask":
                return self._ask()
            if self.path == "/v1/outcomes":
                return self._outcomes()
            self._send(404, {"error": "not found"})

        def _decide(self, state, questions):
            """The decision path: queue cap, model lock, heads/shadow
            routing, traffic log. Returns (response, request_id), or None after a 529."""
            if not access.enter():
                self._error(529, "overloaded_error", "server overloaded; retry shortly",
                            {"Retry-After": 1})
                return None
            try:
                with lock:
                    response = router.ask(state, questions)
            finally:
                access.leave()
            request_id = None
            if log is not None:
                request_id = log.record_ask(state, questions, response, model)
                with lock:
                    recent[request_id] = questions
                    while len(recent) > RECENT_ASKS:
                        recent.popitem(last=False)
            response.pop("shadow", None)   # logged above; never shown to callers
            return response, request_id

        def _outcomes(self):
            if log is None:
                return self._send(409, {"error": "outcome logging is off; start with --log"})
            try:
                data = self._body()
                request_id = data.get("request_id") if isinstance(data, dict) else None
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError("body needs 'request_id' and 'outcomes'")
                outcomes = data.get("outcomes")
                with lock:
                    known = recent.get(request_id)
                if known is not None and isinstance(outcomes, dict):
                    for name, value in outcomes.items():
                        if name not in known:
                            raise ValueError(f"request {request_id} asked no question {name!r}")
                        _outcome_index(known[name], value)
                log.record_outcome(request_id, outcomes)
            except (ValueError, TypeError) as exc:
                return self._send(400, {"error": str(exc)})
            self._send(200, {"recorded": request_id})

        def _ask(self):
            try:
                data = self._body()
                if not isinstance(data, dict) or "state" not in data:
                    raise ValueError("body needs 'state' and 'questions'")
                raw = data.get("questions")
                if not isinstance(raw, dict) or not raw:
                    raise ValueError("'questions' must be a non-empty {name: question} object")
                questions = {str(k): question_from_dict(v) for k, v in raw.items()}
            except (ValueError, TypeError) as exc:
                return self._send(400, {"error": str(exc)})
            try:
                decided = self._decide(data["state"], questions)
            except ValueError as exc:  # e.g. too many options, label allocation
                return self._send(422, {"error": str(exc)})
            if decided is None:
                return
            response, request_id = decided
            if request_id is not None:
                response["request_id"] = request_id
            self._send(200, response)

    return Handler


def build_backend(kind: str, model: str, url: str | None, device: str | None,
                  dtype: str | None):
    if kind == "mlx":
        from .backend_impls.mlx import MLXBackend
        return MLXBackend.from_pretrained(model)
    if kind == "transformers":
        import torch
        from .backend_impls.transformers import TransformersBackend, best_device
        resolved = best_device(device)
        dtype = dtype or ("float32" if resolved == "cpu" else "bfloat16")
        return TransformersBackend.from_pretrained(
            model, device=resolved, dtype=getattr(torch, dtype))
    from transformers import AutoTokenizer
    from .backend_impls.transformers import _HFTokenizer
    tokenizer = _HFTokenizer(AutoTokenizer.from_pretrained(model))
    if kind == "vllm":
        from .backend_impls.vllm import VLLMBackend
        return VLLMBackend(model, tokenizer, base_url=url or "http://localhost:8000")
    from .backend_impls.sglang import SGLangBackend
    return SGLangBackend(tokenizer, base_url=url or "http://localhost:30000")


def build_client(args) -> Decider:
    calibration = Calibration(
        permutations=args.permutations,
        contextual=args.contextual,
        temperature=args.temperature,
        basecal=args.basecal,
    )
    backend = build_backend(args.backend, args.model, args.url, args.device, args.dtype)
    reference = None
    if args.basecal > 0.0:
        if not args.reference:
            raise SystemExit("--basecal needs --reference (the -Base twin of --model)")
        reference = build_backend(args.backend, args.reference, args.reference_url,
                                  args.device, args.dtype)
    retriever = None
    if args.retriever:
        from .retrieval import EmbeddingRetriever
        retriever = EmbeddingRetriever(args.retriever, device=args.device)
    return Decider(backend, calibration=calibration, reference=reference, retriever=retriever,
               shortlist_k=args.shortlist_k)


# Measured defaults per flavour (README, docs/BENCHMARKS.md). A preset only fills settings the
# caller left at their defaults; anything passed explicitly wins.
PRESETS = {
    "cpu":   {"backend": "transformers", "model": "Qwen/Qwen3.5-0.8B", "device": "cpu",
              "permutations": 4, "contextual": True, "retriever": "Qwen/Qwen3-Embedding-0.6B"},
    "cuda":  {"backend": "transformers", "model": "Qwen/Qwen3.5-4B", "device": "cuda",
              "permutations": 4, "contextual": True, "retriever": "Qwen/Qwen3-Embedding-0.6B"},
    "apple": {"backend": "mlx", "model": "mlx-community/Qwen3.5-9B-4bit",
              "permutations": 4, "contextual": True, "retriever": "Qwen/Qwen3-Embedding-0.6B"},
}


def apply_preset(args, parser) -> None:
    for key, value in PRESETS[args.preset].items():
        if getattr(args, key) == parser.get_default(key):
            setattr(args, key, value)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m typed_decisions.server")
    parser.add_argument("--preset", choices=sorted(PRESETS),
                        help="measured defaults for your hardware; explicit flags override")
    parser.add_argument("--backend", choices=["transformers", "mlx", "vllm", "sglang"],
                        default="transformers")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--url", help="model server base URL (vllm / sglang)")
    parser.add_argument("--device", help="transformers only: cpu, mps or cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"],
                        help="transformers only; default bfloat16 on GPU, float32 on CPU")
    parser.add_argument("--heads", help="directory of distilled heads; only promoted ones serve")
    parser.add_argument("--shadow-heads", help="directory of heads to run in shadow (any "
                        "promotion state): logged for `distill shadow`, never served; needs --log")
    parser.add_argument("--head-min-confidence", type=float, default=0.0,
                        help="below this margin a head defers to the teacher")
    parser.add_argument("--api-key", action="append", default=[],
                        help="require this Bearer key on /v1/* (repeatable; or TD_API_KEYS, "
                        "comma-separated). Off when none are given.")
    parser.add_argument("--rate-limit-rpm", type=int,
                        help="requests per minute per caller before 429 (off by default)")
    parser.add_argument("--max-queue", type=int, default=64,
                        help="requests waiting for the model before 529")
    parser.add_argument("--log", help="append asks + outcomes to this JSONL (Phase 3 data)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--permutations", type=int, default=1)
    parser.add_argument("--contextual", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--retriever", nargs="?", const="Qwen/Qwen3-Embedding-0.6B",
                        help="shortlist Choices over 26 options with this embedding model "
                        "(design doc 3C); bare flag uses Qwen3-Embedding-0.6B")
    parser.add_argument("--shortlist-k", type=int, default=10)
    parser.add_argument("--basecal", type=float, default=0.0,
                        help="pool weight for the -Base twin, 0..1 (see examples/basecal.py)")
    parser.add_argument("--reference", help="BaseCal: the -Base twin, e.g. Qwen/Qwen3.5-4B-Base")
    parser.add_argument("--reference-url", help="BaseCal over vllm/sglang: the base model's server")
    args = parser.parse_args(argv)
    if args.preset:
        apply_preset(args, parser)

    log = TrafficLog(args.log) if args.log else None
    heads = []
    if args.heads:
        heads, skipped = load_heads(args.heads, device=args.device)
        for family, reason in skipped:
            print(f"not serving head {family}: {reason}")
        print(f"serving {len(heads)} distilled head(s)")
    shadow = []
    if args.shadow_heads:
        if not log:
            raise SystemExit("--shadow-heads is pointless without --log")
        shadow, _ = load_heads(args.shadow_heads, device=args.device, include_unpromoted=True)
        print(f"shadowing {len(shadow)} head(s)")
    keys = args.api_key + [k for k in os.environ.get("TD_API_KEYS", "").split(",") if k]
    handler = make_handler(build_client(args), args.model, log, heads,
                           args.head_min_confidence, shadow, api_keys=keys,
                           rate_limit_rpm=args.rate_limit_rpm, max_queue=args.max_queue)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"typed-decisions serving {args.model} via {args.backend} on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
