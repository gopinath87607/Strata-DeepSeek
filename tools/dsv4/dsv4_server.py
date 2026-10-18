#!/usr/bin/env python3
"""OpenAI- and Anthropic-compatible serving in front of dsv4_run (P6).

One request at a time: the engine owns the GPU, so a server-wide lock serializes a request from prompt rendering
through the token stream.  Each request renders the DSML head (tools/dsv4/dsml_render.py), encodes with the
artifact's tokenizer (tools/strata_tokenizer.py), runs the driver with --stream-ids --stop-eos, and translates
the id stream onto one of two wire contracts:

  POST /v1/chat/completions   OpenAI Chat: chat.completion or chat.completion.chunk SSE, [DONE]
  POST /v1/messages           Anthropic Messages: message or message_start .. message_stop SSE
  GET  /v1/models             the single served model
  GET  /health                liveness

The engine is process-per-request: the driver's ~4 s model open and a cold prefill are paid on every request.
Residency, prefix reuse and continuous batching are P7; this server is the correctness surface.

Defaults are the reference host's wire defaults (crates/ds4-server/src/parse.rs:18-20): temperature 1.0,
top_p 1.0, min_p 0.05, top_k off.  A request's temperature 0 is the engine's greedy path.  The thinking toggle
follows the engine's own default (thinking off unless asked; a bare enable is effort HIGH), through the
DSML template's <think> channel.  A generated </think> splits the reply into reasoning and content: the OpenAI
surface carries it as message.reasoning_content / delta.reasoning_content, the Anthropic surface as a
{"type": "thinking"} block.

    python3 tools/dsv4/dsv4_server.py --port 8000
    curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
        -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Hi"}],"max_tokens":32}'

Only text content is served.  Images, tools and tool results are refused with a 400 by name rather than dropped
(the vision sidecar and the tool loop are not part of this port yet).  The server binds 127.0.0.1 by default;
a non-loopback --host without --api-key is refused.
"""

import argparse
import codecs
import json
import os
import pathlib
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # tools/strata_tokenizer.py
sys.path.insert(0, str(HERE))          # dsml_render.py, dsv4_prompt.py

from strata_tokenizer import Tokenizer  # noqa: E402
import dsml_render                       # noqa: E402
from dsv4_prompt import EOS_LITERAL, USER_LITERAL, find_binary, first_part  # noqa: E402

DEFAULT_MODEL = "deepseek-v4.1-flash"
DEFAULT_TEMPERATURE = 1.0   # parse.rs:18, the wire default
DEFAULT_TOP_P = 1.0         # parse.rs:19
DEFAULT_MIN_P = 0.05        # parse.rs:20 (the engine's own default is 0.0)
DEFAULT_MAX_TOKENS = 4096   # this server's default; the reference host's own default is 393216
DEFAULT_CTX = 1048576
SPECIAL_STOPS = {EOS_LITERAL.encode(), USER_LITERAL.encode(), "<｜Assistant｜>".encode()}
THINK_DIVIDER = "</think>"


class ApiError(Exception):
    def __init__(self, status, message, type_="invalid_request_error", param=None):
        super().__init__(message)
        self.status, self.message, self.type, self.param = status, message, type_, param


def _as_int(v, name, lo=None, hi=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not v.is_integer()):
        raise ApiError(400, "%s must be an integer" % name, param=name)
    v = int(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ApiError(400, "%s is out of range" % name, param=name)
    return v


def _as_float(v, name, lo, hi, lo_open=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ApiError(400, "%s must be a number" % name, param=name)
    v = float(v)
    if not (v == v) or v < lo or v > hi or (lo_open and v == lo):
        raise ApiError(400, "%s is out of range" % name, param=name)
    return v


def sampling_of(body):
    """The four sampler knobs, with the wire defaults."""
    t = body.get("temperature")
    temperature = DEFAULT_TEMPERATURE if t is None else _as_float(t, "temperature", 0.0, 10.0)
    p = body.get("top_p")
    top_p = DEFAULT_TOP_P if p is None else _as_float(p, "top_p", 0.0, 1.0, lo_open=True)
    m = body.get("min_p")
    min_p = DEFAULT_MIN_P if m is None else _as_float(m, "min_p", 0.0, 1.0)
    if min_p >= 1.0:
        raise ApiError(400, "min_p must be below 1", param="min_p")
    k = body.get("top_k")
    top_k = 0 if k is None else _as_int(k, "top_k", 0)
    seed = body.get("seed")
    seed = secrets.randbits(63) if seed is None else _as_int(seed, "seed", 0)
    return {"temperature": temperature, "top_p": top_p, "min_p": min_p, "top_k": top_k, "seed": seed}


def text_content(content, where):
    """A message's content as one string; only text parts are served."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                kind = part.get("type") if isinstance(part, dict) else type(part).__name__
                raise ApiError(400, "%s: only text parts are served (got %r)" % (where, kind), param=where)
            out.append(part["text"])
        return "".join(out)
    raise ApiError(400, "%s must be a string or a list of text parts" % where, param=where)


class Engine:
    """One driver process per request; the lock is the GPU's."""

    def __init__(self, args):
        self.args = args
        self.binary = find_binary(args.binary)
        self.lock = threading.Lock()

    def generate(self, ids, gen, sampling, on_token):
        """Run `gen` tokens; on_token(id) per id.  Returns when the driver exits at EOS, the cap, or a failure."""
        fd, path = tempfile.mkstemp(prefix="dsv4_srv_", suffix=".txt")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(",".join(str(i) for i in ids))
            cmd = [self.binary, "--ids-file", path, "--model", self.args.model,
                   "--engram-dir", self.args.engram_dir, "--gen", str(gen),
                   "--stream-ids", "--stop-eos",
                   "--temperature", repr(sampling["temperature"]),
                   "--top-p", repr(sampling["top_p"]),
                   "--min-p", repr(sampling["min_p"]),
                   "--seed", str(sampling["seed"])]
            if sampling["top_k"]:
                cmd += ["--top-k", str(sampling["top_k"])]
            # P7: one block for the whole request by default (--chunk 0), and the layer split when asked.
            cmd += ["--chunk", str(self.args.chunk if self.args.chunk is not None else 0)]
            if self.args.devices:
                cmd += ["--devices", self.args.devices]
            if self.args.expert_cache_mib:
                cmd += ["--expert-cache-mib", self.args.expert_cache_mib]
            cmd += ["--ctx", str(self.args.ctx)]
            if self.args.gpu is not None:
                cmd += ["--gpu", str(self.args.gpu)]
            if self.args.no_engram:
                cmd.append("--no-engram")
            if self.args.engram_feed:
                cmd.append("--engram-feed")
            deadline = time.monotonic() + self.args.timeout
            proc = None
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
                for line in proc.stdout:
                    line = line.strip()
                    if not line or line.startswith("generated="):
                        continue
                    if line.lstrip("-").isdigit():
                        on_token(int(line))
                    if time.monotonic() > deadline:
                        raise ApiError(504, "generation exceeded --timeout", "timeout_error")
            finally:
                if proc is not None:
                    proc.stdout.close()
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
            if proc.returncode != 0:
                raise ApiError(500, "the engine exited %d" % proc.returncode, "engine_error")
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


class Turn:
    """Decode the id stream into reasoning/content deltas, apply stop strings, and know why it ended."""

    def __init__(self, tok, think, stops):
        self.tok, self.think, self.stops = tok, think, list(stops or [])
        self.dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.raw = ""
        self.sent_r = 0            # raw characters already handed out as reasoning
        self.sent_c = 0            # raw characters already handed out as content
        self.split = None          # index just past the first </think>, when think mode is on
        self.stop_at = None        # index where a stop string starts
        self.stop_seq = None
        self.ended = False         # an end-of-turn special token was generated
        self.stop_hold = max((len(s) for s in self.stops), default=1) - 1

    def feed(self, tid):
        """Feed one generated id; returns a list of (kind, text) deltas, kind in {"reasoning", "content"}."""
        raw = self.tok.token_bytes(tid)
        if raw in SPECIAL_STOPS:
            self.ended = True
            return self.settle()
        self.raw += self.dec.decode(raw)
        return self.settle()

    def finish(self):
        self.raw += self.dec.decode(b"", final=True)
        return self.settle(final=True)

    def settle(self, final=False):
        out = []
        limit = len(self.raw)
        if self.stop_at is None:
            for s in self.stops:
                p = self.raw.find(s)
                if p >= 0 and (self.stop_at is None or p < self.stop_at):
                    self.stop_at, self.stop_seq = p, s
        if self.stop_at is not None:
            limit = min(limit, self.stop_at)
        if self.think:
            if self.split is None:
                q = self.raw.find(THINK_DIVIDER, 0, limit)
                if q >= 0:
                    self.split = q + len(THINK_DIVIDER)
            if self.split is None:
                safe = limit if final else max(0, limit - (len(THINK_DIVIDER) - 1))
                if safe > self.sent_r:
                    out.append(("reasoning", self.raw[self.sent_r:safe]))
                    self.sent_r = safe
            else:
                r_end = self.split - len(THINK_DIVIDER)
                if r_end > self.sent_r:
                    out.append(("reasoning", self.raw[self.sent_r:r_end]))
                    self.sent_r = r_end
                if self.sent_c < self.split:
                    self.sent_c = self.split
                safe = limit if final or self.stop_at is not None else max(self.split, limit - self.stop_hold)
                if safe > self.sent_c:
                    out.append(("content", self.raw[self.sent_c:safe]))
                    self.sent_c = safe
        else:
            safe = limit if final or self.stop_at is not None else max(0, limit - self.stop_hold)
            if safe > self.sent_c:
                out.append(("content", self.raw[self.sent_c:safe]))
                self.sent_c = safe
        return out

    def finish_reason(self, generated, max_tokens):
        if self.stop_at is not None:
            return "stop", "stop_sequence"
        if self.ended or generated < max_tokens:
            return "stop", "end_turn"
        return "length", "max_tokens"


def make_prompt_openai(body, tok):
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, "messages must be a non-empty list", param="messages")
    if body.get("tools") or body.get("functions"):
        raise ApiError(400, "tools are not served yet", param="tools")
    msgs = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise ApiError(400, "messages[%d] is not an object" % i, param="messages")
        role = m.get("role")
        if role not in ("system", "user", "assistant"):
            raise ApiError(400, "messages[%d].role %r is not served" % (i, role), param="messages")
        mm = {"role": role, "content": text_content(m.get("content"), "messages[%d].content" % i)}
        if role == "assistant" and isinstance(m.get("reasoning_content"), str):
            mm["reasoning_content"] = m["reasoning_content"]
        msgs.append(mm)
    try:
        think = dsml_render.think_mode(body.get("reasoning_effort"),
                                       body.get("enable_thinking", body.get("think")))
    except ValueError as e:
        raise ApiError(400, str(e), param="reasoning_effort")
    return dsml_render.render(msgs, think=think), think != "none"


def make_prompt_anthropic(body, tok):
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, "messages must be a non-empty list", param="messages")
    if body.get("tools"):
        raise ApiError(400, "tools are not served yet", param="tools")
    msgs = []
    system = body.get("system")
    if system:
        msgs.append({"role": "system", "content": text_content(system, "system")})
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            raise ApiError(400, "messages[%d] must have role user or assistant" % i, param="messages")
        content = m.get("content")
        reasoning = ""
        if isinstance(content, list):
            texts = []
            for part in content:
                kind = part.get("type") if isinstance(part, dict) else type(part).__name__
                if kind == "text" and isinstance(part.get("text"), str):
                    texts.append(part["text"])
                elif kind == "thinking" and m["role"] == "assistant" and isinstance(part.get("thinking"), str):
                    reasoning += part["thinking"]
                else:
                    raise ApiError(400, "messages[%d]: only text parts are served (got %r)" % (i, kind),
                                   param="messages")
            content = "".join(texts)
        elif not isinstance(content, str):
            raise ApiError(400, "messages[%d].content must be a string or a list of text parts" % i,
                           param="messages")
        mm = {"role": m["role"], "content": content}
        if reasoning:
            mm["reasoning_content"] = reasoning
        msgs.append(mm)
    th = body.get("thinking") or {}
    enabled = th.get("type") == "enabled" if isinstance(th, dict) else False
    try:
        think = dsml_render.think_mode(None, enabled)
    except ValueError as e:
        raise ApiError(400, str(e), param="thinking")
    return dsml_render.render(msgs, think=think), think != "none"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "dsv4-server/0.1"

    def log_message(self, fmt, *args):
        pass   # the request line is logged once, with token counts, by the handlers

    # ---- plumbing ---------------------------------------------------------------------------------------------

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            raise ApiError(400, "the request body is not JSON")
        if not isinstance(body, dict):
            raise ApiError(400, "the request body must be a JSON object")
        return body

    def _auth(self):
        key = self.server.args.api_key
        if not key:
            return
        got = self.headers.get("Authorization", "")
        if got.startswith("Bearer "):
            got = got[7:]
        elif self.headers.get("x-api-key"):
            got = self.headers["x-api-key"]
        if not secrets.compare_digest(got, key):
            raise ApiError(401, "invalid API key", "authentication_error")

    def _json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True   # the stream has no length; the close delimits it

    def _sse(self, obj, event=None):
        if event:
            self.wfile.write(("event: %s\n" % event).encode("utf-8"))
        self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n")
        self.wfile.flush()

    def _error(self, err, anthropic):
        if anthropic:
            body = {"type": "error", "error": {"type": err.type, "message": err.message}}
        else:
            body = {"error": {"message": err.message, "type": err.type, "param": err.param, "code": None}}
        try:
            self._json(body, err.status)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ---- routing ----------------------------------------------------------------------------------------------

    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok"})
        elif self.path == "/v1/models":
            self._json({"object": "list", "data": [{"id": self.server.args.model_id, "object": "model",
                                                    "owned_by": "strata"}]})
        else:
            self._error(ApiError(404, "no such endpoint", "not_found_error"), anthropic=False)

    def do_POST(self):
        anthropic = self.path == "/v1/messages"
        try:
            self._auth()
            body = self._body()
            if self.path == "/v1/chat/completions":
                self._chat(body)
            elif anthropic:
                self._messages(body)
            else:
                raise ApiError(404, "no such endpoint", "not_found_error")
        except ApiError as e:
            self._error(e, anthropic)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:   # a bug here must not take the server down
            self._error(ApiError(500, "server error: %s" % e, "server_error"), anthropic)

    # ---- the request's shared half ----------------------------------------------------------------------------

    def _prepare(self, prompt, max_tokens):
        tok = self.server.tok
        ids = tok.encode(prompt, parse_special=True)
        if len(ids) >= self.server.args.ctx:
            raise ApiError(400, "the prompt is longer than the context (%d tokens)" % len(ids), param="messages")
        ct = self.server.args.ctx - len(ids)
        if max_tokens > ct:
            max_tokens = ct
        return ids, max_tokens

    def _log(self, model, n_prompt, n_gen, **extra):
        fields = " ".join("%s=%s" % (k, v) for k, v in extra.items())
        sys.stderr.write("[serve] %s prompt=%d gen=%d %s\n" % (model, n_prompt, n_gen, fields))
        sys.stderr.flush()

    # ---- OpenAI Chat ------------------------------------------------------------------------------------------

    def _chat(self, body):
        eng = self.server.engine
        model = body.get("model") or self.server.args.model_id
        max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
        max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else _as_int(max_tokens, "max_tokens", 1)
        sampling = sampling_of(body)
        stops = body.get("stop")
        stops = [stops] if isinstance(stops, str) else (stops or [])
        if not all(isinstance(s, str) and s for s in stops):
            raise ApiError(400, "stop must be a string or a list of non-empty strings", param="stop")
        prompt, think = make_prompt_openai(body, self.server.tok)
        ids, max_tokens = self._prepare(prompt, max_tokens)
        stream = bool(body.get("stream"))
        so = body.get("stream_options")
        include_usage = isinstance(so, dict) and bool(so.get("include_usage"))
        cid = "chatcmpl-" + secrets.token_hex(12)
        created = int(time.time())
        turn = Turn(self.server.tok, think, stops)
        n_gen = [0]
        collected = []
        t0 = time.time()

        def on_token(tid):
            n_gen[0] += 1
            deltas = turn.feed(tid)
            if stream:
                for kind, text in deltas:
                    self._chunk(cid, created, model, _delta(kind, text))
            else:
                collected.extend(deltas)

        if stream:
            self._sse_start()
            self._chunk(cid, created, model, {"role": "assistant", "content": ""})
        with eng.lock:
            self._log(model, len(ids), 0, stream=stream, seed=sampling["seed"])
            eng.generate(ids, max_tokens, sampling, on_token)
            tail = turn.finish()
        if stream:
            for kind, text in tail:
                self._chunk(cid, created, model, _delta(kind, text))
        else:
            collected.extend(tail)
        reason, _ = turn.finish_reason(n_gen[0], max_tokens)
        if stream:
            self._chunk(cid, created, model, {}, finish_reason=reason)
            if include_usage:
                self._sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                           "choices": [], "usage": _usage(len(ids), n_gen[0])})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            message, reasoning = _assemble(collected)
            msg = {"role": "assistant", "content": message}
            if reasoning:
                msg["reasoning_content"] = reasoning
            self._json({"id": cid, "object": "chat.completion", "created": created, "model": model,
                        "choices": [{"index": 0, "message": msg, "finish_reason": reason}],
                        "usage": _usage(len(ids), n_gen[0])})
        self._log(model, len(ids), n_gen[0], finish=reason, secs="%.2f" % (time.time() - t0))

    def _chunk(self, cid, created, model, delta, finish_reason=None):
        self._sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]})

    # ---- Anthropic Messages -----------------------------------------------------------------------------------

    def _messages(self, body):
        eng = self.server.engine
        model = body.get("model") or self.server.args.model_id
        if body.get("max_tokens") is None:
            raise ApiError(400, "max_tokens is required", param="max_tokens")
        max_tokens = _as_int(body.get("max_tokens"), "max_tokens", 1)
        sampling = {**sampling_of(body), "seed": secrets.randbits(63)}   # the Messages surface has no seed
        stops = body.get("stop_sequences") or []
        if not all(isinstance(s, str) and s for s in stops):
            raise ApiError(400, "stop_sequences must be non-empty strings", param="stop_sequences")
        prompt, think = make_prompt_anthropic(body, self.server.tok)
        ids, max_tokens = self._prepare(prompt, max_tokens)
        stream = bool(body.get("stream"))
        mid = "msg_" + secrets.token_hex(12)
        turn = Turn(self.server.tok, think, stops)
        n_gen = [0]
        collected = []
        writer = AnthropicBlocks(self._sse) if stream else None
        t0 = time.time()

        def on_token(tid):
            n_gen[0] += 1
            deltas = turn.feed(tid)
            if stream:
                writer.feed(deltas)
            else:
                collected.extend(deltas)

        if stream:
            self._sse_start()
            self._sse({"type": "message_start", "message": {
                "id": mid, "type": "message", "role": "assistant", "model": model, "content": [],
                "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": len(ids), "output_tokens": 0}}},
                event="message_start")
        with eng.lock:
            self._log(model, len(ids), 0, stream=stream, seed=sampling["seed"])
            eng.generate(ids, max_tokens, sampling, on_token)
            tail = turn.finish()
        reason, stop_reason = turn.finish_reason(n_gen[0], max_tokens)
        if stream:
            writer.feed(tail)
            writer.ensure_text()
            writer.close()
            self._sse({"type": "message_delta",
                       "delta": {"stop_reason": stop_reason, "stop_sequence": turn.stop_seq},
                       "usage": {"output_tokens": n_gen[0]}}, event="message_delta")
            self._sse({"type": "message_stop"}, event="message_stop")
        else:
            collected.extend(tail)
            self._json({"id": mid, "type": "message", "role": "assistant", "model": model,
                        "content": _anthropic_blocks(collected), "stop_reason": stop_reason,
                        "stop_sequence": turn.stop_seq,
                        "usage": {"input_tokens": len(ids), "output_tokens": n_gen[0]}})
        self._log(model, len(ids), n_gen[0], finish=reason, secs="%.2f" % (time.time() - t0))


def _delta(kind, text):
    return {"content": text} if kind == "content" else {"reasoning_content": text}


class AnthropicBlocks:
    """The Messages SSE block writer: a thinking block then a text block, each opened on its first delta."""

    def __init__(self, send):
        self.send = send
        self.index = -1
        self.open = None

    def feed(self, deltas):
        for kind, text in deltas:
            if not text:
                continue
            if kind != self.open:
                self.close()
                self.index += 1
                if kind == "reasoning":
                    self.send({"type": "content_block_start", "index": self.index,
                               "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
                              "content_block_start")
                else:
                    self.send({"type": "content_block_start", "index": self.index,
                               "content_block": {"type": "text", "text": ""}},
                              "content_block_start")
                self.open = kind
            if kind == "reasoning":
                self.send({"type": "content_block_delta", "index": self.index,
                           "delta": {"type": "thinking_delta", "thinking": text}}, "content_block_delta")
            else:
                self.send({"type": "content_block_delta", "index": self.index,
                           "delta": {"type": "text_delta", "text": text}}, "content_block_delta")

    def ensure_text(self):
        if self.index < 0:
            self.index = 0
            self.open = "content"
            self.send({"type": "content_block_start", "index": 0,
                       "content_block": {"type": "text", "text": ""}}, "content_block_start")

    def close(self):
        if self.open is not None:
            self.send({"type": "content_block_stop", "index": self.index}, "content_block_stop")
            self.open = None


def _assemble(deltas):
    """Buffered OpenAI: join the deltas in order."""
    reasoning, content = "", ""
    for kind, text in deltas:
        if kind == "reasoning":
            reasoning += text
        else:
            content += text
    return content, reasoning


def _anthropic_blocks(deltas):
    reasoning, content = "", ""
    for kind, text in deltas:
        if kind == "reasoning":
            reasoning += text
        else:
            content += text
    blocks = []
    if reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    blocks.append({"type": "text", "text": content})
    return blocks


def _usage(n_in, n_out):
    return {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler, args, tok, engine):
        super().__init__(addr, handler)
        self.args, self.tok, self.engine = args, tok, engine
        self.args.model_id = args.model_id or DEFAULT_MODEL


def main(argv):
    ap = argparse.ArgumentParser(description="OpenAI/Anthropic serving in front of dsv4_run")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key", default=None, help="require this key (Bearer or x-api-key)")
    ap.add_argument("--model", default="/mnt/nvme/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf")
    ap.add_argument("--model-id", default=DEFAULT_MODEL, help="the id the wire answers with")
    ap.add_argument("--tokenizer-gguf", default=None, help="default: part 1 of --model")
    ap.add_argument("--engram-dir", default="/mnt/nvme")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--chunk", type=int, default=None,
                    help="tokens per block; default 0 = auto: the largest block that fits in free VRAM, "
                         "one block for the whole request when it does (up to 4096 rows)")
    ap.add_argument("--ctx", type=int, default=DEFAULT_CTX)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--devices", default=None,
                    help="P7 layer split: comma-separated gpu list, one stage each (e.g. 0,1,2,3)")
    ap.add_argument("--expert-cache-mib", default=None,
                    help="P7 expert cache: MiB of device slots per stage (a comma list gives one per stage)")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--engram-feed", action="store_true",
                    help="P7 engram feed: read the next block's engram rows while the current block computes")
    args = ap.parse_args(argv[1:])

    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.api_key:
        sys.exit("dsv4_server: refusing a non-loopback --host without --api-key")
    tok = Tokenizer.from_gguf(args.tokenizer_gguf or first_part(args.model))
    engine = Engine(args)
    srv = Server((args.host, args.port), Handler, args, tok, engine)
    sys.stderr.write("[serve] %s on http://%s:%d (model %s, engine %s)\n"
                     % (args.model_id, args.host, args.port, args.model, engine.binary))
    sys.stderr.flush()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
