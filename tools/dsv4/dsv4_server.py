#!/usr/bin/env python3
"""OpenAI- and Anthropic-compatible serving in front of dsv4_run (P6).

One request at a time: the engine owns the GPU, so a server-wide lock serializes a request from prompt rendering
through the token stream.  Each request renders the DSML head (tools/dsv4/dsml_render.py), encodes with the
artifact's tokenizer (tools/strata_tokenizer.py), runs the driver with --stream-ids --stop-eos, and translates
the id stream onto one of two wire contracts:

  POST /v1/chat/completions   OpenAI Chat: chat.completion or chat.completion.chunk SSE, [DONE]
  POST /v1/messages           Anthropic Messages: message or message_start .. message_stop SSE
  GET  /v1/models             the single served model
  GET  /metrics               the web page's Monitor tab: live request, host and GPU readings, history
  GET  /health                liveness
  GET  /                      the Strata chat page (serve/web), open it in a browser

The engine is resident by default: one driver process is started with --serve, loads the weights once, and then
answers one request per stdin line.  The first request pays the model load; later ones pay only the prefill and
the decode, and the driver's expert cache survives between requests.  --no-resident restores the P6
one-process-per-request path.  Prefix reuse and continuous batching are still open.

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
import queue
import secrets
import shutil
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


class Monitor:
    """Host and GPU readings for the web page's Monitor tab, plus the chart history (one point per poll)."""

    HIST = 60

    def __init__(self):
        self.lock = threading.Lock()
        self.history = {k: [] for k in ("tok_s", "prefill_tok_s_mean", "gpu_util", "gpu_mem_used",
                                        "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "disk_read_mb")}
        self.have_smi = shutil.which("nvidia-smi") is not None
        self.static = {"cpu_name": self._cpu_name(), "threads": os.cpu_count() or 1}
        cores = self._cpu_cores()
        if cores:
            self.static["cores"] = cores
        names = self._gpu_names()
        if names:
            counts = {}
            for n in names:
                counts[n] = counts.get(n, 0) + 1
            self.static["gpu_name"] = " + ".join(f"{c} x {n}" if c > 1 else n for n, c in counts.items())
        elif not self.have_smi:
            self.static["gpu_note"] = "nvidia-smi not found"
        self.prev_cpu = self._cpu()
        self.prev_disk = self._disk()
        self.prev_t = time.monotonic()
        self.cache_hw, self.cache_t = None, 0.0

    @staticmethod
    def _cpu_name():
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return None

    @staticmethod
    def _cpu_cores():
        """Physical cores from /proc/cpuinfo (one entry per (physical id, core id)); None when it does not say."""
        try:
            with open("/proc/cpuinfo") as f:
                pairs = set()
                phys = core = None
                for line in f:
                    if line.startswith("physical id"):
                        phys = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":", 1)[1].strip()
                    elif not line.strip() and phys is not None and core is not None:
                        pairs.add((phys, core))
                        phys = core = None
                if phys is not None and core is not None:
                    pairs.add((phys, core))
        except OSError:
            return None
        return len(pairs) or None

    @staticmethod
    def _cpu():
        try:
            with open("/proc/stat") as f:
                parts = f.readline().split()
            vals = [int(x) for x in parts[1:9]]
            idle = vals[3] + vals[4]
            return (sum(vals) - idle, sum(vals))
        except (OSError, ValueError, IndexError):
            return None

    @staticmethod
    def _disk():
        """(sectors read, sectors written) over the whole block devices (partitions and virtual devices skipped)."""
        rd = wr = 0
        try:
            for dev in os.listdir("/sys/block"):
                if dev.startswith(("loop", "ram", "dm-", "md", "zram", "sr")):
                    continue
                try:
                    with open("/sys/block/%s/stat" % dev) as f:
                        p = [int(x) for x in f.read().split()]
                    rd += p[2]
                    wr += p[6]
                except (OSError, ValueError, IndexError):
                    continue
        except OSError:
            return None
        return (rd, wr)

    @staticmethod
    def _ram():
        vals = {}
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    vals[k] = int(v.split()[0]) * 1024
        except (OSError, ValueError):
            return {}
        total = vals.get("MemTotal")
        avail = vals.get("MemAvailable", vals.get("MemFree"))
        return {"ram_total": total, "ram_used": total - avail if total is not None and avail is not None else None}

    @staticmethod
    def _num(s):
        try:
            return float(s.strip())
        except (AttributeError, ValueError):
            return None

    def _gpus(self):
        """Per-GPU readings from nvidia-smi (PCI order), memory in bytes; [] when there is no NVIDIA GPU."""
        if not self.have_smi:
            return []
        q = ("index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,"
             "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current")
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=4).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        gpus = []
        for line in out.splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) < 11:
                continue
            mem = self._num(f[3])
            tot = self._num(f[4])
            gpus.append({"index": int(self._num(f[0]) or 0), "name": f[1],
                         "util": self._num(f[2]),
                         "mem_used": int(mem * 1048576) if mem is not None else None,
                         "mem_total": int(tot * 1048576) if tot is not None else None,
                         "temp": self._num(f[5]), "power": self._num(f[6]),
                         "power_limit": self._num(f[7]),
                         "pcie_gen": self._num(f[8]), "pcie_gen_max": self._num(f[9]),
                         "pcie_width": self._num(f[10])})
        return gpus

    def _gpu_names(self):
        if not self.have_smi:
            return []
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=4).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        return [x.strip() for x in out.splitlines() if x.strip()]

    def sample(self, live):
        """One poll's hardware readings; also appends to the chart history."""
        with self.lock:
            now = time.monotonic()
            if self.cache_hw is not None and now - self.cache_t < 0.9:
                return dict(self.cache_hw)     # two open pages poll in the same second: one nvidia-smi
            dt = max(1e-3, now - self.prev_t)
            hw = {}
            cpu_now, disk_now = self._cpu(), self._disk()
            if cpu_now and self.prev_cpu:
                busy = cpu_now[0] - self.prev_cpu[0]
                total = cpu_now[1] - self.prev_cpu[1]
                hw["cpu"] = 100.0 * busy / total if total > 0 else None
            if disk_now and self.prev_disk:
                hw["disk_read_mb"] = max(0.0, (disk_now[0] - self.prev_disk[0]) * 512 / 1e6 / dt)
                hw["disk_write_mb"] = max(0.0, (disk_now[1] - self.prev_disk[1]) * 512 / 1e6 / dt)
            hw.update(self._ram())
            gpus = self._gpus()
            if gpus:
                utils = [g["util"] for g in gpus if g["util"] is not None]
                mems = [g["mem_used"] for g in gpus if g["mem_used"] is not None]
                tots = [g["mem_total"] for g in gpus if g["mem_total"] is not None]
                temps = [g["temp"] for g in gpus if g["temp"] is not None]
                pows = [g["power"] for g in gpus if g["power"] is not None]
                lims = [g["power_limit"] for g in gpus if g["power_limit"] is not None]
                hw.update({
                    "gpus": [{"index": g["index"], "util": g["util"], "mem_used": g["mem_used"],
                              "mem_total": g["mem_total"], "temp": g["temp"]} for g in gpus],
                    "gpu_util": sum(utils) / len(utils) if utils else None,
                    "gpu_mem_used": sum(mems) if mems else None,
                    "gpu_mem_total": sum(tots) if tots else None,
                    "gpu_temp": max(temps) if temps else None,
                    "gpu_power": sum(pows) if pows else None,
                    "gpu_power_limit": sum(lims) if lims else None,
                    "gpu_pcie_gen": gpus[0]["pcie_gen"], "gpu_pcie_gen_max": gpus[0]["pcie_gen_max"],
                    "gpu_pcie_width": gpus[0]["pcie_width"],
                })
            self.prev_cpu, self.prev_disk, self.prev_t = cpu_now, disk_now, now
            series = {"tok_s": live.get("tok_s") if live.get("state") == "generating" else None,
                      "prefill_tok_s_mean": live.get("prefill_tok_s_mean"),
                      "gpu_util": hw.get("gpu_util"), "gpu_mem_used": hw.get("gpu_mem_used"),
                      "gpu_temp": hw.get("gpu_temp"), "gpu_power": hw.get("gpu_power"),
                      "gpu_pcie_rx_mb": None, "cpu": hw.get("cpu"), "disk_read_mb": hw.get("disk_read_mb")}
            for k, v in series.items():
                arr = self.history[k]
                arr.append(v)
                if len(arr) > self.HIST:
                    del arr[0]
            self.cache_hw, self.cache_t = hw, now
            return dict(hw)


class Engine:
    """The driver; the handler's lock is the GPU's.

    Resident by default: one driver with --serve loads the weights once, prints ready, and answers one request
    per stdin line.  --no-resident keeps the P6 one-process-per-request path.
    """

    SLOT_BYTES = 7208056   # the expert cache's slot (the largest payload), as the driver prints it

    def __init__(self, args):
        self.args = args
        self.binary = find_binary(args.binary)
        self.lock = threading.Lock()
        self.proc = None
        self.proc_ready = False
        self.q = None       # the current driver's stdout lines, fed by a reader thread
        # Monitor tab state: the live request, the finished-request table and its totals.
        self.stats_lock = threading.Lock()
        self.live = {"state": "idle"}
        self.requests = []
        self.totals = {"requests": 0, "prompt_tokens": 0, "reused": 0, "prompt_ms": 0.0,
                       "output_tokens": 0, "decode_ms": 0.0}
        self.since = time.time()
        self.monitor = Monitor()

    # ---- resident mode (the default) ----

    def _serve_cmd(self):
        cmd = [self.binary, "--serve", "--serve-ctx", str(self.args.serve_ctx),
               "--model", self.args.model, "--engram-dir", self.args.engram_dir,
               "--gen", "0", "--chunk", str(self.args.chunk if self.args.chunk is not None else 0),
               "--ctx", str(self.args.ctx), "--stream-ids"]
        if self.args.devices:
            cmd += ["--devices", self.args.devices]
        if self.args.expert_cache_mib:
            cmd += ["--expert-cache-mib", self.args.expert_cache_mib]
        if self.args.gpu is not None:
            cmd += ["--gpu", str(self.args.gpu)]
        if self.args.no_engram:
            cmd.append("--no-engram")
        if self.args.engram_feed:
            cmd.append("--engram-feed")
        if self.args.ram_image:
            cmd.append("--ram-image")
        return cmd

    def _kill(self):
        proc = self.proc
        self.proc = None
        self.proc_ready = False
        self.q = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass

    def _read_lines(self, proc, q):
        """Feed the driver's stdout lines to a queue; None marks EOF.  One daemon thread per driver."""
        try:
            for line in proc.stdout:
                q.put(line)
        except (OSError, ValueError):
            pass
        finally:
            q.put(None)

    def _await_line(self, deadline, what):
        """The next stdout line, or a 504 once the deadline passes; EOF is None.  Called with the lock held."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self._kill()
            raise ApiError(504, "the engine did not %s in time" % what, "timeout_error")
        try:
            return self.q.get(timeout=remaining)
        except queue.Empty:
            self._kill()
            raise ApiError(504, "the engine did not %s in time" % what, "timeout_error")

    def _ensure(self):
        """Start the resident driver if it is not up, and wait for its ready line.  Called with the lock held."""
        if self.proc is not None and self.proc_ready and self.proc.poll() is None:
            return
        self._kill()
        proc = subprocess.Popen(self._serve_cmd(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                text=True, bufsize=1)
        q = queue.Queue()
        threading.Thread(target=self._read_lines, args=(proc, q), daemon=True).start()
        self.q = q
        deadline = time.monotonic() + max(600.0, self.args.timeout)
        try:
            while True:
                line = self._await_line(deadline, "become ready")
                if line is None:
                    break
                s = line.strip()
                if s == "ready":
                    self.proc, self.proc_ready = proc, True
                    return
                if s.startswith("error="):
                    raise ApiError(500, s[len("error="):], "engine_error")
        finally:
            if self.proc is not proc:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                if proc.stdout is not None:
                    try:
                        proc.stdout.close()
                    except OSError:
                        pass
        raise ApiError(500, "the engine exited before it was ready", "engine_error")

    def warm(self):
        """Load the model before the first request.  The handler's lock serializes it with generate()."""
        with self.lock:
            try:
                self._ensure()
            except Exception:
                pass

    def close(self):
        with self.lock:
            self._kill()

    KEEP = 200        # the finished-request table the Monitor tab reads

    def _set_live(self, live):
        with self.stats_lock:
            self.live = live

    def generate(self, ids, gen, sampling, on_token):
        """Run `gen` tokens; on_token(id) per id.  Returns the request's record for the Monitor tab.

        The record's `finish` is set by the handler once the turn's stop reason is known (finished()); a
        client that hangs up or an engine failure is recorded here, because no reason will arrive.
        """
        t0 = time.monotonic()
        prompt_tokens = len(ids)
        max_tokens = int(gen)
        at = {"first": None, "last": None, "n": 0, "prefill": None}
        rec = {"time": time.time(), "finish": None, "prompt_tokens": prompt_tokens, "reused": 0,
               "output_tokens": 0, "prompt_ms": 0.0, "decode_tok_s": None, "decode_ms": 0.0,
               "duration_s": 0.0, "hit_rate": None}
        self._set_live({"state": "reading", "prompt_tokens": prompt_tokens, "prompt_read": 0,
                        "prompt_total": prompt_tokens, "max_tokens": max_tokens, "generated": 0,
                        "tok_s": None, "prefill_tok_s_mean": None})

        def tick(tid):
            now = time.monotonic()
            if at["first"] is None:
                at["first"] = now
                spent = now - t0
                at["prefill"] = prompt_tokens / spent if spent > 1e-3 else None
            at["n"] += 1
            at["last"] = now
            spent = now - at["first"]
            tok_s = (at["n"] - 1) / spent if at["n"] > 1 and spent > 1e-3 else None
            self._set_live({"state": "generating", "prompt_tokens": prompt_tokens,
                            "prompt_read": prompt_tokens, "prompt_total": prompt_tokens,
                            "max_tokens": max_tokens, "generated": at["n"], "tok_s": tok_s,
                            "prefill_tok_s_mean": at["prefill"]})
            on_token(tid)

        try:
            if getattr(self.args, "no_resident", False):
                self._generate_once(ids, max_tokens, sampling, tick)
            else:
                self._generate_resident(ids, max_tokens, sampling, tick)
        except (BrokenPipeError, ConnectionResetError):
            self._fill_record(rec, at, t0)
            self.finished(rec, "disconnect")
            raise
        except BaseException:
            self._fill_record(rec, at, t0)
            self.finished(rec, "error")
            raise
        self._fill_record(rec, at, t0)
        return rec

    @staticmethod
    def _fill_record(rec, at, t0):
        first, last, n = at["first"], at["last"], at["n"]
        rec["output_tokens"] = n
        if first is not None:
            rec["prompt_ms"] = round((first - t0) * 1000.0, 1)   # time to first token, prefill included
        if n > 1 and last > first:
            rec["decode_ms"] = round((last - first) * 1000.0, 1)
            rec["decode_tok_s"] = (n - 1) / (last - first)
        rec["duration_s"] = round(time.monotonic() - t0, 2)

    def finished(self, rec, reason):
        """Close out a request: set its finish reason, add it to the table and the totals, and go idle."""
        with self.stats_lock:
            rec["finish"] = reason
            self.requests.insert(0, rec)
            del self.requests[self.KEEP:]
            t = self.totals
            t["requests"] += 1
            t["prompt_tokens"] += rec["prompt_tokens"]
            t["reused"] += rec["reused"]
            t["prompt_ms"] += rec["prompt_ms"]
            t["output_tokens"] += rec["output_tokens"]
            t["decode_ms"] += rec["decode_ms"]
            self.live = {"state": "idle"}

    def metrics(self):
        """The Monitor tab's report: the live request, host and GPU readings, the engine's shape, the history."""
        with self.stats_lock:
            live = dict(self.live)
            requests = [dict(r) for r in self.requests]
            totals = {**self.totals, "since": self.since}
        hw = self.monitor.sample(live)
        with self.monitor.lock:
            history = {k: list(v) for k, v in self.monitor.history.items()}
        caches = []
        for part in str(self.args.expert_cache_mib or "").split(","):
            try:
                caches.append(max(0, int(part)))
            except ValueError:
                pass
        cache_mib = sum(caches)
        eng = {"model": self.args.model_id, "max_context": self.args.ctx,
               "expert_cache_mib": cache_mib,
               "expert_slots": sum((m * 1048576) // self.SLOT_BYTES for m in caches),
               "images": False}
        return {"live": live, "hardware": hw, "hardware_static": self.monitor.static,
                "engine": eng, "history": history, "requests": requests, "totals": totals,
                "requests_kept": len(requests)}

    def _generate_resident(self, ids, gen, sampling, on_token):
        """The resident driver: one request line, then its ids until `done`.  The lock is held by the caller."""
        self._ensure()
        req = ("ids=%s gen=%d temperature=%r top_p=%r min_p=%r top_k=%d seed=%d stop_eos=1\n"
               % (",".join(str(i) for i in ids), gen, float(sampling["temperature"]),
                  float(sampling["top_p"]), float(sampling["min_p"]), int(sampling["top_k"]),
                  int(sampling["seed"])))
        try:
            self.proc.stdin.write(req)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._kill()
            raise ApiError(500, "the engine stopped accepting requests", "engine_error")
        deadline = time.monotonic() + self.args.timeout
        err = None
        done = False
        try:
            while True:
                line = self._await_line(deadline, "answer")
                if line is None:
                    break
                s = line.strip()
                if not s:
                    continue
                if s == "done":
                    done = True
                    break
                if s.startswith("error="):
                    err = s[len("error="):]
                    continue
                if s.startswith("generated="):
                    continue
                if s.lstrip("-").isdigit():
                    on_token(int(s))
        finally:
            if not done:
                # A client that hung up mid-answer leaves the driver generating; its leftover ids would be
                # read as the next request's, so the driver goes with the connection.
                self._kill()
        if not done:
            raise ApiError(500, "the engine stopped answering", "engine_error")
        if err:
            raise ApiError(500, err, "engine_error")

    def _generate_once(self, ids, gen, sampling, on_token):
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
            if self.args.ram_image:
                cmd.append("--ram-image")
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

    _STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                     ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".json": "application/json",
                     ".png": "image/png", ".ico": "image/x-icon", ".woff": "font/woff",
                     ".woff2": "font/woff2", ".ttf": "font/ttf"}

    def _static(self, path):
        """Serve the Strata web app (the clone's serve/web): / is index.html, web/<file> is the asset."""
        root = self.server.web_dir
        if root is None:
            return False
        rel = path.lstrip("/") or "index.html"
        if rel.startswith("web/"):
            rel = rel[len("web/"):]
        elif rel not in ("index.html", "monitor.html"):
            return False
        target = (root / rel).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return False
        if not target.is_file():
            return False
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", self._STATIC_TYPES.get(target.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)
        return True

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            # The web app reads these; the engine is text-only and has no image generator.
            self._json({"status": "ok", "model": self.server.args.model_id, "images": False,
                        "image_generation": False, "max_context": self.server.args.ctx})
        elif path == "/metrics":
            self._json(self.server.engine.metrics())
        elif path in ("/config", "/settings", "/mcp"):
            self._error(ApiError(404, "no such endpoint", "not_found_error"), anthropic=False)
        elif path == "/v1/models":
            self._json({"object": "list", "data": [{"id": self.server.args.model_id, "object": "model",
                                                    "owned_by": "strata"}]})
        elif self._static(path):
            pass
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
            rec = eng.generate(ids, max_tokens, sampling, on_token)
            tail = turn.finish()
        reason, _ = turn.finish_reason(n_gen[0], max_tokens)
        eng.finished(rec, reason)
        if stream:
            for kind, text in tail:
                self._chunk(cid, created, model, _delta(kind, text))
        else:
            collected.extend(tail)
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
            rec = eng.generate(ids, max_tokens, sampling, on_token)
            tail = turn.finish()
        reason, stop_reason = turn.finish_reason(n_gen[0], max_tokens)
        eng.finished(rec, reason)
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
        # The chat page is the clone's own serve/web; --web-dir points it elsewhere, a missing dir leaves the API only.
        root = pathlib.Path(args.web_dir).expanduser() if args.web_dir else HERE.parent.parent / "serve" / "web"
        self.web_dir = root.resolve() if root.is_dir() else None
        # Warm the resident driver while the port opens; the first generate() waits on the same lock.
        if not getattr(args, "no_resident", False):
            threading.Thread(target=engine.warm, daemon=True).start()


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
    ap.add_argument("--ram-image", action="store_true",
                    help="load the whole image into host RAM once at start; expert misses are staged from there")
    ap.add_argument("--no-resident", action="store_true",
                    help="one driver process per request (the P6 path); the default keeps the model resident")
    ap.add_argument("--serve-ctx", type=int, default=16384,
                    help="resident mode: the driver's token budget (default 16384; a longer request is refused)")
    ap.add_argument("--web-dir", default=None,
                    help="the chat page's files (default: the clone's serve/web; a missing dir serves the API only)")
    args = ap.parse_args(argv[1:])

    if args.serve_ctx < 1:
        sys.exit("dsv4_server: --serve-ctx must be at least 1")

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
        engine.close()
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
