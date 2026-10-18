#!/usr/bin/env python3
"""Text in front of dsv4_run: the tokenizer and the DSML head live in Python.

`dsv4_run` takes token ids (tools/dsv4/dsv4_run.cu).  This wrapper renders a chat turn (or takes text raw),
encodes it with the artifact's own tokenizer (tools/strata_tokenizer.py), runs the driver, and decodes what it
generated.  The tokenizer's vocab and merges are GGUF metadata, which lives in part 1, so the 40-part base is
fine: only that part is read.

    python3 tools/dsv4/dsv4_prompt.py --prompt "Hello" --gen 32
    python3 tools/dsv4/dsv4_prompt.py --prompt "Hi" --raw --gen 16 --trace

The binary defaults to $DSV4_RUN, then build/dsv4_run, then /tmp/dsv4-build/dsv4_run; --binary overrides it.
"""

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # tools/strata_tokenizer.py
sys.path.insert(0, str(HERE))          # dsml_render.py

from strata_tokenizer import Tokenizer  # noqa: E402
import dsml_render                      # noqa: E402

EOS_LITERAL = "<｜end▁of▁sentence｜>"
USER_LITERAL = "<｜User｜>"


def first_part(model: str) -> str:
    suffix = ".part01-of-40"
    if model.endswith(suffix):
        return model
    return model + suffix if pathlib.Path(model + suffix).exists() else model


def find_binary(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("DSV4_RUN")
    if env:
        return env
    root = HERE.parents[1]
    for candidate in (root / "build" / "dsv4_run", pathlib.Path("/tmp/dsv4-build/dsv4_run")):
        if candidate.exists():
            return str(candidate)
    return str(pathlib.Path("/tmp/dsv4-build/dsv4_run"))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="render, encode, run dsv4_run, decode")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="the user message (or raw text with --raw)")
    src.add_argument("--prompt-file", help="read the prompt from this file")
    ap.add_argument("--raw", action="store_true", help="send the text as-is, no chat head")
    ap.add_argument("--system", default="", help="a system message (chat mode)")
    ap.add_argument("--think", default="none", choices=sorted(dsml_render.THINK_MODES),
                    help="the thinking mode the chat head asks for (default none)")
    ap.add_argument("--gen", type=int, default=0, help="tokens to generate (greedy)")
    ap.add_argument("--model", default="/mnt/nvme/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf",
                    help="the base name or an assembled GGUF; 40 parts are found beside it")
    ap.add_argument("--tokenizer-gguf", default=None,
                    help="where the tokenizer metadata is read (default: part 1 of --model)")
    ap.add_argument("--engram-dir", default="/mnt/nvme")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--logits-out", default=None)
    ap.add_argument("--no-stop", action="store_true", help="do not trim at an end-of-sentence or user head")
    args = ap.parse_args(argv[1:])

    text = args.prompt if args.prompt is not None else pathlib.Path(args.prompt_file).read_text()
    tok_gguf = args.tokenizer_gguf or first_part(args.model)
    tok = Tokenizer.from_gguf(tok_gguf)

    if args.raw:
        prompt = text
    else:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": text})
        prompt = dsml_render.render(messages, think=args.think)
    ids = tok.encode(prompt, parse_special=True)
    print("prompt: %d tokens" % len(ids), file=sys.stderr)

    binary = find_binary(args.binary)
    fd, ids_path = tempfile.mkstemp(prefix="dsv4_ids_", suffix=".txt")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(",".join(str(i) for i in ids))
        cmd = [binary, "--ids-file", ids_path, "--model", args.model,
               "--engram-dir", args.engram_dir, "--gen", str(args.gen)]
        if args.chunk is not None:
            cmd += ["--chunk", str(args.chunk)]
        if args.ctx is not None:
            cmd += ["--ctx", str(args.ctx)]
        if args.gpu is not None:
            cmd += ["--gpu", str(args.gpu)]
        if args.trace:
            cmd.append("--trace")
        if args.no_engram:
            cmd.append("--no-engram")
        if args.no_check:
            cmd.append("--no-check")
        if args.logits_out:
            cmd += ["--logits-out", args.logits_out]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    finally:
        try:
            os.unlink(ids_path)
        except OSError:
            pass
    if proc.returncode != 0:
        print("dsv4_run exited %d" % proc.returncode, file=sys.stderr)
        return proc.returncode

    generated = []
    for line in proc.stdout.splitlines():
        if line.startswith("generated="):
            payload = line[len("generated="):].strip()
            generated = [int(x) for x in payload.split(",") if x]
    if not generated:
        print(proc.stdout, end="")
        return 0
    if not args.no_stop:
        # The model was not asked to stop: it emits the turn's end token and starts a new turn.  Decode the
        # whole run, then cut at the first end-of-sentence or user head, as a server would.
        cut = len(generated)
        for i, t in enumerate(generated):
            piece = tok.decode([t])
            if piece in (EOS_LITERAL, USER_LITERAL):
                cut = i
                break
        generated = generated[:cut]
    print("generated ids: %s" % ",".join(str(i) for i in generated), file=sys.stderr)
    print(tok.decode(generated))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
