"""The DeepSeek-V4.1 DSML prompt head, as the reference server renders it.

A byte-for-byte port of `render_dsml_chat_choice_ex` with the V4.1 spaced head
(`DsmlHead::V41 { system_token: true }`) from the reference's
`crates/ds4-server/src/render.rs`, which reproduces the official `encoding.py`.
It is the oracle for `serve/chat_template_deepseek4.jinja`: the template test
renders the same conversations through both and compares bytes, and the
reference's own golden vectors (`crates/ds4-server/tests/render.rs`) pin this
module first.

The head, in order:

    <｜begin▁of▁sentence｜>
    [<｜System｜> if there is a system body or an effort prefix]
    [effort prefix: the 75 (HIGH) or 100 (MAX) line]
    [system messages joined by a blank line, then the tools prompt]
    per message, skipping system:
        user    : <｜User｜> first, a blank line for a user-side block that
                  follows another one without an assistant reply between
        tool    : the same user block, <tool_result>escaped</tool_result>
        assistant: <｜Assistant｜> when an assistant head is pending, then
                  </think> (plain) or [<think>reasoning</think>] or </think>,
                  the content, the DSML call block, <｜end▁of▁sentence｜>
    tail, add_generation_prompt only: <｜Assistant｜> then <think> or </think>

The spaced generation's tags all open with a space: `<｜DSML｜ calls>`,
`<｜DSML｜ invoke`, `<｜DSML｜ parameter`; the string values are raw and only
a closing parameter tag inside one is escaped (as `&lt;`).

Message dicts are Strata's normalized shape, the one its templates receive:
role, content (a string, or a list of {"type": "text"|"image"} parts when the
message carries an image), reasoning_content, and tool_calls as
[{"function": {"name", "arguments": <mapping>}}].

    python3 -m tools.dsv4.dsml_render < conversation.json     # a prompt on stdout

This module is data-shaping only; it does not run a model.
"""
from __future__ import annotations

import json
import sys

BOS = "<｜begin▁of▁sentence｜>"
SYSTEM_TOKEN = "<｜System｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
EOS = "<｜end▁of▁sentence｜>"
IMAGE = "<｜deepseek_image｜>"
THINK_START = "<think>"
THINK_END = "</think>"

TOOL_CALLS_START = "<｜DSML｜ calls>"
TOOL_CALLS_END = "</｜DSML｜ calls>"
INVOKE_START = "<｜DSML｜ invoke"
INVOKE_END = "</｜DSML｜ invoke>"
PARAM_START = "<｜DSML｜ parameter"
PARAM_END = "</｜DSML｜ parameter>"

THINK_HIGH_PREFIX = (
    "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n")
THINK_MAX_PREFIX = (
    "Reasoning Effort: 100 (range 1-100, the higher the value, the more thorough the reasoning)\n\n")

TOOLS_HEAD = (
    "## Tools\n\n"
    "You have access to a set of tools to help answer the user's question. "
    'You can invoke tools by writing a "' + TOOL_CALLS_START + '" block like the following:\n\n'
    + TOOL_CALLS_START + "\n"
    + INVOKE_START + ' name="$TOOL_NAME">\n'
    + PARAM_START + ' name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE' + PARAM_END + "\n"
    "...\n"
    + INVOKE_END + "\n"
    + INVOKE_START + ' name="$TOOL_NAME2">\n...\n'
    + INVOKE_END + "\n"
    + TOOL_CALLS_END + "\n\n"
    'String parameters should be specified as is and set `string="true"`. '
    "For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set "
    '`string="false"`.\n\n'
    "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside "
    "<think>...</think> BEFORE any tool calls or final response.\n\n"
    "Otherwise, output directly after </think> with tool calls or final response.\n\n"
    "### Available Tool Schemas\n\n")
TOOLS_TAIL = ("\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool "
              "calls.\n")

IMAGE_TYPES = ("image", "image_url", "input_image")
USER_LIKE = ("user", "tool", "function")
SYSTEM_LIKE = ("system", "developer")

# The reference's parse_reasoning_effort_name (route/mod.rs L355-363).
THINK_MODES = {"none": "none", "off": "none",
               "low": "low", "medium": "low", "minimal": "low",
               "high": "high", "xhigh": "high",
               "max": "max"}


def think_mode(effort=None, enabled=None) -> str:
    """The reference's (thinking_enabled, effort) -> mode.  The engine's default is thinking OFF, effort HIGH
    (parse.rs L2838-2843): no effort and no flag is `none`, an explicit enable with no effort is `high`."""
    mode = THINK_MODES.get(str(effort).strip().lower()) if effort not in (None, "") else None
    if enabled is False or mode == "none":
        return "none"
    if enabled is True and mode is None:
        return "high"
    if enabled is None and effort in (None, ""):
        return "none"
    if mode is None:
        raise ValueError("unknown reasoning effort %r: use none, low, medium, high or max" % (effort,))
    return mode


def _parts(content, images: bool = True) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if images and part.get("type") in IMAGE_TYPES:
            out.append(IMAGE)
        elif "text" in part:
            out.append(part["text"])
    return "".join(out)


def _attr(s) -> str:
    """append_dsml_attr_escaped: & < > " in that order, each as one entity."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _sentinel(s: str, end: str, repl: str) -> str:
    """append_sentinel_escape: every occurrence of `end` loses its `<` to `repl`."""
    return s.replace(end, repl + end[1:])


def _param(key, value) -> str:
    name = PARAM_START + ' name="' + _attr(key) + '" string="'
    if isinstance(value, str):
        return name + 'true">' + _sentinel(value, PARAM_END, "&lt;") + PARAM_END
    # pyjson: Python's json.dumps shape, raw non-ASCII; the closing tag's "<" becomes <
    return name + 'false">' + _sentinel(json.dumps(value, ensure_ascii=False), PARAM_END, "\\u003c") + PARAM_END


def _calls(text: str, calls) -> str:
    """append_dsml_tool_calls_text's spaced branch.  `calls` is a list of {"function": {...}} or bare calls."""
    if not calls:
        return text
    text += "\n\n" + TOOL_CALLS_START + "\n"
    for i, call in enumerate(calls):
        fn = call.get("function", call) if isinstance(call, dict) else call
        if i > 0:
            text += "\n"
        text += INVOKE_START + ' name="' + _attr(fn.get("name", "")) + '">\n'
        args = fn.get("arguments")
        if args is None:
            args = {}
        if isinstance(args, dict):
            for j, (key, value) in enumerate(args.items()):
                if j:
                    text += "\n"
                text += _param(key, value)
            text += "\n"          # zero parameters: the invoke body is one empty line (reference `spaced`)
        else:
            # the reference's json_args_parse failure: one raw "arguments" parameter
            text += _param("arguments", args if isinstance(args, str) else json.dumps(args, ensure_ascii=False))
            text += "\n"
        text += INVOKE_END
    return text + "\n" + TOOL_CALLS_END


def _escaped_tool_result(content) -> str:
    """append_tool_result_text over the rendered content: a closing </tool_result> inside it keeps its text."""
    return _sentinel(_parts(content), "</tool_result>", "&lt;")


def render(messages, tools=None, think: str = "none", add_generation_prompt: bool = True) -> str:
    """The prompt for `messages` (Strata's shape), `tools` (schemas), and a think mode from `think_mode()`."""
    thinking = think != "none"
    effort = {"none": "", "low": "", "high": THINK_HIGH_PREFIX, "max": THINK_MAX_PREFIX}.get(think)
    if effort is None:
        raise ValueError("unknown think mode %r" % (think,))

    sys_text = ""
    for m in messages:
        if m.get("role") in SYSTEM_LIKE:
            if sys_text:
                sys_text += "\n\n"
            sys_text += _parts(m.get("content"), images=False)
    tool_text = ""
    if tools:
        tool_text = TOOLS_HEAD + "\n".join(json.dumps(t, ensure_ascii=False) for t in tools) + TOOLS_TAIL
        if sys_text:
            sys_text += "\n\n"
        sys_text += tool_text

    tool_context = bool(tools) or any(
        m.get("role") in ("tool", "function") or (m.get("role") == "assistant" and m.get("tool_calls"))
        for m in messages)
    last_user = max((i for i, m in enumerate(messages) if m.get("role") in USER_LIKE), default=-1)

    out = BOS
    if effort or sys_text:
        out += SYSTEM_TOKEN
    out += effort + sys_text

    pending = pending_tool = False
    for i, m in enumerate(messages):
        role = m.get("role")
        if role in SYSTEM_LIKE:
            continue
        if role == "user":
            out += "\n\n" if pending else USER
            out += _parts(m.get("content"))
            pending, pending_tool = True, False
        elif role in ("tool", "function"):
            if pending:
                out += "\n\n"
            elif not pending_tool:
                out += USER
            out += "<tool_result>" + _escaped_tool_result(m.get("content")) + "</tool_result>"
            pending, pending_tool = True, True
        elif role == "assistant":
            if pending:
                out += ASSISTANT
                if thinking:
                    if tool_context or i > last_user:
                        reasoning = m.get("reasoning_content")
                        out += THINK_START + (reasoning if isinstance(reasoning, str) else "") + THINK_END
                    else:
                        out += THINK_END
                else:
                    out += THINK_END
            out += _parts(m.get("content"))
            out = _calls(out, m.get("tool_calls"))
            out += EOS
            pending = pending_tool = False
        # any other role: the reference's loop ignores it

    if add_generation_prompt and pending:
        out += ASSISTANT + (THINK_START if thinking else THINK_END)
    return out


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        sys.exit(__doc__)
    spec = json.load(sys.stdin)
    prompt = render(spec.get("messages") or [], spec.get("tools"),
                    think_mode(spec.get("reasoning_effort"), spec.get("enable_thinking")),
                    spec.get("add_generation_prompt", True))
    sys.stdout.write(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
