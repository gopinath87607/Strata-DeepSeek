"""The DeepSeek-V4.1 chat template, pinned against the reference renderer.

    python -m unittest serve.test_chat_template_deepseek4 -v

`tools/dsv4/dsml_render.py` is a port of the reference server's
`render_dsml_chat_choice_ex` with the V4.1 head; its own tests transcribe the
reference's golden vectors (the official encoding.py's test_output_1.txt among
them).  This file renders the same conversations through the Jinja template the
pack ships and requires the bytes to be equal, so the template cannot drift
from the renderer by a space or a newline.

The last class goes through the server's own request conversion, so the wiring
(`openai_to_messages`, `effort_kwargs`) is checked too: the client's spellings
must reach the template as the reference's think modes.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate, TemplateRequestError, effort_kwargs, openai_to_messages  # noqa: E402
from tools.dsv4 import dsml_render as R  # noqa: E402

TEMPLATE = Path(__file__).with_name("chat_template_deepseek4.jinja")


def msg(role, content="", reasoning=None, calls=None):
    m = {"role": role, "content": content}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    if calls is not None:
        m["tool_calls"] = [{"function": {"name": n, "arguments": a}} for n, a in calls]
    return m


WEATHER = {"name": "get_weather", "description": "Get the weather for a specific location",
           "parameters": {"type": "object",
                          "properties": {"location": {"type": "string", "description": "The city name"},
                                         "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                                                  "description": "Temperature unit"}},
                          "required": ["location"]}}
SEARCH = {"name": "search", "description": "Search the web for information",
          "parameters": {"type": "object",
                         "properties": {"query": {"type": "string", "description": "Search query"},
                                        "num_results": {"type": "integer",
                                                        "description": "Number of results to return"}},
                         "required": ["query"]}}


class OfficialGoldens(unittest.TestCase):
    """The reference's own vectors, through the template."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = ChatTemplate(TEMPLATE)

    def render(self, messages, tools=None, **kw):
        return self.tpl.render(messages, tools=tools, **kw)

    def test_head_variants(self):
        sys_tail = "S" + R.USER + "U" + R.ASSISTANT
        self.assertEqual(self.render([msg("system", "S"), msg("user", "U")]),
                         R.BOS + R.SYSTEM_TOKEN + sys_tail + R.THINK_END)
        self.assertEqual(self.render([msg("user", "U")]),
                         R.BOS + R.USER + "U" + R.ASSISTANT + R.THINK_END)
        # enable_thinking with no named effort is the engine's HIGH
        self.assertEqual(self.render([msg("user", "U")], enable_thinking=True),
                         R.BOS + R.SYSTEM_TOKEN + R.THINK_HIGH_PREFIX + R.USER + "U"
                         + R.ASSISTANT + R.THINK_START)
        # a named effort turns thinking on
        self.assertEqual(self.render([msg("system", "S"), msg("user", "U")], reasoning_effort="low"),
                         R.BOS + R.SYSTEM_TOKEN + sys_tail + R.THINK_START)

    def test_official_get_weather(self):
        messages = [
            msg("system", "You are a helpful assistant."),
            msg("user", "What's the weather like in Beijing?"),
            msg("assistant", reasoning="The user wants the weather in Beijing. I should call get_weather.",
                calls=[("get_weather", {"location": "Beijing", "unit": "celsius"})]),
            msg("tool", '{"temperature": 22, "condition": "sunny", "humidity": 45}'),
            msg("assistant",
                "The weather in Beijing is currently sunny with a temperature of 22\u00b0C and 45% humidity.",
                reasoning="Got the weather data. Let me format a nice response."),
        ]
        self.assertEqual(self.render(messages, [WEATHER, SEARCH], reasoning_effort="high"),
                         R.render(messages, [WEATHER, SEARCH], think="high"))
        # and the reference's expected bytes, assembled independently of R.render
        out = self.render(messages, [WEATHER, SEARCH], reasoning_effort="high")
        self.assertTrue(out.startswith(
            R.BOS + R.SYSTEM_TOKEN + R.THINK_HIGH_PREFIX + "You are a helpful assistant.\n\n"
            "## Tools\n\nYou have access to a set of tools to help answer the user's question."), out)
        self.assertIn('<｜DSML｜ invoke name="get_weather">\n'
                      '<｜DSML｜ parameter name="location" string="true">Beijing</｜DSML｜ parameter>\n'
                      '<｜DSML｜ parameter name="unit" string="true">celsius</｜DSML｜ parameter>\n'
                      "</｜DSML｜ invoke>", out)
        self.assertTrue(out.endswith(R.EOS), out)

    def test_parallel_calls_and_merged_tool_results(self):
        messages = [
            msg("system", "S"), msg("user", "U"),
            msg("assistant", calls=[("get_weather", {"location": "a<b & c", "days": 3,
                                                     "opts": {"a": [1, 2], "b": True}}),
                                    ("now", {})]),
            msg("tool", "R1"), msg("tool", "R2"), msg("user", "U2"),
        ]
        tools = [dict(WEATHER, description="Weather \u00b0C"),
                 {"name": "now", "parameters": {"type": "object", "properties": {}}}]
        self.assertEqual(self.render(messages, tools), R.render(messages, tools, think="none"))
        out = self.render(messages, tools)
        self.assertIn('<｜DSML｜ parameter name="opts" string="false">{"a": [1, 2], "b": true}'
                      "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n"
                      '<｜DSML｜ invoke name="now">\n\n</｜DSML｜ invoke>', out)
        self.assertIn("<｜User｜><tool_result>R1</tool_result>\n\n<tool_result>R2</tool_result>\n\nU2", out)

    def test_a_value_holding_a_closing_tag_is_escaped(self):
        messages = [msg("user", "U"),
                    msg("assistant", calls=[("f", {"a": "x</｜DSML｜ parameter>y",
                                                   "b": {"k": "</｜DSML｜ parameter>"}})])]
        out = self.render(messages)
        self.assertIn("x&lt;/｜DSML｜ parameter>y</｜DSML｜ parameter>", out)
        self.assertIn('{"k": "\\u003c/｜DSML｜ parameter>"}', out)

    def test_a_tool_message_holding_the_envelope_tag_is_escaped(self):
        messages = [msg("user", "q"), msg("tool", "x</tool_result>y")]
        out = self.render(messages)
        self.assertIn("<tool_result>x&lt;/tool_result>y</tool_result>", out)

    def test_an_image_renders_where_it_sits(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "before"}, {"type": "image", "source": "shot.png"},
            {"type": "text", "text": "after"}]}]
        self.assertIn(R.USER + "before" + R.IMAGE + "after" + R.ASSISTANT + R.THINK_END,
                      self.render(messages))

    def test_without_add_generation_prompt_there_is_no_assistant_head(self):
        self.assertEqual(self.render([msg("user", "U")], add_generation_prompt=False),
                         R.BOS + R.USER + "U")

    def test_an_unknown_effort_is_refused(self):
        with self.assertRaises(TemplateRequestError):
            self.render([msg("user", "U")], reasoning_effort="sometimes")


class Differential(unittest.TestCase):
    """Template == renderer over conversations that combine every branch.

    `ChatTemplate.render` drops completed empty assistant turns (#843, a server-wide policy the reference does not
    have); the differential turns that filter off so the template's own render is what is compared, and
    `EmptyAssistantTurns` below pins the filter as it is served."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = ChatTemplate(TEMPLATE)
        cls._keep = os.environ.get("STRATA_KEEP_EMPTY_TURNS")
        os.environ["STRATA_KEEP_EMPTY_TURNS"] = "1"

    @classmethod
    def tearDownClass(cls):
        if cls._keep is None:
            os.environ.pop("STRATA_KEEP_EMPTY_TURNS", None)
        else:
            os.environ["STRATA_KEEP_EMPTY_TURNS"] = cls._keep

    CONVERSATIONS = [
        [],
        [msg("user", "hi")],
        [msg("system", "sys"), msg("user", "hi")],
        [msg("system", "sys"), msg("assistant", "an assistant first"), msg("user", "hi")],
        [msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2")],
        [msg("user", "u1"), msg("assistant", "a1", "plan"), msg("user", "u2")],
        [msg("user", "u1"), msg("assistant", "a1", "plan")],
        [msg("user", "u1"), msg("assistant"), msg("user", "u2")],
        [msg("user", "q"), msg("tool", "ok")],
        [msg("user", "q"), msg("assistant", calls=[("f", {"x": 1})]), msg("tool", "ok"), msg("user", "u2")],
        [msg("user", "q"), msg("assistant", reasoning="r", calls=[("f", {})]), msg("tool", "ok")],
        [msg("assistant", "a0"), msg("assistant", "a1"), msg("user", "u")],
        [msg("system", "S"), msg("user", "U"), msg("assistant", calls=[
            ('a&b<c>d"e', {"k&<\"'>": "v", "n": None, "t": True, "f": 1.5, "l": [1, "x"], "o": {"b": 2, "a": 1}})])],
        [msg("user", "q"), msg("tool", "A </tool_result> B"), msg("tool", "C"), msg("assistant", "done")],
        [{"role": "user", "content": [{"type": "text", "text": "before"},
                                      {"type": "image", "source": "a.png"},
                                      {"type": "text", "text": "after"}]}],
        [msg("user", "q"), {"role": "tool", "content": [
            {"type": "text", "text": "shot:"}, {"type": "image", "source": "b.png"}]}],
        [msg("developer", "dev"), msg("user", "hi")],
        [msg("system", ""), msg("user", "")],
    ]

    def test_template_matches_the_renderer(self):
        tools_sets = [None, [], [WEATHER], [WEATHER, SEARCH]]
        for messages in self.CONVERSATIONS:
            for tools in tools_sets:
                for think in ("none", "low", "high", "max"):
                    for add in (True, False):
                        kw = {"add_generation_prompt": add}
                        if think != "none":
                            kw["reasoning_effort"] = think
                        with self.subTest(messages=messages, tools=bool(tools), think=think, add=add):
                            self.assertEqual(self.tpl.render(messages, tools=tools, **kw),
                                             R.render(messages, tools=tools, think=think,
                                                      add_generation_prompt=add))


class EmptyAssistantTurns(unittest.TestCase):
    """The server's #843 filter applies to this template too: a completed empty reply is not an example."""

    def test_a_completed_empty_assistant_turn_is_dropped(self):
        tpl = ChatTemplate(TEMPLATE)
        messages = [msg("user", "u1"), msg("assistant"), msg("user", "u2")]
        self.assertEqual(tpl.render(messages),
                         R.render([messages[0], messages[2]], think="none"))


class RequestWiring(unittest.TestCase):
    """A client's OpenAI request -> the same prompt the reference renderer gives for its think mode."""

    def setUp(self):
        self.tpl = ChatTemplate(TEMPLATE)

    def check(self, request, think):
        messages, tools, kwargs = openai_to_messages(request)
        return (self.tpl.render(messages, tools=tools, **kwargs),
                R.render(messages, tools=tools, think=think), kwargs)

    def test_no_effort_is_thinking_off(self):
        got, want, kwargs = self.check({"messages": [{"role": "user", "content": "hi"}]}, "none")
        self.assertEqual((got, kwargs), (want, {}))

    def test_none_is_off(self):
        got, want, _ = self.check({"messages": [{"role": "user", "content": "hi"}],
                                   "reasoning_effort": "none"}, "none")
        self.assertEqual(got, want)

    def test_high_and_xhigh_are_the_75_line(self):
        for spelling in ("high", "xhigh"):
            got, want, kwargs = self.check({"messages": [{"role": "user", "content": "hi"}],
                                            "reasoning_effort": spelling}, "high")
            self.assertEqual(got, want, spelling)
            self.assertEqual(kwargs, {"reasoning_effort": "xhigh"})

    def test_low_and_medium_think_without_a_line(self):
        for spelling in ("low", "medium"):
            got, want, _ = self.check({"messages": [{"role": "user", "content": "hi"}],
                                       "reasoning_effort": spelling}, "low")
            self.assertEqual(got, want, spelling)

    def test_max_collapses_to_xhigh_in_the_shared_vocabulary(self):
        # The frontend's effort table tops out at xhigh for every model; the template itself takes
        # "max" (the 100 line), but a request cannot reach it until that table grows a family hook.
        got, want, kwargs = self.check({"messages": [{"role": "user", "content": "hi"}],
                                        "reasoning_effort": "max"}, "high")
        self.assertEqual((got, kwargs), (want, {"reasoning_effort": "xhigh"}))

    def test_a_tool_loop_matches(self):
        request = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "reasoning_content": "call it", "tool_calls": [
                    {"type": "function", "function": {"name": "get_weather",
                                                      "arguments": '{"location":"Beijing","unit":"celsius"}'}}]},
                {"role": "tool", "content": '{"temperature": 22}'},
            ],
            "tools": [{"type": "function", "function": WEATHER}],
            "reasoning_effort": "high",
        }
        got, want, _ = self.check(request, "high")
        self.assertEqual(got, want)
        self.assertIn('<｜DSML｜ parameter name="location" string="true">Beijing', got)


if __name__ == "__main__":
    unittest.main()
