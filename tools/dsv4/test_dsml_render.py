"""The reference's V4.1 DSML render, byte for byte.

    python -m unittest tools.dsv4.test_dsml_render -v

The vectors are transcribed from the reference server's own golden tests
(`reference/ds4-dfm-rs/crates/ds4-server/tests/render.rs`, the tests
`dsml_v41_head_matches_official_encoding` and
`dsml_v41_spaced_tools_match_official_encoding`).  Those in turn reproduce the
official `encoding.py`, whose `test_output_1.txt` is byte-identical to case 1.
They pin what a plausible-looking renderer gets wrong: the system token is
written only for a non-empty head, tool blocks and tags carry a leading space,
string values are raw, non-string values are `json.dumps`-shaped, and adjacent
user-side blocks share one `<｜User｜>`.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.dsv4 import dsml_render as R  # noqa: E402


def msg(role, content="", reasoning=None, calls=None):
    m = {"role": role, "content": content}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    if calls is not None:
        m["tool_calls"] = [{"function": {"name": n, "arguments": a}} for n, a in calls]
    return m


class Head(unittest.TestCase):
    """dsml_v41_head_matches_official_encoding: the four official conversations, plus the user-only head."""

    def test_plain(self):
        self.assertEqual(R.render([msg("system", "S"), msg("user", "U")], think="none"),
                         R.BOS + R.SYSTEM_TOKEN + "S" + R.USER + "U" + R.ASSISTANT + R.THINK_END)

    def test_user_only_writes_no_system_token(self):
        self.assertEqual(R.render([msg("user", "U")], think="none"),
                         R.BOS + R.USER + "U" + R.ASSISTANT + R.THINK_END)

    def test_high_prefix(self):
        tail = R.USER + "U" + R.ASSISTANT
        self.assertEqual(R.render([msg("system", "S"), msg("user", "U")], think="high"),
                         R.BOS + R.SYSTEM_TOKEN + R.THINK_HIGH_PREFIX + "S" + tail + R.THINK_START)
        self.assertEqual(R.render([msg("user", "U")], think="high"),
                         R.BOS + R.SYSTEM_TOKEN + R.THINK_HIGH_PREFIX + tail + R.THINK_START)

    def test_max_is_its_own_line(self):
        out = R.render([msg("user", "U")], think="max")
        self.assertTrue(out.startswith(R.BOS + R.SYSTEM_TOKEN + R.THINK_MAX_PREFIX), out)

    def test_low_thinks_without_a_prefix(self):
        self.assertEqual(R.render([msg("user", "U")], think="low"),
                         R.BOS + R.USER + "U" + R.ASSISTANT + R.THINK_START)

    def test_add_generation_prompt_false_stops_before_the_assistant_head(self):
        self.assertEqual(R.render([msg("user", "U")], think="none", add_generation_prompt=False),
                         R.BOS + R.USER + "U")

    def test_an_unknown_think_mode_is_refused(self):
        with self.assertRaises(ValueError):
            R.render([msg("user", "U")], think="sometimes")


def spaced_tools_head():
    """"The V4.1 TOOLS_TEMPLATE verbatim, as the reference test assembles it."""
    return ("## Tools\n\n"
            "You have access to a set of tools to help answer the user's question. "
            'You can invoke tools by writing a "' + R.TOOL_CALLS_START + '" block like the following:\n\n'
            + R.TOOL_CALLS_START + "\n"
            + R.INVOKE_START + ' name="$TOOL_NAME">\n'
            + R.PARAM_START + ' name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE' + R.PARAM_END + "\n"
            "...\n"
            + R.INVOKE_END + "\n"
            + R.INVOKE_START + ' name="$TOOL_NAME2">\n...\n'
            + R.INVOKE_END + "\n"
            + R.TOOL_CALLS_END + "\n\n"
            'String parameters should be specified as is and set `string="true"`. '
            "For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set "
            '`string="false"`.\n\n'
            "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside "
            "<think>...</think> BEFORE any tool calls or final response.\n\n"
            "Otherwise, output directly after </think> with tool calls or final response.\n\n"
            "### Available Tool Schemas\n\n")


SPACED_TOOLS_TAIL = ("\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke "
                     "tool calls.\n")


def weather_schema(name="get_weather", description="Get the weather for a specific location",
                   properties=None, required=("location",)):
    return {"name": name, "description": description,
            "parameters": {"type": "object",
                           "properties": properties if properties is not None else {
                               "location": {"type": "string", "description": "The city name"},
                               "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                                        "description": "Temperature unit"}},
                           "required": list(required)}}


class OfficialSpacedTools(unittest.TestCase):
    """dsml_v41_spaced_tools_match_official_encoding, the two engine golden conversations."""

    def test_case1_get_weather(self):
        schemas = [
            weather_schema(),
            {"name": "search", "description": "Search the web for information",
             "parameters": {"type": "object",
                            "properties": {"query": {"type": "string", "description": "Search query"},
                                           "num_results": {"type": "integer",
                                                           "description": "Number of results to return"}},
                            "required": ["query"]}},
        ]
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
        want = (R.BOS + R.SYSTEM_TOKEN + R.THINK_HIGH_PREFIX + "You are a helpful assistant.\n\n"
                + spaced_tools_head()
                + '{"name": "get_weather", "description": "Get the weather for a specific location", '
                  '"parameters": {"type": "object", "properties": {"location": {"type": "string", '
                  '"description": "The city name"}, "unit": {"type": "string", "enum": ["celsius", '
                  '"fahrenheit"], "description": "Temperature unit"}}, "required": ["location"]}}\n'
                + '{"name": "search", "description": "Search the web for information", "parameters": '
                  '{"type": "object", "properties": {"query": {"type": "string", "description": '
                  '"Search query"}, "num_results": {"type": "integer", "description": "Number of results '
                  'to return"}}, "required": ["query"]}}'
                + SPACED_TOOLS_TAIL
                + R.USER + "What's the weather like in Beijing?"
                + R.ASSISTANT + R.THINK_START + "The user wants the weather in Beijing. I should call "
                  "get_weather." + R.THINK_END + "\n\n"
                + R.TOOL_CALLS_START + "\n"
                + R.INVOKE_START + ' name="get_weather">\n'
                + R.PARAM_START + ' name="location" string="true">Beijing' + R.PARAM_END + "\n"
                + R.PARAM_START + ' name="unit" string="true">celsius' + R.PARAM_END + "\n"
                + R.INVOKE_END + "\n" + R.TOOL_CALLS_END + R.EOS
                + R.USER + '<tool_result>{"temperature": 22, "condition": "sunny", "humidity": 45}'
                  "</tool_result>"
                + R.ASSISTANT + R.THINK_START + "Got the weather data. Let me format a nice response."
                + R.THINK_END
                + "The weather in Beijing is currently sunny with a temperature of 22\u00b0C and 45% humidity."
                + R.EOS)
        self.assertEqual(R.render(messages, schemas, think="high"), want)

    def test_case2_parallel_calls_merged_tool_results_and_escapes(self):
        schemas = [
            {"name": "get_weather", "description": "Weather \u00b0C",
             "parameters": {"type": "object",
                            "properties": {"location": {"type": "string"}, "days": {"type": "integer"}},
                            "required": ["location"]}},
            {"name": "now", "parameters": {"type": "object", "properties": {}}},
        ]
        messages = [
            msg("system", "S"),
            msg("user", "U"),
            msg("assistant", calls=[("get_weather", {"location": "a<b & c", "days": 3,
                                                     "opts": {"a": [1, 2], "b": True}}),
                                    ("now", {})]),
            msg("tool", "R1"),
            msg("tool", "R2"),
            msg("user", "U2"),
        ]
        want = (R.BOS + R.SYSTEM_TOKEN + "S\n\n"
                + spaced_tools_head()
                + '{"name": "get_weather", "description": "Weather \u00b0C", "parameters": '
                  '{"type": "object", "properties": {"location": {"type": "string"}, "days": '
                  '{"type": "integer"}}, "required": ["location"]}}\n'
                + '{"name": "now", "parameters": {"type": "object", "properties": {}}}'
                + SPACED_TOOLS_TAIL
                + R.USER + "U" + R.ASSISTANT + R.THINK_END + "\n\n"
                + R.TOOL_CALLS_START + "\n"
                + R.INVOKE_START + ' name="get_weather">\n'
                + R.PARAM_START + ' name="location" string="true">a<b & c' + R.PARAM_END + "\n"
                + R.PARAM_START + ' name="days" string="false">3' + R.PARAM_END + "\n"
                + R.PARAM_START + ' name="opts" string="false">{"a": [1, 2], "b": true}' + R.PARAM_END + "\n"
                + R.INVOKE_END + "\n"
                + R.INVOKE_START + ' name="now">\n\n' + R.INVOKE_END + "\n"
                + R.TOOL_CALLS_END + R.EOS
                + R.USER + "<tool_result>R1</tool_result>\n\n<tool_result>R2</tool_result>\n\nU2"
                + R.ASSISTANT + R.THINK_END)
        self.assertEqual(R.render(messages, schemas, think="none"), want)

    def test_a_tool_name_and_a_key_escape_every_attribute_character(self):
        messages = [msg("user", "U"),
                    msg("assistant", calls=[('a&b<c>d"e', {'k&<"\'>': "v"})])]
        out = R.render(messages, think="none")
        self.assertIn('<｜DSML｜ invoke name="a&amp;b&lt;c&gt;d&quot;e">', out)
        self.assertIn('name="k&amp;&lt;&quot;\'&gt;" string="true">v<', out)

    def test_a_closing_parameter_tag_inside_a_value_is_escaped(self):
        messages = [msg("user", "U"),
                    msg("assistant", calls=[("f", {"a": "x</｜DSML｜ parameter>y",
                                                   "b": {"k": "</｜DSML｜ parameter>"}})])]
        out = R.render(messages, think="none")
        self.assertIn('name="a" string="true">x&lt;/｜DSML｜ parameter>y<', out)
        self.assertIn('name="b" string="false">{"k": "\\u003c/｜DSML｜ parameter>"}<', out)


class ThinkMode(unittest.TestCase):
    def test_the_engine_default_is_thinking_off(self):
        self.assertEqual(R.think_mode(), "none")
        self.assertEqual(R.think_mode(None, None), "none")

    def test_enabling_without_an_effort_is_high(self):
        self.assertEqual(R.think_mode(None, True), "high")

    def test_flags_and_efforts(self):
        self.assertEqual(R.think_mode("low"), "low")
        self.assertEqual(R.think_mode("medium"), "low")
        self.assertEqual(R.think_mode("xhigh"), "high")
        self.assertEqual(R.think_mode("max"), "max")
        self.assertEqual(R.think_mode("none"), "none")
        self.assertEqual(R.think_mode("high", False), "none")
        with self.assertRaises(ValueError):
            R.think_mode("sometimes")


if __name__ == "__main__":
    unittest.main()
