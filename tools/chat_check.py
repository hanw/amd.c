#!/usr/bin/env python3
"""Check src/chat.c against the model's chat template, rendered as Hugging Face
transformers renders it (jinja2 sandbox, trim_blocks, lstrip_blocks, tojson
= json.dumps(ensure_ascii=False)).
usage: chat_check.py TEMPLATE.txt build/chat_render [OUT_DIR]
TEMPLATE.txt: the output of `ie-serve MODEL --show-template`. For each case
the script writes OUT_DIR/NAME.json (the request, as an OpenAI client sends
it) and OUT_DIR/NAME.txt (the reference prompt), runs chat_render on the
request and compares the texts. (pip install jinja2)"""
import copy
import json
import os
import subprocess
import sys

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "查询城市天气",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "城市名"},
        "days": {"type": "integer", "description": "天数"},
        "detail": {"type": "boolean"}},
        "required": ["city"]}}}
SEARCH = {"type": "function", "function": {
    "name": "web_search", "description": "Search the web.\nReturns \"titles\" and urls.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}, "filters": {"type": "object"}, "max": {"type": "number", "minimum": 0.5}}}}}


def call(name, args, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


CASES = [
    ("t1_tools_first_turn", {"messages": [{"role": "user", "content": "上海明天天气怎么样？"}], "tools": [WEATHER, SEARCH]}),
    ("t2_tools_system_low", {"messages": [{"role": "system", "content": "  你是助手。 "},
                                          {"role": "user", "content": "search for amd r9700"}],
                             "tools": [SEARCH], "reasoning_effort": "low"}),
    ("t3_call_and_result", {"messages": [
        {"role": "user", "content": "上海和北京明天天气？"},
        {"role": "assistant", "content": "", "reasoning_content": "要查两个城市。",
         "tool_calls": [call("get_weather", {"city": "上海", "days": 1}, "c0"),
                        call("get_weather", {"city": "北京", "days": 1, "detail": True}, "c1")]},
        {"role": "tool", "tool_call_id": "c0", "content": "晴，22 度"},
        {"role": "tool", "tool_call_id": "c1", "content": "多云，18 度"}], "tools": [WEATHER]}),
    ("t4_call_with_text_no_think", {"messages": [
        {"role": "user", "content": "find news"},
        {"role": "assistant", "content": "Let me search.",
         "tool_calls": [call("web_search", {"query": "AMD news", "filters": {"days": 7, "lang": ["en", "zh"]}, "max": 1.5}, "c0")]},
        {"role": "tool", "content": "[{\"title\": \"x\"}]"},
        {"role": "assistant", "content": "Here is the news."},
        {"role": "user", "content": "thanks"}],
        "tools": [SEARCH], "chat_template_kwargs": {"enable_thinking": False}}),
    ("t5_no_tools_multi", {"messages": [{"role": "user", "content": "hi　"},
                                        {"role": "assistant", "content": "hello", "reasoning_content": "greet"},
                                        {"role": "user", "content": "bye"}], "reasoning_effort": "medium"}),
]


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def raise_exception(m):
    raise Exception(m)


def main():
    tpl_path, render, out = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "build/chat_check")
    os.makedirs(out, exist_ok=True)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    tpl = env.from_string(open(tpl_path).read())
    fails = 0
    for name, req in CASES:
        kw = {}
        if "reasoning_effort" in req:
            kw["reasoning_effort"] = req["reasoning_effort"]
        kw.update(req.get("chat_template_kwargs", {}))
        ref = tpl.render(messages=req["messages"], tools=req.get("tools"), add_generation_prompt=True, **kw)
        # the request: OpenAI clients send the arguments as a JSON string
        body = copy.deepcopy(req)
        for m in body["messages"]:
            for c in m.get("tool_calls", []):
                c["function"]["arguments"] = json.dumps(c["function"]["arguments"], ensure_ascii=False)
        jp, tp = os.path.join(out, name + ".json"), os.path.join(out, name + ".txt")
        open(jp, "w").write(json.dumps(body, ensure_ascii=False))
        open(tp, "w").write(ref)
        got = subprocess.run([render, jp], capture_output=True).stdout.decode("utf-8")
        if got == ref:
            print("same  %s (%d chars)" % (name, len(ref)))
        else:
            fails += 1
            i = next((k for k in range(min(len(got), len(ref))) if got[k] != ref[k]), min(len(got), len(ref)))
            print("DIFF  %s at char %d\n  want %r\n  got  %r" % (name, i, ref[max(0, i - 40):i + 60], got[max(0, i - 40):i + 60]))
    print("%d cases, %d different" % (len(CASES), fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
