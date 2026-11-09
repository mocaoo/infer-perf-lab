"""Decode SSE events without depending on TCP packet boundaries."""

import json


async def events(content):
    data = []
    async for raw in content:
        # aiohttp StreamReader iteration yields complete lines even if packets split
        # UTF-8 code points. Other fields/comments do not carry completion data.
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                payload = "\n".join(data)
                data.clear()
                if payload == "[DONE]":
                    return
                obj = json.loads(payload)
                if not isinstance(obj, dict):
                    raise RuntimeError("SSE completion event must be an object")
                if obj.get("error"):
                    raise RuntimeError(str(obj["error"]))
                yield obj
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    # A normal vLLM stream terminates with [DONE], not a silent socket close.
    raise RuntimeError("Truncated SSE stream: missing [DONE]")
