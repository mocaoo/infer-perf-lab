"""Keep Prometheus labels and use exact metric names for counter sums."""

import math

import aiohttp


def parse_metrics(text):
    values = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        # Metric labels may contain spaces; a sample can also have a timestamp.
        quote = False
        escaped = False
        depth = 0
        end = None
        for index, char in enumerate(line):
            if escaped:
                escaped = False
                continue
            if char == "\\" and quote:
                escaped = True
                continue
            if char == '"':
                quote = not quote
            elif not quote:
                depth += (char == "{") - (char == "}")
                if char.isspace() and depth == 0:
                    end = index
                    break
        if end is None:
            continue
        try:
            value = float(line[end:].split()[0])
        except (ValueError, IndexError):
            continue
        if math.isfinite(value):
            values[line[:end]] = value
    return values


def counter(values, name):
    """Sum label series, excluding similarly named histograms and counters."""
    key = name if name.startswith("vllm:") else "vllm:" + name
    return sum(v for k, v in values.items() if k.split("{", 1)[0] == key)


async def metrics(session, base_url):
    async with session.get(
        base_url.rstrip("/") + "/metrics", timeout=aiohttp.ClientTimeout(total=5)
    ) as response:
        response.raise_for_status()
        return parse_metrics(await response.text())


def counter_delta(before, after):
    """Missing series and resets invalidate a measurement boundary."""
    tracked = ("spec_decode", "generation_tokens", "preemptions")
    keys = {k for k in before.keys() | after.keys() if any(t in k for t in tracked)}
    if keys - after.keys():
        raise RuntimeError("Counter series disappeared during measurement")
    result = {key: after[key] - before.get(key, 0) for key in keys}
    if any(value < 0 for value in result.values()):
        raise RuntimeError("Counter reset during measurement; discard this case")
    return result
