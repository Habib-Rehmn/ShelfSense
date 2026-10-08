"""Replay calibrated frames into attention events, preserving frame records."""

import json


def derive_attention(source, output, engine):
    count = 0
    for line_number, line in enumerate(source, 1):
        try:
            result = engine.process(json.loads(line))
            output.write(json.dumps(result, allow_nan=False) + "\n")
            count += 1
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Attention input line {line_number}: {exc}") from exc
    if not count:
        raise ValueError("Attention input contains no frames")
    for summary in engine.finish():
        output.write(json.dumps(summary, allow_nan=False) + "\n")
    return count
