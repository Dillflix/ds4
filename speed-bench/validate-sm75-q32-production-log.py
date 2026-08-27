#!/usr/bin/env python3
"""Validate the exact Q4-32/Q3A4 routed recipe in a production log."""

from __future__ import annotations

import re
import sys
from pathlib import Path


LINE = re.compile(
    r"^ds4: routed-quant-audit layer=(\d+) "
    r"gate=(\S+) up=(\S+) down=(\S+)$"
)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} LOG Q3A4_LAYER_CSV")
    log_path = Path(sys.argv[1])
    expected_q3a4 = {int(value) for value in sys.argv[2].split(",") if value}
    rows: dict[int, tuple[str, str, str]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE.match(line)
        if match:
            layer = int(match.group(1))
            if layer in rows:
                raise SystemExit(f"duplicate routed-quant audit for layer {layer}")
            rows[layer] = (match.group(2), match.group(3), match.group(4))
    expected_layers = set(range(43))
    if set(rows) != expected_layers:
        missing = sorted(expected_layers - set(rows))
        extra = sorted(set(rows) - expected_layers)
        raise SystemExit(
            f"routed-quant audit does not contain exactly layers 0..42; "
            f"missing={missing} extra={extra}"
        )
    observed_q3a4: set[int] = set()
    for layer, (gate, up, down) in rows.items():
        if down != "sm75_q4_32":
            raise SystemExit(f"layer {layer} down is {down}, not sm75_q4_32")
        if gate == up == "sm75_q3a4":
            observed_q3a4.add(layer)
        elif gate == up == "sm75_q4_32":
            pass
        else:
            raise SystemExit(
                f"layer {layer} has unsupported gate/up pair {gate}/{up}"
            )
    if observed_q3a4 != expected_q3a4:
        raise SystemExit(
            "Q3A4 layer set differs from the requested model recipe; "
            f"expected={sorted(expected_q3a4)} observed={sorted(observed_q3a4)}"
        )
    print(
        "validated routed recipe: "
        f"Q3A4 gate/up={len(observed_q3a4)}, "
        f"Q4-32 gate/up={43-len(observed_q3a4)}, Q4-32 down=43"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
