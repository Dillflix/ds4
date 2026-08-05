#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = ROOT / "speed-bench" / "summarize-sm75-memory-clock-sweep.py"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ds4-memory-clock-test-") as temp:
        output = Path(temp)
        run_lines = [
            "trial\tslot\tscenario\tdevice\trequested_sm_clock_mhz\t"
            "requested_memory_clock_mhz\tlog\ttelemetry\n"
        ]
        slot = 0
        for trial in (1, 2):
            for device in (1, 2):
                for memory, elapsed, sm_clock in (
                    (6500, 10.0 + device, 1000),
                    (5000, 9.0 + device, 1200),
                ):
                    slot += 1
                    log = output / f"run-{slot}.log"
                    log.write_text(
                        f"timed_per_call_ms={elapsed + trial * 0.01}\n"
                        "output_validation=exact-zero\nharness_status=ok\n",
                        encoding="utf-8",
                    )
                    telemetry = output / f"telemetry-{slot}.csv"
                    telemetry.write_text(
                        "timestamp,index,utilization.gpu [%],clocks.current.sm [MHz],"
                        "clocks.current.memory [MHz],power.draw [W],temperature.gpu,"
                        "clocks_event_reasons.sw_power_cap\n"
                        f"now,{device},100 %,{sm_clock} MHz,{memory} MHz,240 W,45,Active\n",
                        encoding="utf-8",
                    )
                    run_lines.append(
                        f"{trial}\t{slot}\tq4-late\t{device}\t1620\t{memory}\t"
                        f"{log}\t{telemetry}\n"
                    )
        (output / "runs.tsv").write_text("".join(run_lines), encoding="utf-8")
        subprocess.run([sys.executable, str(SUMMARIZER), str(output)], check=True)
        samples = read_rows(output / "samples.csv")
        assert len(samples) == 8
        assert {row["active_clock_events"] for row in samples} == {
            "clocks_event_reasons.sw_power_cap=1/1"
        }
        decisions = read_rows(output / "decisions.csv")
        assert len(decisions) == 2
        for row in decisions:
            assert row["best_memory_clock_mhz"] == "5000"
            assert math.isclose(float(row["best_sm_clock_mhz"]), 1200.0)
            assert float(row["speedup_over_max"]) > 1.0
        analysis = (output / "analysis.txt").read_text(encoding="utf-8")
        assert "Best measured settings" in analysis
    print("SM75 memory-clock summarizer synthetic test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
