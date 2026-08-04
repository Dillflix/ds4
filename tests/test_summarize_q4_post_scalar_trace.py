#!/usr/bin/env python3

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "speed-bench" / "summarize-q4-post-scalar-trace.py"


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    source = root / "kernels.csv"
    groups = root / "groups.csv"
    targets = root / "targets.csv"
    source.write_text(
        "Generating SQLite file /tmp/a.sqlite\n"
        "Processing [/tmp/a.sqlite]...\n"
        "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n"
        '50.0,500,344,1.0,1.0,1,2,0.1,"void moe_gate_up_mid_q4K_tile8_mma_kernel<(unsigned int)512, (bool)1>(float *)"\n'
        '25.0,250,344,1.0,1.0,1,2,0.1,"void moe_down_q4K_tile16_mma_sm75_kernel<(unsigned int)512, true>(float *)"\n'
        '15.0,150,56,1.0,1.0,1,2,0.1,"void matmul_q8_0_mma_sm75_exact_kernel<(unsigned int)256>(float *)"\n'
        "10.0,100,10,1.0,1.0,1,2,0.1,other_kernel\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), str(groups), str(targets)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "q4_gate_up=50.000000% instances=344" in result.stdout
    assert "q4_down=25.000000% instances=344" in result.stdout
    with groups.open(newline="", encoding="utf-8") as handle:
        group_rows = {row["group"]: row for row in csv.DictReader(handle)}
    assert group_rows["dense_q8_t256"]["total_time_ns"] == "150"
    with targets.open(newline="", encoding="utf-8") as handle:
        target_rows = list(csv.DictReader(handle))
    assert len(target_rows) == 2
    assert {row["scalar_specialization"] for row in target_rows} == {"true"}

print("q4 post-scalar trace summarizer: OK")
