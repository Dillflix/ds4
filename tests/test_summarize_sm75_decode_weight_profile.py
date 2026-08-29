import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "speed-bench" / "summarize-sm75-decode-weight-profile.py"


def test_summarizes_required_decode_metrics(tmp_path: Path) -> None:
    ncu_dir = tmp_path / "ncu"
    out_dir = tmp_path / "out"
    ncu_dir.mkdir()
    units = {
        "ID": "",
        "Process ID": "",
        "Process Name": "",
        "Kernel Name": "",
        "gpu__time_duration.sum": "us",
        "dram__bytes.sum.per_second": "Gbyte/s",
        "dram__bytes.avg.pct_of_peak_sustained_elapsed": "%",
        "dram__bytes_read.sum": "Mbyte",
        "dram__bytes_write.sum": "Kbyte",
        "lts__t_sector_hit_rate.pct": "%",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed": "%",
        "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct": "%",
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio": "sector",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "inst",
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio": "inst",
        "sm__warps_active.avg.pct_of_peak_sustained_active": "%",
        "smsp__warps_eligible.avg.per_cycle_active": "warp",
        "launch__registers_per_thread": "register/thread",
        "launch__shared_mem_per_block": "byte/block",
        "launch__occupancy_limit_registers": "block",
        "launch__waves_per_multiprocessor": "",
    }
    row = {
        "ID": "1",
        "Process ID": "123",
        "Process Name": "cuda_sm75_decode_weight_profile",
        "Kernel Name": "matmul_q8_0_preq_warp8_kernel",
        "gpu__time_duration.sum": "125",
        "dram__bytes.sum.per_second": "400",
        "dram__bytes.avg.pct_of_peak_sustained_elapsed": "78.5",
        "dram__bytes_read.sum": "35.651584",
        "dram__bytes_write.sum": "131.072",
        "lts__t_sector_hit_rate.pct": "12.5",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed": "41.0",
        "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct": "96.0",
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio": "7.5",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "3.5",
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio": "0.5",
        "sm__warps_active.avg.pct_of_peak_sustained_active": "37.5",
        "smsp__warps_eligible.avg.per_cycle_active": "1.25",
        "launch__registers_per_thread": "64",
        "launch__shared_mem_per_block": "0",
        "launch__occupancy_limit_registers": "4",
        "launch__waves_per_multiprocessor": "14.22",
    }
    path = ncu_dir / "q8-single-t32.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, row.keys())
        writer.writeheader()
        writer.writerow(units)
        writer.writerow(row)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(ncu_dir),
            str(out_dir),
            "--scenarios",
            "q8-single-t32",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "DRAM-saturated" in result.stdout
    with (out_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = next(csv.DictReader(handle))
    assert summary["dram_gb_s"] == "400.0"
    assert summary["dram_peak_pct"] == "78.5"
    assert summary["global_load_efficiency_pct"] == "96.0"
    assert summary["expected_weight_stream_mib"] == "34.0"
    assert summary["duration_us"] == "125.0"
    assert summary["dram_read_mib"] == "34.0"
    assert summary["dram_write_mib"] == "0.125"
