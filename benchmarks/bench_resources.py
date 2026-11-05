"""Hardware facts and memory telemetry, using only tools that need no privileges. macOS-oriented.

What is measured: resident size (ps RSS) of each runner process, sampled every few seconds, so the maximum reported is
the maximum sampled, not an exact peak. RSS includes file-mapped weights and can share pages, so it is neither the size
of the weights file nor the memory the system would recover. Runners are listed separately and never added up.
Paths, user names and host names are never stored.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from pathlib import Path

from bench_core import append_jsonl


def _run(command: list, timeout: float = 10) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _sysctl(name: str) -> str:
    return _run(["sysctl", "-n", name]).strip()


def hardware_info() -> dict:
    gpus = re.findall(r"Chipset Model: (.+)", _run(["system_profiler", "SPDisplaysDataType"], timeout=30))
    memsize = _sysctl("hw.memsize")
    return {"cpu": _sysctl("machdep.cpu.brand_string") or None,
            "physical_cores": int(_sysctl("hw.physicalcpu") or 0) or None,
            "logical_cores": int(_sysctl("hw.logicalcpu") or 0) or None,
            "memory_gib": int(memsize) / 2**30 if memsize else None,
            "os": f"macOS {_run(['sw_vers', '-productVersion']).strip()} ({_run(['sw_vers', '-buildVersion']).strip()})",
            "gpus_reported_by_system": gpus}


def _cpu_seconds(text: str) -> float:
    """Parses the TIME column of ps: [[dd-]hh:]mm:ss.cc"""
    seconds = 0.0
    for part in text.strip().replace("-", ":").split(":"):
        seconds = seconds * 60 + float(part.replace(",", "."))
    return seconds


def serve_pid():
    pids = _run(["pgrep", "-f", "ollama serve"]).split()
    return int(pids[0]) if pids else None


def runners() -> list:
    """Child processes of the Ollama server: one per loaded model. Only a short hash of the weights file is kept."""
    parent = serve_pid()
    if parent is None:
        return []
    rows = []
    for line in _run(["ps", "-axo", "pid=,ppid=,rss=,time=,command="]).splitlines():
        parts = line.split(None, 4)
        if len(parts) == 5 and parts[1] == str(parent):
            blob = re.search(r"sha256-([0-9a-f]{12})", parts[4])
            rows.append({"pid": int(parts[0]), "rss_mib": round(int(parts[2]) / 1024, 1),
                         "cpu_s": _cpu_seconds(parts[3]), "blob": blob.group(1) if blob else None})
    return rows


def system_snapshot() -> dict:
    swap = re.search(r"used = ([\d.,]+)M", _sysctl("vm.swapusage"))
    free = re.search(r"free percentage: (\d+)%", _run(["memory_pressure", "-Q"]))
    speed = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", _run(["pmset", "-g", "therm"]))
    vm_stat = _run(["vm_stat"])
    swapouts = re.search(r"Swapouts:\s+(\d+)", vm_stat)
    level = _sysctl("kern.memorystatus_vm_pressure_level")
    return {"pressure_level": int(level) if level.isdigit() else None,
            "memory_free_pct": int(free.group(1)) if free else None,
            "swap_used_mib": float(swap.group(1).replace(",", ".")) if swap else None,
            "swapouts_pages": int(swapouts.group(1)) if swapouts else None,
            "cpu_speed_limit_pct": int(speed.group(1)) if speed else None}


def wait_inference_stopped(max_wait_s: float = 30, window_s: float = 2, max_cores: float = 0.5) -> dict:
    """After a cancelled request, waits until no runner uses more than max_cores of CPU over a window."""
    started = time.monotonic()
    while True:
        before = {r["pid"]: r["cpu_s"] for r in runners()}
        time.sleep(window_s)
        after = {r["pid"]: r["cpu_s"] for r in runners()}
        cores = max((after[pid] - before.get(pid, after[pid])) / window_s for pid in after) if after else 0.0
        waited = time.monotonic() - started
        if cores <= max_cores:
            return {"stopped": True, "waited_s": round(waited, 1), "cores_in_last_window": round(cores, 2)}
        if waited >= max_wait_s:
            return {"stopped": False, "waited_s": round(waited, 1), "cores_in_last_window": round(cores, 2)}


class ResourceSampler:
    """Background sampler. Each row carries the current block, so runs can be separated at analysis time."""

    def __init__(self, path: Path, interval_s: float):
        self.path, self.interval_s = Path(path), interval_s
        self.block = "idle"
        self._critical_since = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.interval_s + 15)

    def set_block(self, block: str):
        self.block = block
        self._critical_since = None

    def critical_seconds(self) -> float:
        """How long the system has been at critical memory pressure without a break during this block."""
        return time.monotonic() - self._critical_since if self._critical_since is not None else 0.0

    def _loop(self):
        while not self._stop.is_set():
            snapshot = system_snapshot()
            if snapshot["pressure_level"] == 4:
                self._critical_since = self._critical_since or time.monotonic()
            else:
                self._critical_since = None
            append_jsonl(self.path, {"t": round(time.time(), 1), "block": self.block, "runners": runners(), **snapshot})
            self._stop.wait(self.interval_s)
