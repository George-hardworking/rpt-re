"""Memory-aware worker sizing for scripts/py main programs.

Each pipeline passes its own reserve_gib / mem_per_worker_gib; this module only
reads /proc/meminfo and applies the shared formula.
"""

from __future__ import annotations

import os


def host_memory_snapshot() -> tuple[float, float, float]:
    """Return (MemAvailable GiB, MemTotal GiB, swap used %). Reads /proc/meminfo."""
    fields: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, raw, *_ = line.split()
            fields[key.rstrip(":")] = int(raw)
    avail_gib = fields["MemAvailable"] / (1024**2)
    total_gib = fields["MemTotal"] / (1024**2)
    swap_total = fields.get("SwapTotal", 0)
    swap_free = fields.get("SwapFree", 0)
    swap_pct = 0.0 if swap_total == 0 else 100.0 * (swap_total - swap_free) / swap_total
    return avail_gib, total_gib, swap_pct


def resolve_workers(
    n_tasks: int,
    *,
    reserve_gib: float,
    mem_per_worker_gib: float,
    override: int | None = None,
) -> tuple[int, dict]:
    """Choose process count: min(memory budget, cpu_count, n_tasks). CLI override wins."""
    n_tasks = max(0, int(n_tasks))
    if n_tasks <= 0:
        return 1, {"source": "empty_tasks", "resolved_workers": 1}

    if override is not None:
        workers = max(1, min(int(override), n_tasks))
        return workers, {
            "source": "cli_override",
            "resolved_workers": workers,
            "n_tasks": n_tasks,
        }

    avail_gib, total_gib, swap_pct = host_memory_snapshot()
    budget_gib = max(0.0, avail_gib - reserve_gib)
    mem_cap = max(1, int(budget_gib // mem_per_worker_gib)) if budget_gib > 0 else 1
    cpu_cap = max(1, os.cpu_count() or 1)
    workers = max(1, min(mem_cap, cpu_cap, n_tasks))
    if avail_gib < reserve_gib:
        workers = 1
    return workers, {
        "source": "memory_budget",
        "avail_gib": round(avail_gib, 1),
        "total_gib": round(total_gib, 1),
        "reserve_gib": round(reserve_gib, 1),
        "budget_gib": round(budget_gib, 1),
        "mem_per_worker_gib": round(mem_per_worker_gib, 1),
        "mem_cap": mem_cap,
        "cpu_cap": cpu_cap,
        "swap_percent": round(swap_pct, 1),
        "n_tasks": n_tasks,
        "resolved_workers": workers,
    }
