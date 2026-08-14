"""Query free NVIDIA GPUs and assign independent jobs onto remaining VRAM."""

from __future__ import annotations

import subprocess


def nvidia_smi_gpus() -> list[tuple[int, int, int]]:
    """Return (index, free_mib, util_pct) from nvidia-smi."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows: list[tuple[int, int, int]] = []
    for line in out.strip().splitlines():
        idx_s, free_s, util_s = [p.strip() for p in line.split(",")]
        rows.append((int(idx_s), int(free_s), int(util_s)))
    return rows


def select_gpu_free_mib(
    gpu_ids: list[int] | None,
    *,
    min_free_mib: int,
) -> list[tuple[int, int]]:
    """Return (gpu_index, free_mib) for GPUs that may host a train process."""
    found = nvidia_smi_gpus()
    by_id = {idx: (free, util) for idx, free, util in found}
    if gpu_ids is not None:
        selected: list[tuple[int, int]] = []
        for idx in gpu_ids:
            if idx not in by_id:
                raise ValueError(f"GPU {idx} not found; have {sorted(by_id)}")
            selected.append((idx, by_id[idx][0]))
        return selected
    return [(idx, free) for idx, free, _util in found if free >= min_free_mib]


def vram_mib_from_gib(vram_gib: float) -> int:
    return int(round(float(vram_gib) * 1024))


def gpu_concurrent_slot_budget(
    gpu_free: list[tuple[int, int]],
    *,
    min_vram_mib: int,
    jobs_per_gpu_max: int,
) -> tuple[int, list[int]]:
    """Sum of per-GPU slots: min(jobs_per_gpu_max, free // min_vram_mib)."""
    if min_vram_mib <= 0:
        raise ValueError(f"min_vram_mib must be positive, got {min_vram_mib}")
    per_gpu = [
        min(jobs_per_gpu_max, max(0, free // min_vram_mib))
        for _, free in gpu_free
    ]
    return sum(per_gpu), per_gpu
