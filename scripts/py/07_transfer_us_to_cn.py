"""CLI: US-trained CNN -> China direct inference or head-only finetune + backtest."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from backtest.engine import H1_METRICS
from config import (
    BACKTEST_CNN_ROOT,
    BATCH_SIZE,
    GPU_MIN_FREE_GIB,
    HORIZON_BACKTEST_DIR,
    MARKET_CN,
    N_ENSEMBLE,
    TRAIN_JOB_RAM_GIB,
    TRAIN_JOBS_PER_GPU_MAX,
    TRAIN_VAL_SPLIT_SEED,
    TRANSFER_COMPARE_ROOT,
    TRANSFER_MODE_DIRECT,
    TRANSFER_MODE_FINETUNE,
    VRAM_PER_JOB_GIB,
    WINDOW_DAYS,
    market_processed_dir,
    market_sample_config,
    sample_freq_for_horizon,
    transfer_backtest_output_root,
    transfer_models_root,
    us_seed_checkpoint,
)
from models.dataset import (
    ImageLabelDataset,
    collect_samples,
    model_run_dir,
    model_run_tag,
    split_samples,
)
from models.train_loop import (
    finetune_head_model,
    load_full_checkpoint,
    predict_dataset,
    training_is_finished,
)
from utils.gpu import (
    gpu_concurrent_slot_budget,
    nvidia_smi_gpus,
    select_gpu_free_mib,
    vram_mib_from_gib,
)
from utils.workers import resolve_workers

DEFAULT_RESERVE_GIB = 16.0
CHILD_ENV = "RPT_TRANSFER_CHILD"
COMPARE_SOURCES = (
    ("cn_local", BACKTEST_CNN_ROOT),
    ("us_direct", transfer_backtest_output_root(TRANSFER_MODE_DIRECT)),
    ("us_finetune", transfer_backtest_output_root(TRANSFER_MODE_FINETUNE)),
)
_T_RE = re.compile(r"^([-+]?\d*\.?\d+)(\**)$")


def log(msg: str) -> None:
    print(msg, flush=True)


def all_configs() -> list[tuple[int, int]]:
    return [(d, h) for d in WINDOW_DAYS for h in WINDOW_DAYS]


def run_tag_dir(models_root: Path, image_days: int, horizon: int) -> Path:
    return models_root / model_run_tag(image_days, horizon)


def seed_pred_path(
    models_root: Path,
    image_days: int,
    horizon: int,
    seed: int,
) -> Path:
    return run_tag_dir(models_root, image_days, horizon) / f"seed{seed}_pred.feather"


def ensemble_pred_path(models_root: Path, image_days: int, horizon: int) -> Path:
    return run_tag_dir(models_root, image_days, horizon) / "ensemble_pred.feather"


def merge_ensemble(
    models_root: Path,
    image_days: int,
    horizon: int,
    seeds: list[int],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        path = seed_pred_path(models_root, image_days, horizon, seed)
        if not path.is_file():
            raise FileNotFoundError(f"missing seed prediction: {path}")
        frame = pd.read_feather(path)[["PERMNO", "Date", "p_up"]].rename(
            columns={"p_up": f"p_seed{seed}"}
        )
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["PERMNO", "Date"])
    p_cols = [f"p_seed{seed}" for seed in seeds]
    merged["p_up"] = merged[p_cols].mean(axis=1)
    return merged[["PERMNO", "Date", "p_up"]]


def prepare_transfer_datasets(
    images_root: Path,
    image_days: int,
    horizon: int,
    *,
    split_seed: int,
    pixel_mean: float,
    pixel_std: float,
    year_limit: int | None = None,
    max_samples: int | None = None,
) -> tuple[ImageLabelDataset, ImageLabelDataset, ImageLabelDataset, int]:
    sample_freq = sample_freq_for_horizon(horizon)
    samples = collect_samples(
        images_root,
        image_days,
        horizon,
        sample_freq=sample_freq,
        year_limit=year_limit,
        max_samples=max_samples,
    )
    cfg = market_sample_config(MARKET_CN)
    train_samples, val_samples, test_samples = split_samples(
        samples,
        seed=split_seed,
        train_end=cfg.train_end,
        test_start=cfg.test_start,
        sample_end=cfg.sample_end,
    )
    log(
        f"I{image_days}/R{horizon} freq={sample_freq} split_seed={split_seed} "
        f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)} "
        f"pixel_mean={pixel_mean:.6f} pixel_std={pixel_std:.6f}"
    )
    mmap_cache: dict = {}
    train_ds = ImageLabelDataset(
        images_root,
        image_days,
        train_samples,
        pixel_mean,
        pixel_std,
        sample_freq=sample_freq,
        mmap_cache=mmap_cache,
    )
    val_ds = ImageLabelDataset(
        images_root,
        image_days,
        val_samples,
        pixel_mean,
        pixel_std,
        sample_freq=sample_freq,
        mmap_cache=mmap_cache,
    )
    test_ds = ImageLabelDataset(
        images_root,
        image_days,
        test_samples,
        pixel_mean,
        pixel_std,
        sample_freq=sample_freq,
        mmap_cache=mmap_cache,
    )
    return train_ds, val_ds, test_ds, len(test_samples)


def us_checkpoint(image_days: int, horizon: int, seed: int) -> Path:
    path = us_seed_checkpoint(image_days, horizon, seed)
    if not path.is_file():
        raise FileNotFoundError(f"missing US checkpoint: {path}")
    return path


def direct_one_seed(
    *,
    test_ds: ImageLabelDataset,
    models_root: Path,
    image_days: int,
    horizon: int,
    seed: int,
    us_ckpt: Path,
    device: str | None,
    batch_size: int,
    fresh: bool,
) -> None:
    pred_path = seed_pred_path(models_root, image_days, horizon, seed)
    if fresh and pred_path.is_file():
        pred_path.unlink()

    tag = model_run_tag(image_days, horizon)
    if pred_path.is_file():
        log(f"{tag} seed{seed} skip predict (exists): {pred_path}")
        return

    from models.train_loop import _device

    dev = _device(device)
    model, src_cfg = load_full_checkpoint(us_ckpt, dev)
    if src_cfg.image_days != image_days or src_cfg.horizon != horizon:
        raise ValueError(
            f"US checkpoint I{src_cfg.image_days}/R{src_cfg.horizon} "
            f"!= I{image_days}/R{horizon}: {us_ckpt}"
        )
    pred_df = predict_dataset(model, test_ds, device=device, batch_size=batch_size)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_feather(pred_path)
    log(f"{tag} seed{seed} direct predict n={len(pred_df)} -> {pred_path}")


def finetune_one_seed(
    *,
    train_ds: ImageLabelDataset,
    val_ds: ImageLabelDataset,
    test_ds: ImageLabelDataset,
    n_test: int,
    models_root: Path,
    image_days: int,
    horizon: int,
    seed: int,
    us_ckpt: Path,
    device: str | None,
    batch_size: int,
    fresh: bool,
) -> None:
    ckpt_dir = model_run_dir(models_root, image_days, horizon, seed)
    pred_path = seed_pred_path(models_root, image_days, horizon, seed)
    if fresh and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    if fresh and pred_path.is_file():
        pred_path.unlink()

    tag = model_run_tag(image_days, horizon)
    log(f"{tag} finetune seed={seed}")

    model = finetune_head_model(
        train_ds,
        val_ds,
        image_days=image_days,
        horizon=horizon,
        seed=seed,
        pixel_mean=train_ds.pixel_mean,
        pixel_std=train_ds.pixel_std,
        n_test=n_test,
        ckpt_dir=ckpt_dir,
        init_checkpoint=us_ckpt,
        device=device,
        batch_size=batch_size,
        log_fn=log,
    )
    pred_df = predict_dataset(model, test_ds, device=device, batch_size=batch_size)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_feather(pred_path)
    log(f"{tag} seed{seed} finetune predict n={len(pred_df)} -> {pred_path}")


def direct_seed_done(
    models_root: Path,
    image_days: int,
    horizon: int,
    seed: int,
) -> bool:
    return seed_pred_path(models_root, image_days, horizon, seed).is_file()


def finetune_seed_done(
    models_root: Path,
    image_days: int,
    horizon: int,
    seed: int,
) -> bool:
    ckpt_dir = model_run_dir(models_root, image_days, horizon, seed)
    if not training_is_finished(ckpt_dir):
        return False
    return seed_pred_path(models_root, image_days, horizon, seed).is_file()


def resolve_configs(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.image_days is not None and args.horizon is not None:
        return [(args.image_days, args.horizon)]
    if args.all_configs:
        return all_configs()
    raise ValueError("require --all-configs or both --image-days and --horizon")


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.all_seeds:
        return list(range(N_ENSEMBLE))
    return [args.seed]


def parse_gpu_ids(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    ids = [int(p.strip()) for p in raw.split(",") if p.strip() != ""]
    if not ids:
        raise ValueError("--gpu-ids is empty")
    return ids


def vram_need_mib(image_days: int, override_gib: float | None, batch_size: int) -> int:
    gib = override_gib if override_gib is not None else VRAM_PER_JOB_GIB[image_days]
    scale = batch_size / BATCH_SIZE
    return vram_mib_from_gib(gib * scale)


def build_child_cmd(
    args: argparse.Namespace,
    command: str,
    image_days: int,
    horizon: int,
    seed: int,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        command,
        "--image-days",
        str(image_days),
        "--horizon",
        str(horizon),
        "--seed",
        str(seed),
        "--split-seed",
        str(args.split_seed),
        "--device",
        "cuda",
        "--batch-size",
        str(args.batch_size),
    ]
    if args.images is not None:
        cmd.extend(["--images", str(args.images)])
    if args.us_models is not None:
        cmd.extend(["--us-models", str(args.us_models)])
    if args.year_limit is not None:
        cmd.extend(["--year-limit", str(args.year_limit)])
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    cmd.append("--no-all-seeds")
    return cmd


def _pump_stdout(proc: subprocess.Popen, prefix: str) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        log(f"{prefix}{line.rstrip()}")


def run_child_job(
    args: argparse.Namespace,
    command: str,
    image_days: int,
    horizon: int,
    seed: int,
    gpu_id: int,
) -> None:
    cmd = build_child_cmd(args, command, image_days, horizon, seed)
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env[CHILD_ENV] = "1"
    tag = model_run_tag(image_days, horizon)
    prefix = f"[{command} {tag} seed{seed} gpu{gpu_id}] "
    log(f"{prefix}start")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(ROOT),
    )
    pump = threading.Thread(target=_pump_stdout, args=(proc, prefix), daemon=True)
    pump.start()
    rc = proc.wait()
    pump.join()
    if rc != 0:
        raise RuntimeError(f"{prefix}exited {rc}")
    log(f"{prefix}done")


def launch_parallel_jobs(
    args: argparse.Namespace,
    command: str,
    jobs: list[tuple[int, int, int]],
    gpu_free: list[tuple[int, int]],
) -> None:
    remaining = {idx: free for idx, free in gpu_free}
    if not remaining:
        raise RuntimeError("no GPU with enough free memory")
    for image_days, _horizon, _seed in jobs:
        need = vram_need_mib(image_days, args.vram_per_job_gb, args.batch_size)
        if not any(free >= need for free in remaining.values()):
            raise RuntimeError(
                f"I{image_days} needs {need} MiB VRAM but max free is {max(remaining.values())} MiB"
            )
    n_conc, diag = resolve_workers(
        len(jobs),
        reserve_gib=args.reserve_gib,
        mem_per_worker_gib=TRAIN_JOB_RAM_GIB,
        override=args.workers,
    )
    min_need = min(
        vram_need_mib(d, args.vram_per_job_gb, args.batch_size) for d, _h, _s in jobs
    )
    max_gpu_slots, _per_gpu_slots = gpu_concurrent_slot_budget(
        gpu_free,
        min_vram_mib=min_need,
        jobs_per_gpu_max=args.jobs_per_gpu_max,
    )
    n_conc = min(n_conc, max(1, max_gpu_slots), len(jobs))
    log(f"parallel jobs={len(jobs)} n_conc={n_conc} ram_diag={diag}")

    lock = threading.Lock()
    gpu_cv = threading.Condition(lock)
    active_on_gpu = {idx: 0 for idx in remaining}

    def acquire_gpu(image_days: int) -> int:
        need = vram_need_mib(image_days, args.vram_per_job_gb, args.batch_size)
        with gpu_cv:
            while True:
                candidates = [
                    idx
                    for idx, free in remaining.items()
                    if free >= need and active_on_gpu[idx] < args.jobs_per_gpu_max
                ]
                if candidates:
                    gpu_id = max(candidates, key=lambda idx: remaining[idx])
                    remaining[gpu_id] -= need
                    active_on_gpu[gpu_id] += 1
                    return gpu_id
                gpu_cv.wait()

    def release_gpu(gpu_id: int, image_days: int) -> None:
        need = vram_need_mib(image_days, args.vram_per_job_gb, args.batch_size)
        with gpu_cv:
            remaining[gpu_id] += need
            active_on_gpu[gpu_id] -= 1
            gpu_cv.notify_all()

    def run_job(job: tuple[int, int, int]) -> None:
        image_days, horizon, seed = job
        gpu_id = acquire_gpu(image_days)
        try:
            run_child_job(args, command, image_days, horizon, seed, gpu_id)
        finally:
            release_gpu(gpu_id, image_days)

    with ThreadPoolExecutor(max_workers=n_conc) as pool:
        futs = [pool.submit(run_job, job) for job in jobs]
        for fut in as_completed(futs):
            fut.result()


def merge_ensembles_for_configs(
    models_root: Path,
    configs: list[tuple[int, int]],
    seeds: list[int],
) -> None:
    for image_days, horizon in configs:
        out_path = ensemble_pred_path(models_root, image_days, horizon)
        if all(seed_pred_path(models_root, image_days, horizon, s).is_file() for s in seeds):
            ensemble = merge_ensemble(models_root, image_days, horizon, seeds)
            ensemble.to_feather(out_path)
            log(f"ensemble n={len(ensemble)} -> {out_path}")


def run_direct_or_finetune(args: argparse.Namespace, command: str) -> None:
    configs = resolve_configs(args)
    seeds = resolve_seeds(args)
    mode = TRANSFER_MODE_DIRECT if command == "direct" else TRANSFER_MODE_FINETUNE
    models_root = transfer_models_root(mode)

    cn_root = market_processed_dir(MARKET_CN)
    images_root = args.images if args.images is not None else cn_root / "images"

    if args.fresh:
        for image_days, horizon in configs:
            target = run_tag_dir(models_root, image_days, horizon)
            if target.exists():
                shutil.rmtree(target)

    is_child = os.environ.get(CHILD_ENV) == "1"
    child_seeds = [args.seed] if is_child else seeds

    if is_child:
        image_days, horizon = configs[0]
        seed = args.seed
        us_ckpt = us_checkpoint(image_days, horizon, seed)
        _, src_cfg = load_full_checkpoint(us_ckpt, device="cpu")
        train_ds, val_ds, test_ds, n_test = prepare_transfer_datasets(
            images_root,
            image_days,
            horizon,
            split_seed=args.split_seed,
            pixel_mean=src_cfg.pixel_mean,
            pixel_std=src_cfg.pixel_std,
            year_limit=args.year_limit,
            max_samples=args.max_samples,
        )
        if command == "direct":
            direct_one_seed(
                test_ds=test_ds,
                models_root=models_root,
                image_days=image_days,
                horizon=horizon,
                seed=seed,
                us_ckpt=us_ckpt,
                device=args.device,
                batch_size=args.batch_size,
                fresh=False,
            )
        else:
            finetune_one_seed(
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                n_test=n_test,
                models_root=models_root,
                image_days=image_days,
                horizon=horizon,
                seed=seed,
                us_ckpt=us_ckpt,
                device=args.device,
                batch_size=args.batch_size,
                fresh=False,
            )
        return

    done_fn = direct_seed_done if command == "direct" else finetune_seed_done
    jobs = [
        (image_days, horizon, seed)
        for image_days, horizon in configs
        for seed in seeds
        if not done_fn(models_root, image_days, horizon, seed)
    ]
    log(f"{command} pending jobs={len(jobs)} configs={configs} seeds={seeds}")

    use_cpu = args.device == "cpu"
    can_parallel = (not is_child) and (not use_cpu) and len(jobs) > 1
    if can_parallel:
        gpu_ids = parse_gpu_ids(args.gpu_ids)
        min_free_mib = vram_mib_from_gib(args.gpu_min_free_gb)
        gpu_free = select_gpu_free_mib(gpu_ids, min_free_mib=min_free_mib)
        if not gpu_free:
            raise RuntimeError(
                f"no GPU with free memory >= {args.gpu_min_free_gb} GiB; "
                f"nvidia-smi={nvidia_smi_gpus()}"
            )
        log(f"nvidia-smi={nvidia_smi_gpus()}")
        if jobs:
            launch_parallel_jobs(args, command, jobs, gpu_free)
        merge_ensembles_for_configs(models_root, configs, seeds)
        return

    for image_days, horizon in configs:
        for seed in seeds:
            if done_fn(models_root, image_days, horizon, seed):
                continue
            us_ckpt = us_checkpoint(image_days, horizon, seed)
            _, src_cfg = load_full_checkpoint(us_ckpt, device="cpu")
            train_ds, val_ds, test_ds, n_test = prepare_transfer_datasets(
                images_root,
                image_days,
                horizon,
                split_seed=args.split_seed,
                pixel_mean=src_cfg.pixel_mean,
                pixel_std=src_cfg.pixel_std,
                year_limit=args.year_limit,
                max_samples=args.max_samples,
            )
            if command == "direct":
                direct_one_seed(
                    test_ds=test_ds,
                    models_root=models_root,
                    image_days=image_days,
                    horizon=horizon,
                    seed=seed,
                    us_ckpt=us_ckpt,
                    device=args.device,
                    batch_size=args.batch_size,
                    fresh=False,
                )
            else:
                finetune_one_seed(
                    train_ds=train_ds,
                    val_ds=val_ds,
                    test_ds=test_ds,
                    n_test=n_test,
                    models_root=models_root,
                    image_days=image_days,
                    horizon=horizon,
                    seed=seed,
                    us_ckpt=us_ckpt,
                    device=args.device,
                    batch_size=args.batch_size,
                    fresh=False,
                )
    merge_ensembles_for_configs(models_root, configs, seeds)


def run_backtest_mode(args: argparse.Namespace, mode: str) -> None:
    models_root = transfer_models_root(mode)
    output_root = transfer_backtest_output_root(mode)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "py" / "04_backtest.py"),
        "--market",
        MARKET_CN,
        "--models",
        str(models_root),
        "--output-root",
        str(output_root),
        "--eval-horizons",
        "1",
    ]
    if args.images is not None:
        cmd.extend(["--images", str(args.images)])
    if args.image_days is not None and args.horizon is not None:
        cmd.extend(["--image-days", str(args.image_days), "--horizon", str(args.horizon)])
    else:
        cmd.append("--all-configs")
    if args.fresh:
        cmd.append("--fresh")
    log(f"backtest {mode}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def read_h1_models(path: Path, sheet: str) -> dict[str, pd.Series]:
    raw = pd.read_excel(path, sheet_name=sheet, header=[0, 1])
    name_col = ("metric", "sig_rank")
    models: dict[str, pd.Series] = {}
    for _, row in raw.iterrows():
        model = row[name_col]
        if pd.isna(model) or str(model) in {"sig_rank", "nan"}:
            continue
        models[str(model)] = row
    if not models:
        raise ValueError(f"no model rows in {path} sheet={sheet}")
    return models


def parse_t_stat(cell: object) -> float:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return float("nan")
    text = str(cell).strip()
    if not text:
        return float("nan")
    m = _T_RE.match(text)
    if not m:
        return float("nan")
    return float(m.group(1))


def extract_hl_row(path: Path, sheet: str, tag: str) -> dict[str, object]:
    models = read_h1_models(path, sheet)
    if tag not in models:
        raise KeyError(f"{path} sheet={sheet} missing row {tag!r}")
    row = models[tag]
    out: dict[str, object] = {"config": tag}
    for metric in H1_METRICS:
        val = row.get((metric, "DH"))
        if val is None or (isinstance(val, float) and pd.isna(val)):
            out[metric] = float("nan")
        else:
            out[metric] = float(val)
    if "t" in models:
        t_row = models["t"]
        out["t_stat"] = parse_t_stat(t_row.get(("Annualized Return", "DH")))
    return out


def config_xlsx_path(output_root: Path, horizon: int, tag: str, eval_horizon: int) -> Path:
    freq_dir = HORIZON_BACKTEST_DIR[horizon]
    return output_root / MARKET_CN / freq_dir / f"{tag}_h{eval_horizon}.xlsx"


def run_compare(args: argparse.Namespace) -> list[Path]:
    configs = resolve_configs(args)
    eval_h = 1
    sheets = ("equal", "float", "total")
    written: list[Path] = []

    for horizon in sorted({h for _i, h in configs}):
        freq_dir = HORIZON_BACKTEST_DIR[horizon]
        out_path = TRANSFER_COMPARE_ROOT / MARKET_CN / freq_dir / "local_vs_direct_vs_finetune_h1.xlsx"
        if out_path.is_file() and not args.fresh:
            log(f"skip compare (exists): {out_path}")
            written.append(out_path)
            continue
        if args.fresh and out_path.is_file():
            out_path.unlink()

        h_configs = [(i, h) for i, h in configs if h == horizon]
        tables: dict[str, pd.DataFrame] = {}
        for sheet in sheets:
            rows: list[dict[str, object]] = []
            for image_days, h in h_configs:
                tag = model_run_tag(image_days, h)
                for source_name, root in COMPARE_SOURCES:
                    xlsx = config_xlsx_path(root, h, tag, eval_h)
                    if not xlsx.is_file():
                        raise FileNotFoundError(f"missing backtest for compare: {xlsx}")
                    hl = extract_hl_row(xlsx, sheet, tag)
                    rows.append({"source": source_name, **hl})
            tables[sheet] = pd.DataFrame(rows)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp.xlsx")
        with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
            for sheet, frame in tables.items():
                frame.to_excel(writer, sheet_name=sheet, index=False)
        tmp.replace(out_path)
        log(f"wrote compare {out_path}")
        written.append(out_path)
    return written


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--images", type=Path, default=None, help="processed/cn/images root")
    parser.add_argument(
        "--us-models",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--image-days", type=int, default=None, choices=WINDOW_DAYS)
    parser.add_argument("--horizon", type=int, default=None, choices=WINDOW_DAYS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=TRAIN_VAL_SPLIT_SEED,
        help="fixed 70/30 train/val split seed (shared across ensemble)",
    )
    parser.add_argument(
        "--all-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"seeds 0..{N_ENSEMBLE - 1} (default: on)",
    )
    parser.add_argument(
        "--all-configs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="all 9 I×R configs (default: on)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu-ids", type=str, default=None)
    parser.add_argument("--gpu-min-free-gb", type=float, default=GPU_MIN_FREE_GIB)
    parser.add_argument("--vram-per-job-gb", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--jobs-per-gpu-max", type=int, default=TRAIN_JOBS_PER_GPU_MAX)
    parser.add_argument("--reserve-gib", type=float, default=DEFAULT_RESERVE_GIB)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--year-limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="US CNN weights -> China direct inference / head finetune / backtest / compare"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name in (TRANSFER_MODE_DIRECT, TRANSFER_MODE_FINETUNE):
        p = sub.add_parser(name, help=f"{name} transfer on China images")
        add_common_args(p)

    p_bt = sub.add_parser("backtest", help="run step-04 backtest for direct and/or finetune")
    add_common_args(p_bt)
    p_bt.add_argument(
        "--mode",
        choices=(TRANSFER_MODE_DIRECT, TRANSFER_MODE_FINETUNE, "both"),
        default="both",
    )

    p_cmp = sub.add_parser("compare", help="H-L compare cn_local vs us_direct vs us_finetune")
    add_common_args(p_cmp)

    p_all = sub.add_parser("all", help="direct -> finetune -> backtest -> compare")
    add_common_args(p_all)

    args = parser.parse_args()
    if args.jobs_per_gpu_max <= 0:
        raise ValueError("--jobs-per-gpu-max must be positive")

    if args.command == TRANSFER_MODE_DIRECT:
        run_direct_or_finetune(args, TRANSFER_MODE_DIRECT)
    elif args.command == TRANSFER_MODE_FINETUNE:
        run_direct_or_finetune(args, TRANSFER_MODE_FINETUNE)
    elif args.command == "backtest":
        mode = args.mode
        if mode in (TRANSFER_MODE_DIRECT, "both"):
            run_backtest_mode(args, TRANSFER_MODE_DIRECT)
        if mode in (TRANSFER_MODE_FINETUNE, "both"):
            run_backtest_mode(args, TRANSFER_MODE_FINETUNE)
    elif args.command == "compare":
        run_compare(args)
    elif args.command == "all":
        run_direct_or_finetune(args, TRANSFER_MODE_DIRECT)
        run_direct_or_finetune(args, TRANSFER_MODE_FINETUNE)
        run_backtest_mode(args, TRANSFER_MODE_DIRECT)
        run_backtest_mode(args, TRANSFER_MODE_FINETUNE)
        run_compare(args)
    else:
        raise ValueError(f"unknown command={args.command}")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    main()
