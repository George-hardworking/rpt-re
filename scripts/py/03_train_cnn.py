"""CLI: train panel CNN on memmap images and write OOS P(up) predictions."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import (
    BATCH_SIZE,
    GPU_MIN_FREE_GIB,
    MODELS_ROOT,
    N_ENSEMBLE,
    TRAIN_JOB_RAM_GIB,
    TRAIN_JOBS_PER_GPU,
    TRAIN_VAL_SPLIT_SEED,
    VRAM_PER_JOB_GIB,
    WINDOW_DAYS,
)
from models.cnn import cnn_num_blocks
from models.dataset import (
    ImageLabelDataset,
    collect_samples,
    load_or_compute_pixel_stats,
    model_run_dir,
    model_run_tag,
    pixel_stats_cache_path,
    split_samples,
)
from models.train_loop import (
    BEST_CKPT,
    ckpt_dir_paths,
    load_checkpoint,
    predict_dataset,
    train_model,
    training_is_finished,
)
from utils.gpu import nvidia_smi_gpus, select_gpu_free_mib, vram_mib_from_gib
from utils.workers import resolve_workers

DEFAULT_RESERVE_GIB = 16.0
CHILD_ENV = "RPT_TRAIN_CHILD"


def log(msg: str) -> None:
    print(msg, flush=True)


def run_dir(
    models_root: Path,
    image_days: int,
    horizon: int,
    *,
    init_from_image_days: int | None = None,
) -> Path:
    return models_root / model_run_tag(
        image_days, horizon, init_from_image_days=init_from_image_days
    )


def ensemble_pred_path(
    models_root: Path,
    image_days: int,
    horizon: int,
    *,
    init_from_image_days: int | None = None,
) -> Path:
    return run_dir(
        models_root, image_days, horizon, init_from_image_days=init_from_image_days
    ) / "ensemble_pred.feather"


def seed_pred_path(
    models_root: Path,
    image_days: int,
    horizon: int,
    seed: int,
    *,
    init_from_image_days: int | None = None,
) -> Path:
    return run_dir(
        models_root, image_days, horizon, init_from_image_days=init_from_image_days
    ) / f"seed{seed}_pred.feather"


def validate_init_from(image_days: int, init_from_image_days: int) -> None:
    if init_from_image_days not in WINDOW_DAYS:
        raise ValueError(f"unsupported --init-from-image-days={init_from_image_days}")
    if cnn_num_blocks(init_from_image_days) + 1 != cnn_num_blocks(image_days):
        raise ValueError(
            f"layer transfer requires source blocks + 1 = target blocks; "
            f"I{init_from_image_days} has {cnn_num_blocks(init_from_image_days)} blocks, "
            f"I{image_days} has {cnn_num_blocks(image_days)}"
        )


def init_checkpoint_path(
    models_root: Path,
    init_from_image_days: int,
    horizon: int,
    seed: int,
) -> Path:
    src_dir = model_run_dir(models_root, init_from_image_days, horizon, seed)
    return src_dir / BEST_CKPT


def merge_ensemble(
    models_root: Path,
    image_days: int,
    horizon: int,
    seeds: list[int],
    *,
    init_from_image_days: int | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        path = seed_pred_path(
            models_root,
            image_days,
            horizon,
            seed,
            init_from_image_days=init_from_image_days,
        )
        if not path.exists():
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


def prepare_datasets(
    images_root: Path,
    image_days: int,
    horizon: int,
    *,
    split_seed: int,
    year_limit: int | None,
    max_samples: int | None,
    models_root: Path,
) -> tuple[ImageLabelDataset, ImageLabelDataset, ImageLabelDataset, int]:
    samples = collect_samples(
        images_root,
        image_days,
        horizon,
        year_limit=year_limit,
        max_samples=max_samples,
    )
    train_samples, val_samples, test_samples = split_samples(samples, seed=split_seed)
    log(
        f"I{image_days}/R{horizon} split_seed={split_seed} samples "
        f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}"
    )

    cache_path = None
    if year_limit is None and max_samples is None:
        cache_path = pixel_stats_cache_path(
            models_root, image_days, horizon, split_seed
        )
    pixel_mean, pixel_std = load_or_compute_pixel_stats(
        images_root,
        image_days,
        horizon,
        train_samples,
        split_seed=split_seed,
        cache_path=cache_path,
    )
    log(f"pixel_mean={pixel_mean:.6f} pixel_std={pixel_std:.6f}")

    mmap_cache: dict = {}
    train_ds = ImageLabelDataset(
        images_root, image_days, train_samples, pixel_mean, pixel_std, mmap_cache=mmap_cache
    )
    val_ds = ImageLabelDataset(
        images_root, image_days, val_samples, pixel_mean, pixel_std, mmap_cache=mmap_cache
    )
    test_ds = ImageLabelDataset(
        images_root, image_days, test_samples, pixel_mean, pixel_std, mmap_cache=mmap_cache
    )
    return train_ds, val_ds, test_ds, len(test_samples)


def train_one_seed(
    *,
    train_ds: ImageLabelDataset,
    val_ds: ImageLabelDataset,
    test_ds: ImageLabelDataset,
    n_test: int,
    models_root: Path,
    image_days: int,
    horizon: int,
    seed: int,
    device: str | None,
    batch_size: int,
    fresh: bool,
    skip_train: bool,
    skip_predict: bool,
    init_from_image_days: int | None,
) -> None:
    ckpt_dir = model_run_dir(
        models_root,
        image_days,
        horizon,
        seed,
        init_from_image_days=init_from_image_days,
    )
    _, best_path = ckpt_dir_paths(ckpt_dir)
    pred_path = seed_pred_path(
        models_root,
        image_days,
        horizon,
        seed,
        init_from_image_days=init_from_image_days,
    )

    if fresh and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    if fresh and pred_path.exists():
        pred_path.unlink()

    tag = model_run_tag(image_days, horizon, init_from_image_days=init_from_image_days)
    log(f"{tag} train_seed={seed}")

    init_from_checkpoint: Path | None = None
    if init_from_image_days is not None:
        init_from_checkpoint = init_checkpoint_path(
            models_root, init_from_image_days, horizon, seed
        )
        if not init_from_checkpoint.is_file():
            raise FileNotFoundError(
                f"missing init checkpoint for layer transfer: {init_from_checkpoint}"
            )

    if skip_train and training_is_finished(ckpt_dir):
        log(f"skip train; load {best_path}")
        from models.train_loop import _device

        model, _cfg = load_checkpoint(best_path, _device(device))
    elif training_is_finished(ckpt_dir):
        log(f"training finished; load {best_path}")
        from models.train_loop import _device

        model, _cfg = load_checkpoint(best_path, _device(device))
    else:
        model = train_model(
            train_ds,
            val_ds,
            image_days=image_days,
            horizon=horizon,
            seed=seed,
            pixel_mean=train_ds.pixel_mean,
            pixel_std=train_ds.pixel_std,
            n_test=n_test,
            ckpt_dir=ckpt_dir,
            init_from_checkpoint=init_from_checkpoint,
            init_from_image_days=init_from_image_days,
            device=device,
            batch_size=batch_size,
            log_fn=log,
        )

    if skip_predict and pred_path.exists():
        log(f"skip predict; existing {pred_path}")
        return

    pred_df = predict_dataset(model, test_ds, device=device, batch_size=batch_size)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_feather(pred_path)
    log(f"saved OOS predictions n={len(pred_df)} -> {pred_path}")


def job_is_done(
    models_root: Path,
    image_days: int,
    horizon: int,
    seed: int,
    *,
    init_from_image_days: int | None,
    skip_predict: bool,
) -> bool:
    ckpt_dir = model_run_dir(
        models_root,
        image_days,
        horizon,
        seed,
        init_from_image_days=init_from_image_days,
    )
    if not training_is_finished(ckpt_dir):
        return False
    if skip_predict:
        return True
    return seed_pred_path(
        models_root,
        image_days,
        horizon,
        seed,
        init_from_image_days=init_from_image_days,
    ).is_file()


def parse_gpu_ids(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    ids = [int(p.strip()) for p in raw.split(",") if p.strip() != ""]
    if not ids:
        raise ValueError("--gpu-ids is empty")
    return ids


def gpu_ids_from_device(device: str | None) -> list[int] | None:
    if device is None or device == "cuda":
        return None
    if device == "cpu":
        return None
    if device.startswith("cuda:"):
        return [int(device.split(":", 1)[1])]
    raise ValueError(f"unsupported --device={device}")


def vram_need_mib(image_days: int, override_gib: float | None) -> int:
    gib = override_gib if override_gib is not None else VRAM_PER_JOB_GIB[image_days]
    return vram_mib_from_gib(gib)


def build_child_cmd(args: argparse.Namespace, image_days: int, horizon: int, seed: int) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
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
        "--models",
        str(args.models),
    ]
    if args.images is not None:
        cmd.extend(["--images", str(args.images)])
    if args.init_from_image_days is not None:
        cmd.extend(["--init-from-image-days", str(args.init_from_image_days)])
    if args.year_limit is not None:
        cmd.extend(["--year-limit", str(args.year_limit)])
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.skip_train:
        cmd.append("--skip-train")
    if args.skip_predict:
        cmd.append("--skip-predict")
    return cmd


def _pump_stdout(proc: subprocess.Popen, prefix: str) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        log(f"{prefix}{line.rstrip()}")


def run_child_job(
    args: argparse.Namespace,
    image_days: int,
    horizon: int,
    seed: int,
    gpu_id: int,
) -> None:
    cmd = build_child_cmd(args, image_days, horizon, seed)
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env[CHILD_ENV] = "1"
    tag = model_run_tag(
        image_days, horizon, init_from_image_days=args.init_from_image_days
    )
    prefix = f"[{tag} seed{seed} gpu{gpu_id}] "
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
    jobs: list[tuple[int, int, int]],
    gpu_free: list[tuple[int, int]],
) -> None:
    remaining = {idx: free for idx, free in gpu_free}
    if not remaining:
        raise RuntimeError("no GPU with enough free memory to host a training job")
    for image_days, _horizon, _seed in jobs:
        need = vram_need_mib(image_days, args.vram_per_job_gb)
        if not any(free >= need for free in remaining.values()):
            raise RuntimeError(
                f"I{image_days} needs {need} MiB VRAM but max free among selected GPUs "
                f"is {max(remaining.values())} MiB"
            )
    n_conc, diag = resolve_workers(
        len(jobs),
        reserve_gib=args.reserve_gib,
        mem_per_worker_gib=TRAIN_JOB_RAM_GIB,
        override=args.workers,
    )
    min_need = min(vram_need_mib(d, args.vram_per_job_gb) for d, _h, _s in jobs)
    max_gpu_slots = sum(free // min_need for free in remaining.values())
    per_gpu_cap = len(remaining) * TRAIN_JOBS_PER_GPU
    n_conc = min(n_conc, max(1, max_gpu_slots), per_gpu_cap, len(jobs))
    log(
        f"parallel jobs={len(jobs)} n_conc={n_conc} gpu_free={gpu_free} "
        f"per_gpu_cap={per_gpu_cap} ram_diag={diag}"
    )

    lock = threading.Lock()
    gpu_cv = threading.Condition(lock)

    def acquire_gpu(image_days: int) -> int:
        need = vram_need_mib(image_days, args.vram_per_job_gb)
        with gpu_cv:
            while True:
                candidates = [idx for idx, free in remaining.items() if free >= need]
                if candidates:
                    gpu_id = max(candidates, key=lambda idx: remaining[idx])
                    remaining[gpu_id] -= need
                    log(
                        f"acquire gpu={gpu_id} need_mib={need} "
                        f"remaining={dict(remaining)}"
                    )
                    return gpu_id
                gpu_cv.wait()

    def release_gpu(gpu_id: int, image_days: int) -> None:
        need = vram_need_mib(image_days, args.vram_per_job_gb)
        with gpu_cv:
            remaining[gpu_id] += need
            gpu_cv.notify_all()

    def run_job(job: tuple[int, int, int]) -> None:
        image_days, horizon, seed = job
        gpu_id = acquire_gpu(image_days)
        try:
            run_child_job(args, image_days, horizon, seed, gpu_id)
        finally:
            release_gpu(gpu_id, image_days)

    with ThreadPoolExecutor(max_workers=n_conc) as pool:
        futs = [pool.submit(run_job, job) for job in jobs]
        for fut in as_completed(futs):
            fut.result()


def run_local_jobs(
    args: argparse.Namespace,
    configs: list[tuple[int, int]],
    seeds: list[int],
) -> None:
    from config import IMAGES_ROOT

    images_root = args.images if args.images is not None else IMAGES_ROOT
    for image_days, horizon in configs:
        train_ds, val_ds, test_ds, n_test = prepare_datasets(
            images_root,
            image_days,
            horizon,
            split_seed=args.split_seed,
            year_limit=args.year_limit,
            max_samples=args.max_samples,
            models_root=args.models,
        )
        for seed in seeds:
            train_one_seed(
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                n_test=n_test,
                models_root=args.models,
                image_days=image_days,
                horizon=horizon,
                seed=seed,
                device=args.device,
                batch_size=args.batch_size,
                fresh=False,
                skip_train=args.skip_train,
                skip_predict=args.skip_predict,
                init_from_image_days=args.init_from_image_days,
            )
        if args.all_seeds:
            ensemble = merge_ensemble(
                args.models,
                image_days,
                horizon,
                seeds,
                init_from_image_days=args.init_from_image_days,
            )
            out_path = ensemble_pred_path(
                args.models,
                image_days,
                horizon,
                init_from_image_days=args.init_from_image_days,
            )
            ensemble.to_feather(out_path)
            log(f"ensemble n={len(ensemble)} -> {out_path}")


def resolve_configs(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.all_configs:
        if args.image_days is not None or args.horizon is not None:
            raise ValueError("--all-configs cannot be combined with --image-days/--horizon")
        configs = [(d, h) for d in WINDOW_DAYS for h in WINDOW_DAYS]
    else:
        if args.image_days is None or args.horizon is None:
            raise ValueError("require --image-days and --horizon, or --all-configs")
        configs = [(args.image_days, args.horizon)]
    if args.init_from_image_days is not None:
        if args.all_configs:
            src_blocks = cnn_num_blocks(args.init_from_image_days)
            configs = [
                (image_days, horizon)
                for image_days, horizon in configs
                if cnn_num_blocks(image_days) == src_blocks + 1
            ]
            if not configs:
                raise ValueError(
                    f"no I/R configs accept --init-from-image-days={args.init_from_image_days}"
                )
        else:
            for image_days, _horizon in configs:
                validate_init_from(image_days, args.init_from_image_days)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN on OHLC images")
    parser.add_argument("--images", type=Path, default=None, help="processed/images root")
    parser.add_argument("--models", type=Path, default=MODELS_ROOT)
    parser.add_argument("--image-days", type=int, default=None, choices=WINDOW_DAYS)
    parser.add_argument("--horizon", type=int, default=None, choices=WINDOW_DAYS)
    parser.add_argument("--seed", type=int, default=0, help="optimization seed (init / shuffle)")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=TRAIN_VAL_SPLIT_SEED,
        help="fixed 70/30 train/val split seed (shared across ensemble members)",
    )
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help=f"train/predict seeds 0..{N_ENSEMBLE - 1} and write ensemble_pred.feather",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="train all image-days × horizon pairs (9 models)",
    )
    parser.add_argument(
        "--init-from-image-days",
        type=int,
        default=None,
        choices=WINDOW_DAYS,
        help="layer transfer: init conv blocks from I{src}/R{horizon} best.pt (5->20, 20->60)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu-ids", type=str, default=None, help="comma-separated physical GPU ids")
    parser.add_argument("--gpu-min-free-gb", type=float, default=GPU_MIN_FREE_GIB)
    parser.add_argument(
        "--vram-per-job-gb",
        type=float,
        default=None,
        help="override per-job VRAM reserve (default depends on image-days)",
    )
    parser.add_argument("--workers", type=int, default=None, help="max concurrent train processes")
    parser.add_argument("--reserve-gib", type=float, default=DEFAULT_RESERVE_GIB)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--year-limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()

    configs = resolve_configs(args)
    seeds = list(range(N_ENSEMBLE)) if args.all_seeds else [args.seed]
    is_child = os.environ.get(CHILD_ENV) == "1"

    if args.fresh:
        for image_days, horizon in configs:
            target = run_dir(
                args.models,
                image_days,
                horizon,
                init_from_image_days=args.init_from_image_days,
            )
            if target.exists():
                shutil.rmtree(target)

    jobs = [
        (image_days, horizon, seed)
        for image_days, horizon in configs
        for seed in seeds
        if not job_is_done(
            args.models,
            image_days,
            horizon,
            seed,
            init_from_image_days=args.init_from_image_days,
            skip_predict=args.skip_predict,
        )
    ]
    log(f"pending jobs={len(jobs)} configs={configs} seeds={seeds}")

    use_cpu = args.device == "cpu"
    can_parallel = (not is_child) and (not use_cpu) and len(jobs) > 1
    if can_parallel:
        gpu_ids = parse_gpu_ids(args.gpu_ids)
        if gpu_ids is None:
            gpu_ids = gpu_ids_from_device(args.device)
        min_free_mib = vram_mib_from_gib(args.gpu_min_free_gb)
        gpu_free = select_gpu_free_mib(gpu_ids, min_free_mib=min_free_mib)
        if not gpu_free:
            raise RuntimeError(
                f"no GPU with free memory >= {args.gpu_min_free_gb} GiB; "
                f"nvidia-smi={nvidia_smi_gpus()}"
            )
        log(f"nvidia-smi={nvidia_smi_gpus()}")
        if jobs:
            launch_parallel_jobs(args, jobs, gpu_free)
        if args.all_seeds:
            for image_days, horizon in configs:
                ensemble = merge_ensemble(
                    args.models,
                    image_days,
                    horizon,
                    seeds,
                    init_from_image_days=args.init_from_image_days,
                )
                out_path = ensemble_pred_path(
                    args.models,
                    image_days,
                    horizon,
                    init_from_image_days=args.init_from_image_days,
                )
                ensemble.to_feather(out_path)
                log(f"ensemble n={len(ensemble)} -> {out_path}")
        return

    run_local_jobs(args, configs, seeds)


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    main()
