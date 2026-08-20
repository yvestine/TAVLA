"""Launch the four hand-noise Co-training jobs on GPUs 0, 4, 5, and 6.

The four jobs are:

  * base initialization, 50% real / 50% simulation
  * real-checkpoint initialization, 50% real / 50% simulation
  * base initialization, 30% real / 70% simulation
  * real-checkpoint initialization, 30% real / 70% simulation

The training process saves only its final step.  This launcher keeps the
individual training logs and renders one tqdm progress bar per job by parsing
the ``Step N:`` lines emitted by ``scripts/train.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from tqdm import tqdm


SIM_REPO_ID = "local/tavla_single_arm_ee_wrench_sim_hand_noise_50"
REAL_REPO_ID = "local/tavla_single_arm_ee_wrench"
CONFIG_50 = "pi0_lora_user_single_arm_ee_wrench_cotrain_base_50_50"
CONFIG_50_REAL = "pi0_lora_user_single_arm_ee_wrench_cotrain_realinit_50_50"
CONFIG_70 = "pi0_lora_user_single_arm_ee_wrench_cotrain_base_70sim_30real"
CONFIG_70_REAL = "pi0_lora_user_single_arm_ee_wrench_cotrain_realinit_70sim_30real"


@dataclass(frozen=True)
class Job:
    name: str
    config_name: str
    gpu: int
    assets_dir: Path


JOBS = (
    Job("base_50_50", CONFIG_50, 0, Path("assets/pi0_lora_user_single_arm_ee_wrench_cotrain_50_50")),
    Job("realinit_50_50", CONFIG_50_REAL, 4, Path("assets/pi0_lora_user_single_arm_ee_wrench_cotrain_50_50")),
    Job("base_70sim_30real", CONFIG_70, 5, Path("assets/pi0_lora_user_single_arm_ee_wrench_cotrain_70sim_30real")),
    Job(
        "realinit_70sim_30real",
        CONFIG_70_REAL,
        6,
        Path("assets/pi0_lora_user_single_arm_ee_wrench_cotrain_70sim_30real"),
    ),
)

STEP_RE = re.compile(r"Step\s+(\d+):")
LOSS_RE = re.compile(r"loss=([-+0-9.eE]+)")


def _repo_ready(repo_id: str) -> bool:
    path = LEROBOT_HOME / repo_id
    return path.is_dir() and (path / "meta").is_dir() and (path / "meta" / "info.json").is_file()


def _stats_ready(job: Job, root: Path) -> bool:
    assets_dir = root / job.assets_dir
    return all(
        (assets_dir / repo_id / "norm_stats.json").is_file()
        for repo_id in (REAL_REPO_ID, SIM_REPO_ID)
    )


def _prepare_stats(root: Path, jobs: tuple[Job, ...], python: str, skip_stats: bool) -> None:
    if skip_stats:
        return

    configs_to_prepare = []
    seen_assets = set()
    for job in jobs:
        if job.assets_dir in seen_assets or _stats_ready(job, root):
            continue
        seen_assets.add(job.assets_dir)
        configs_to_prepare.append(job.config_name)

    for config_name in configs_to_prepare:
        print(f"Preparing normalization statistics for {config_name}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["JAX_PLATFORMS"] = "cpu"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        subprocess.run(
            [python, "scripts/compute_norm_stats.py", "--config-name", config_name],
            cwd=root,
            env=env,
            check=True,
        )


def _monitor_process(job: Job, process: subprocess.Popen[str], bar: tqdm, log_path: Path, total_steps: int) -> None:
    loss = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()

            step_match = STEP_RE.search(line)
            if step_match:
                step = int(step_match.group(1))
                loss_match = LOSS_RE.search(line)
                if loss_match:
                    loss = loss_match.group(1)
                bar.n = min(total_steps, step + 1)
                if loss is None:
                    bar.set_postfix_str(f"step={step}", refresh=False)
                else:
                    bar.set_postfix_str(f"step={step} loss={loss}", refresh=False)
                bar.refresh()

    return_code = process.wait()
    bar.n = total_steps if return_code == 0 else bar.n
    bar.set_description(f"{job.name} {'DONE' if return_code == 0 else 'FAILED'}")
    bar.refresh()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/cotrain_hand_noise_50"))
    parser.add_argument("--skip-stats", action="store_true", help="Do not compute missing dataset norm stats.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing experiment directories.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching training.")
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    root = Path(__file__).resolve().parents[1]
    python = sys.executable

    if args.dry_run:
        for job in JOBS:
            command = [
                python,
                "scripts/train.py",
                job.config_name,
                "--exp-name",
                "final",
                "--num-train-steps",
                str(args.steps),
                "--save-interval",
                str(args.steps + 1),
                "--log-interval",
                str(args.log_interval),
                "--batch-size",
                str(args.batch_size),
            ]
            if args.overwrite:
                command.append("--overwrite")
            print(f"GPU {job.gpu}: {' '.join(command)}")
        return

    missing = [repo_id for repo_id in (REAL_REPO_ID, SIM_REPO_ID) if not _repo_ready(repo_id)]
    if missing:
        raise FileNotFoundError(
            "Missing LeRobot dataset(s): "
            + ", ".join(missing)
            + ". Run scripts/convert_tavla_raw_hdf5_to_lerobot.py first."
        )

    _prepare_stats(root, JOBS, python, args.skip_stats)
    missing_stats = [
        str(job.assets_dir / repo_id)
        for job in JOBS
        for repo_id in (REAL_REPO_ID, SIM_REPO_ID)
        if not (root / job.assets_dir / repo_id / "norm_stats.json").is_file()
    ]
    if missing_stats:
        raise FileNotFoundError(
            "Normalization statistics are still missing after preparation: " + ", ".join(sorted(set(missing_stats)))
        )

    processes: list[tuple[Job, subprocess.Popen[str]]] = []
    bars = []
    threads = []
    try:
        for position, job in enumerate(JOBS):
            command = [
                python,
                "scripts/train.py",
                job.config_name,
                "--exp-name",
                "final",
                "--num-train-steps",
                str(args.steps),
                "--save-interval",
                str(args.steps + 1),
                "--log-interval",
                str(args.log_interval),
                "--batch-size",
                str(args.batch_size),
            ]
            if args.overwrite:
                command.append("--overwrite")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            processes.append((job, process))
            bar = tqdm(
                total=args.steps,
                desc=f"{job.name} RUNNING",
                position=position,
                leave=True,
                dynamic_ncols=True,
            )
            bars.append(bar)
            thread = threading.Thread(
                target=_monitor_process,
                args=(job, process, bar, root / args.log_dir / f"{job.name}.log", args.steps),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        while any(thread.is_alive() for thread in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping all four training jobs...")
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for thread in threads:
            thread.join()
        for bar in bars:
            bar.close()

    failed = [(job.name, process.returncode) for job, process in processes if process.returncode != 0]
    if failed:
        raise SystemExit(f"Training failures: {failed}")

    print("All four training jobs completed.")
    for job in JOBS:
        print(f"{job.name}: checkpoints/{job.config_name}/final/{args.steps - 1}/params")


if __name__ == "__main__":
    main()
