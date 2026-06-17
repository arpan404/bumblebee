#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import re
import signal
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from bumblebee.rl.envs.gymnasium import GymMouseImitationEnv
from bumblebee.rl.envs.mouse import MouseEnvConfig, VirtualScreen
from bumblebee.rl.training.callbacks import RichTrainingCallback

CHECKPOINT_PREFIX = "sac_mouse"
CHECKPOINT_RE = re.compile(rf"^{re.escape(CHECKPOINT_PREFIX)}_(\d+)_steps\.zip$")


def _parallel_env_default() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _runtime_defaults(preset: str) -> dict[str, object]:
    envs = _parallel_env_default()
    manual = {
        "batch_size": 512,
        "checkpoint_freq": 25_000,
        "device": "auto",
        "gradient_steps": 8,
        "num_envs": 1,
        "torch_interop_threads": None,
        "torch_threads": None,
        "train_freq": 16,
        "vec_env": "dummy",
    }
    presets: dict[str, dict[str, object]] = {
        "manual": {},
        "cpu-fast": {
            "device": "cpu",
            "num_envs": envs,
            "vec_env": "dummy",
            "batch_size": 512,
            "train_freq": 16,
            "gradient_steps": 8,
            "torch_threads": 1,
            "torch_interop_threads": 1,
        },
        "cpu-parallel": {
            "device": "cpu",
            "num_envs": envs,
            "vec_env": "subproc",
            "batch_size": 512,
            "train_freq": 16,
            "gradient_steps": 8,
            "torch_threads": 1,
            "torch_interop_threads": 1,
        },
        "m4-max": {
            # Local benchmarks for this tiny MLP + Python env are faster on CPU
            # than MPS. This preset uses more envs and conservative torch threads.
            "device": "cpu",
            "num_envs": envs,
            "vec_env": "subproc",
            "batch_size": 512,
            "train_freq": 16,
            "gradient_steps": 8,
            "torch_threads": 1,
            "torch_interop_threads": 1,
        },
        "mps-heavy": {
            # This is for GPU utilization experiments, not best wall-clock speed.
            "device": "mps",
            "num_envs": envs,
            "vec_env": "subproc",
            "batch_size": 2048,
            "train_freq": 16,
            "gradient_steps": 32,
            "torch_threads": 1,
            "torch_interop_threads": 1,
        },
    }
    defaults = manual | presets[preset]
    return defaults


def apply_runtime_defaults(args: argparse.Namespace) -> argparse.Namespace:
    for name, value in _runtime_defaults(args.preset).items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Bumblebee mouse policy with SAC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        default="manual",
        choices=["manual", "cpu-fast", "cpu-parallel", "m4-max", "mps-heavy"],
        help=(
            "Runtime preset. Explicit CLI values still override preset values. "
            "`m4-max` targets fastest local wall-clock training; `mps-heavy` "
            "intentionally increases update work to exercise Apple GPU."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="artifacts/mouse_demonstrations.npz",
        help="Processed demonstration dataset from prepare_mouse_data.py.",
    )
    parser.add_argument("--output-dir", default="artifacts/rl/sac_mouse")
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--screen-width", type=int, default=4096)
    parser.add_argument("--screen-height", type=int, default=2304)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--dt", type=float, default=1 / 120)
    parser.add_argument("--max-velocity-px-s", type=float, default=6500.0)
    parser.add_argument("--target-radius", type=float, default=8.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=None,
        metavar="UNITS",
        help=(
            "Hidden layer sizes for SAC actor/critic MLPs, for example "
            "`--net-arch 512 512`. Existing checkpoints keep their saved architecture."
        ),
    )
    parser.add_argument("--train-freq", type=int, default=None)
    parser.add_argument("--gradient-steps", type=int, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of virtual mouse environments.",
    )
    parser.add_argument(
        "--vec-env",
        default=None,
        choices=["dummy", "subproc"],
        help="Vector env backend. Dummy is usually faster for this lightweight env; subproc can help only when rollout collection is the bottleneck.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device for SB3. `auto` prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Optional SAC .zip model to resume from. Also accepts `latest`, "
            "`best`, or `final`."
        ),
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help=(
            "Resume from the newest checkpoint in output-dir/checkpoints if one "
            "exists. Falls back to latest_model.zip/final_model.zip."
        ),
    )
    parser.add_argument(
        "--resume-replay-buffer",
        default=None,
        help=(
            "Optional replay buffer .pkl to restore when resuming SAC. By default, "
            "checkpoint replay buffers are matched by step number, then "
            "latest_replay_buffer.pkl is used if present."
        ),
    )
    parser.add_argument(
        "--timesteps-mode",
        default="target",
        choices=["target", "additional"],
        help=(
            "`target` trains until model.num_timesteps reaches --timesteps when "
            "resuming. `additional` always trains --timesteps more steps."
        ),
    )
    parser.add_argument(
        "--reset-num-timesteps",
        action="store_true",
        help="Reset SB3 timestep counters on resume instead of continuing them.",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Run Stable-Baselines3's environment checker before training.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show SB3's progress bar. Requires tqdm/rich in some SB3 versions.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Torch intra-op threads. Presets keep this small to avoid oversubscription.",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=None,
        help="Torch inter-op threads.",
    )
    parser.add_argument(
        "--subproc-start-method",
        default=None,
        choices=["fork", "forkserver", "spawn"],
        help="Multiprocessing start method for SubprocVecEnv.",
    )
    parser.add_argument(
        "--optimize-memory-usage",
        action="store_true",
        help="Use SB3's memory-optimized replay buffer.",
    )
    parser.add_argument(
        "--no-checkpoint-replay-buffer",
        action="store_false",
        dest="checkpoint_replay_buffer",
        help="Do not save the replay buffer at every checkpoint.",
    )
    parser.add_argument(
        "--no-final-replay-buffer",
        action="store_false",
        dest="final_replay_buffer",
        help="Do not save latest_replay_buffer.pkl at shutdown.",
    )
    parser.add_argument(
        "--reward-terms",
        action="store_true",
        help="Include per-step reward term dictionaries in env info. Slower.",
    )
    parser.add_argument(
        "--log-interval-seconds",
        type=float,
        default=5.0,
        help="Rich metrics refresh interval.",
    )
    parser.set_defaults(checkpoint_replay_buffer=True, final_replay_buffer=True)
    return apply_runtime_defaults(parser.parse_args())


def make_env(dataset: str, config: MouseEnvConfig, seed: int, log_dir: Path, rank: int):
    def _factory():
        env = GymMouseImitationEnv(dataset, config=config, seed=seed + rank)
        return Monitor(env, filename=str(log_dir / f"monitor-{rank}.csv"))

    return _factory


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob(f"{CHECKPOINT_PREFIX}_*_steps.zip"):
        step = checkpoint_step(path)
        if step is not None:
            checkpoints.append((step, path))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def replay_buffer_for_checkpoint(checkpoint_path: Path) -> Path | None:
    step = checkpoint_step(checkpoint_path)
    if step is None:
        return None
    candidate = (
        checkpoint_path.parent / f"{CHECKPOINT_PREFIX}_replay_buffer_{step}_steps.pkl"
    )
    return candidate if candidate.exists() else None


def resolve_resume_model(
    resume_from: str | None,
    *,
    output_dir: Path,
    checkpoint_dir: Path,
) -> Path | None:
    if resume_from is None:
        return None

    if resume_from == "latest":
        checkpoint = latest_checkpoint(checkpoint_dir)
        if checkpoint is not None:
            return checkpoint
        for fallback in ("latest_model.zip", "final_model.zip"):
            candidate = output_dir / fallback
            if candidate.exists():
                return candidate
        return None

    if resume_from == "best":
        candidate = output_dir / "best_model.zip"
        return candidate if candidate.exists() else None

    if resume_from == "final":
        candidate = output_dir / "final_model.zip"
        return candidate if candidate.exists() else None

    return Path(resume_from)


def resolve_replay_buffer(
    resume_replay_buffer: str | None,
    *,
    resume_model_path: Path,
    output_dir: Path,
) -> Path | None:
    if resume_replay_buffer:
        return Path(resume_replay_buffer)

    checkpoint_replay_buffer = replay_buffer_for_checkpoint(resume_model_path)
    if checkpoint_replay_buffer is not None:
        return checkpoint_replay_buffer

    candidate = output_dir / "latest_replay_buffer.pkl"
    return candidate if candidate.exists() else None


def configure_torch_runtime(args: argparse.Namespace) -> None:
    if args.torch_threads is None:
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    else:
        torch_threads = max(1, args.torch_threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(torch_threads))
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(torch_threads))
        torch.set_num_threads(torch_threads)

    if args.torch_interop_threads is not None:
        torch.set_num_interop_threads(max(1, args.torch_interop_threads))


def print_runtime_advice(
    console: Console, args: argparse.Namespace, device: str, num_envs: int
) -> None:
    if num_envs == 1:
        console.print(
            "[yellow]Only one env is enabled. Rollout collection is serial, so most CPU cores will stay idle.[/yellow]"
        )
    if args.vec_env == "dummy" and num_envs > 1:
        console.print(
            "[yellow]DummyVecEnv batches multiple envs in one Python process. Use `--vec-env subproc` when CPU rollout collection is the bottleneck.[/yellow]"
        )
    if device == "mps" and (args.batch_size < 1024 or args.gradient_steps < 16):
        console.print(
            "[yellow]MPS is selected, but this is a small MLP workload. Larger `--batch-size`/`--gradient-steps` improves GPU occupancy, though CPU can still be faster.[/yellow]"
        )
    if device == "mps" and num_envs == 1:
        console.print(
            "[yellow]MPS training waits on one env. Use more envs if you want the accelerator fed consistently.[/yellow]"
        )


def install_shutdown_handlers() -> None:
    def _request_shutdown(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")

    signal.signal(signal.SIGTERM, _request_shutdown)


def policy_kwargs_from_args(args: argparse.Namespace) -> dict[str, object] | None:
    if not args.net_arch:
        return None
    return {"net_arch": list(args.net_arch)}


def main() -> None:
    args = parse_args()
    if args.auto_resume:
        if args.resume_from is not None:
            raise ValueError("use either --auto-resume or --resume-from, not both")
        args.resume_from = "latest"

    dataset = Path(args.dataset)
    if not dataset.exists():
        raise FileNotFoundError(
            f"dataset not found: {dataset}. Run scripts/prepare_mouse_data.py first."
        )

    output_dir = Path(args.output_dir)
    log_dir = output_dir / "logs"
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = MouseEnvConfig(
        screen=VirtualScreen(width=args.screen_width, height=args.screen_height),
        max_steps=args.max_steps,
        dt=args.dt,
        max_velocity_px_s=args.max_velocity_px_s,
        target_radius_px=args.target_radius,
        record_reward_terms=args.reward_terms,
    )

    if args.check_env:
        check_env(GymMouseImitationEnv(dataset, config=config, seed=args.seed))

    console = Console()
    device = resolve_device(args.device)
    if device == "mps":
        torch.set_float32_matmul_precision("high")
    if torch.backends.mps.is_available() and device != "mps":
        if args.preset in {"cpu-fast", "cpu-parallel", "m4-max"}:
            console.print(
                "[cyan]Apple MPS is available, but this CPU preset is selected because this small SAC workload benchmarks faster on CPU. Use `--preset mps-heavy` to exercise the GPU.[/cyan]"
            )
        else:
            console.print(
                "[yellow]Apple MPS is available but not selected. Use `--device mps` or `--device auto`.[/yellow]"
            )

    configure_torch_runtime(args)
    num_envs = max(1, args.num_envs)
    env_factories = [
        make_env(str(dataset), config, args.seed, log_dir, rank)
        for rank in range(num_envs)
    ]
    env = (
        SubprocVecEnv(env_factories, start_method=args.subproc_start_method)
        if args.vec_env == "subproc" and num_envs > 1
        else DummyVecEnv(env_factories)
    )
    print_runtime_advice(console, args, device, num_envs)

    console.print(
        Panel.fit(
            f"[bold]Preset[/bold]: {args.preset}\n"
            f"[bold]Torch device[/bold]: {device}\n"
            f"[bold]MPS available[/bold]: {torch.backends.mps.is_available()}\n"
            f"[bold]Torch threads[/bold]: {torch.get_num_threads()}\n"
            f"[bold]Torch interop threads[/bold]: {torch.get_num_interop_threads()}\n"
            f"[bold]Vector env[/bold]: {args.vec_env}\n"
            f"[bold]Envs[/bold]: {num_envs}\n"
            f"[bold]Train freq[/bold]: {args.train_freq}\n"
            f"[bold]Gradient steps[/bold]: {args.gradient_steps}\n"
            f"[bold]Batch size[/bold]: {args.batch_size}\n"
            f"[bold]Net arch[/bold]: {args.net_arch or 'checkpoint/default'}\n"
            f"[bold]Dataset[/bold]: {dataset}",
            title="Training runtime",
            border_style="magenta",
        )
    )

    resume_model_path = resolve_resume_model(
        args.resume_from,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
    )
    if args.resume_from and resume_model_path is None:
        console.print(
            f"[yellow]No resume model found for `{args.resume_from}`. Starting a new model.[/yellow]"
        )

    if resume_model_path:
        if not resume_model_path.exists():
            raise FileNotFoundError(f"resume model not found: {resume_model_path}")
        model = SAC.load(
            resume_model_path,
            env=env,
            tensorboard_log=str(log_dir),
            device=device,
            print_system_info=True,
        )
        console.print(f"[green]Resumed model:[/green] {resume_model_path}")
        if args.net_arch:
            console.print(
                "[yellow]Ignoring --net-arch for resumed checkpoint. Start with a "
                "fresh --output-dir to train a different architecture.[/yellow]"
            )
        replay_buffer = resolve_replay_buffer(
            args.resume_replay_buffer,
            resume_model_path=resume_model_path,
            output_dir=output_dir,
        )
        if replay_buffer:
            if not replay_buffer.exists():
                raise FileNotFoundError(
                    f"resume replay buffer not found: {replay_buffer}"
                )
            initial_replay_buffer = model.replay_buffer
            model.load_replay_buffer(replay_buffer)
            replay_envs = getattr(model.replay_buffer, "n_envs", None)
            if replay_envs != num_envs:
                model.replay_buffer = initial_replay_buffer
                console.print(
                    "[yellow]Skipped replay buffer because it was saved with "
                    f"{replay_envs} envs, but this run uses {num_envs}. "
                    "Pass the same --num-envs to reuse it.[/yellow]"
                )
            else:
                console.print(f"[green]Loaded replay buffer:[/green] {replay_buffer}")
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=min(args.learning_starts, max(args.timesteps // 10, 1)),
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            ent_coef="auto",
            tensorboard_log=str(log_dir),
            seed=args.seed,
            verbose=0,
            device=device,
            optimize_memory_usage=args.optimize_memory_usage,
            policy_kwargs=policy_kwargs_from_args(args),
        )

    resumed = resume_model_path is not None and not args.reset_num_timesteps
    current_timesteps = int(getattr(model, "num_timesteps", 0))
    if resumed and args.timesteps_mode == "target":
        learn_timesteps = max(args.timesteps - current_timesteps, 0)
        display_total_timesteps = max(args.timesteps, current_timesteps)
    elif resumed:
        learn_timesteps = args.timesteps
        display_total_timesteps = current_timesteps + args.timesteps
    else:
        learn_timesteps = args.timesteps
        display_total_timesteps = args.timesteps

    console.print(
        Panel.fit(
            f"[bold]Resume model[/bold]: {resume_model_path or 'none'}\n"
            f"[bold]Current timesteps[/bold]: {current_timesteps:,}\n"
            f"[bold]Timesteps mode[/bold]: {args.timesteps_mode}\n"
            f"[bold]Learn call timesteps[/bold]: {learn_timesteps:,}\n"
            f"[bold]Displayed target[/bold]: {display_total_timesteps:,}\n"
            f"[bold]Reset counters[/bold]: {args.reset_num_timesteps}",
            title="Resume state",
            border_style="blue",
        )
    )

    save_freq = max(args.checkpoint_freq // num_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(checkpoint_dir),
        name_prefix="sac_mouse",
        save_replay_buffer=args.checkpoint_replay_buffer,
        save_vecnormalize=True,
    )
    rich_callback = RichTrainingCallback(
        total_timesteps=display_total_timesteps,
        output_dir=output_dir,
        checkpoint_freq=args.checkpoint_freq,
        log_interval_seconds=args.log_interval_seconds,
    )
    callbacks = CallbackList([checkpoint_callback, rich_callback])

    if learn_timesteps <= 0:
        console.print(
            "[green]Target timesteps already reached; saving current model.[/green]"
        )
        model.save(output_dir / "latest_model.zip")
        if args.final_replay_buffer:
            model.save_replay_buffer(output_dir / "latest_replay_buffer.pkl")
        return

    install_shutdown_handlers()
    interrupted = False
    try:
        model.learn(
            total_timesteps=learn_timesteps,
            callback=callbacks,
            tb_log_name="sac_mouse",
            reset_num_timesteps=not resumed or args.reset_num_timesteps,
            progress_bar=args.progress_bar,
        )
    except KeyboardInterrupt as exc:
        interrupted = True
        console.print(
            f"[yellow]Training interrupted: {exc}. Saving latest model.[/yellow]"
        )
    finally:
        latest_path = output_dir / "latest_model.zip"
        model.save(latest_path)
        if args.final_replay_buffer:
            replay_path = output_dir / "latest_replay_buffer.pkl"
            model.save_replay_buffer(replay_path)

    if interrupted:
        print(f"saved interrupted model to {latest_path}")
        return

    final_path = output_dir / "final_model.zip"
    model.save(final_path)
    print(f"saved final model to {final_path}")


if __name__ == "__main__":
    main()
