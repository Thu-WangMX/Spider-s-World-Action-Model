import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
project_src = project_root / "src"
libero_root = Path(os.environ.get("LIBERO_ROOT", project_root.parent / "LIBERO"))
for path in (libero_root, project_src, project_root):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from libero.libero import benchmark


@dataclass(frozen=True)
class Task:
    suite: str
    task_id: int


@dataclass
class Worker:
    gpu_id: str
    slot_id: int
    proc: subprocess.Popen
    log_file: Path
    manifest_file: Path


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def build_task_list(suites: list[str]) -> list[Task]:
    benchmark_dict = benchmark.get_benchmark_dict()
    tasks: list[Task] = []
    for suite_name in suites:
        task_suite = benchmark_dict[suite_name]()
        n_tasks = int(task_suite.n_tasks)
        print(f"{suite_name}: {n_tasks} tasks", flush=True)
        for task_id in range(n_tasks):
            tasks.append(Task(suite=suite_name, task_id=task_id))
    print(f"Total tasks: {len(tasks)}", flush=True)
    return tasks


def any_result_exists(output_dir: Path, task: Task) -> bool:
    suite_dir = output_dir / task.suite
    return any(suite_dir.glob(f"gpu*_task{task.task_id}_results.json"))


def write_manifests(
    *,
    tasks: list[Task],
    output_dir: Path,
    gpu_ids: list[str],
    workers_per_gpu: int,
) -> list[tuple[str, int, Path]]:
    manifest_dir = output_dir / "worker_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    slots: list[tuple[str, int, list[Task]]] = []
    for gpu_id in gpu_ids:
        for slot_id in range(workers_per_gpu):
            slots.append((gpu_id, slot_id, []))
    if not slots:
        raise ValueError("No worker slots were requested.")

    for index, task in enumerate(tasks):
        slots[index % len(slots)][2].append(task)

    manifests: list[tuple[str, int, Path]] = []
    for gpu_id, slot_id, assigned in slots:
        manifest_file = manifest_dir / f"gpu{gpu_id}_slot{slot_id}.json"
        payload = [{"suite": task.suite, "task_id": task.task_id} for task in assigned]
        with manifest_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(
            f"manifest gpu={gpu_id} slot={slot_id}: {len(payload)} tasks -> {manifest_file}",
            flush=True,
        )
        if payload:
            manifests.append((gpu_id, slot_id, manifest_file))
    return manifests


def launch_worker(
    *,
    gpu_id: str,
    slot_id: int,
    manifest_file: Path,
    args: argparse.Namespace,
    extra_overrides: list[str],
    env_base: dict[str, str],
) -> Worker:
    output_dir = Path(args.output_dir)
    log_dir = output_dir / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"gpu{gpu_id}_slot{slot_id}.log"

    env = env_base.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["EXP_NAME"] = args.exp_name

    cmd = [
        args.python_bin,
        "experiments/libero/eval_libero_worker.py",
        f"task={args.config}",
        f"ckpt={args.ckpt}",
        f"gpu_id={gpu_id}",
        f"EVALUATION.num_trials={args.num_trials}",
        f"EVALUATION.output_dir={args.output_dir}",
        f"+EVALUATION.task_manifest={manifest_file}",
        f"+EVALUATION.skip_existing={str(args.skip_existing).lower()}",
        f"+EVALUATION.continue_on_error={str(args.continue_on_error).lower()}",
        *extra_overrides,
    ]
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=args.project_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    print(
        f"[{time.strftime('%F %T')}] launch worker gpu={gpu_id} slot={slot_id} "
        f"pid={proc.pid} manifest={manifest_file}",
        flush=True,
    )
    return Worker(gpu_id=gpu_id, slot_id=slot_id, proc=proc, log_file=log_file, manifest_file=manifest_file)


def summarize(args: argparse.Namespace, env_base: dict[str, str]) -> None:
    output_dir = Path(args.output_dir)
    subprocess.run(
        [args.python_bin, "experiments/libero/summarize_results.py", f"--output_dir={output_dir}"],
        cwd=args.project_dir,
        env=env_base,
        check=True,
        text=True,
    )
    if args.plus_summary:
        cmd = [
            args.python_bin,
            "experiments/libero/summarize_libero_plus_results.py",
            "--output_dir",
            str(output_dir),
        ]
        if args.classification_path:
            cmd.extend(["--classification_path", args.classification_path])
        subprocess.run(cmd, cwd=args.project_dir, env=env_base, check=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=str(project_root))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suites", default="libero_spatial,libero_object,libero_goal,libero_10")
    parser.add_argument("--expected-num-tasks", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--extra-override", action="append", default=[])
    parser.add_argument("--classification-path", default=None)
    parser.add_argument("--plus-summary", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--exp-name", default="")
    parser.add_argument("--status-interval", type=int, default=300)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = parse_csv(args.gpu_ids)
    if not gpu_ids:
        raise ValueError("--gpu-ids cannot be empty")
    if args.workers_per_gpu <= 0:
        raise ValueError("--workers-per-gpu must be positive")

    suites = parse_csv(args.suites)
    tasks = build_task_list(suites)
    if args.expected_num_tasks is not None and len(tasks) != args.expected_num_tasks:
        raise ValueError(
            f"Task list has {len(tasks)} tasks, expected {args.expected_num_tasks}. "
            "Check LIBERO_ROOT/PYTHONPATH."
        )

    if args.skip_existing:
        before = len(tasks)
        tasks = [task for task in tasks if not any_result_exists(output_dir, task)]
        print(f"Skipping {before - len(tasks)} completed tasks; {len(tasks)} remain.", flush=True)
    if not tasks:
        print("No tasks remain; summarizing existing results.", flush=True)
        env_base = os.environ.copy()
        summarize(args, env_base)
        return

    env_base = os.environ.copy()
    pythonpath_parts = [str(libero_root), str(project_root / "src"), str(project_root)]
    if env_base.get("PYTHONPATH"):
        pythonpath_parts.append(env_base["PYTHONPATH"])
    env_base["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env_base.setdefault("MUJOCO_GL", "egl")
    env_base.setdefault("PYOPENGL_PLATFORM", "egl")
    env_base.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    env_base.setdefault("TOKENIZERS_PARALLELISM", "false")
    env_base.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    for key, value in {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }.items():
        env_base.setdefault(key, value)

    manifests = write_manifests(
        tasks=tasks,
        output_dir=output_dir,
        gpu_ids=gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
    )
    workers = [
        launch_worker(
            gpu_id=gpu_id,
            slot_id=slot_id,
            manifest_file=manifest_file,
            args=args,
            extra_overrides=args.extra_override,
            env_base=env_base,
        )
        for gpu_id, slot_id, manifest_file in manifests
    ]

    failed: list[Worker] = []
    last_status = 0.0
    while workers:
        still_running: list[Worker] = []
        for worker in workers:
            rc = worker.proc.poll()
            if rc is None:
                still_running.append(worker)
                continue
            if rc != 0:
                failed.append(worker)
                with (output_dir / "failed_workers.txt").open("a", encoding="utf-8") as f:
                    f.write(
                        f"gpu={worker.gpu_id},slot={worker.slot_id},rc={rc},"
                        f"log={worker.log_file},manifest={worker.manifest_file}\n"
                    )
                print(
                    f"[{time.strftime('%F %T')}] FAILED worker gpu={worker.gpu_id} "
                    f"slot={worker.slot_id} rc={rc} log={worker.log_file}",
                    flush=True,
                )
            else:
                print(
                    f"[{time.strftime('%F %T')}] worker done gpu={worker.gpu_id} "
                    f"slot={worker.slot_id}",
                    flush=True,
                )
        workers = still_running

        now = time.time()
        if now - last_status >= args.status_interval:
            print(
                f"[{time.strftime('%F %T')}] status running={len(workers)} failed={len(failed)}",
                flush=True,
            )
            last_status = now

        if failed and not args.continue_on_error:
            for worker in workers:
                worker.proc.terminate()
            time.sleep(5)
            for worker in workers:
                if worker.proc.poll() is None:
                    worker.proc.kill()
            break
        if workers:
            time.sleep(5)

    if failed:
        raise SystemExit(f"{len(failed)} worker(s) failed. See {output_dir / 'failed_workers.txt'}")

    summarize(args, env_base)
    print(f"[{time.strftime('%F %T')}] All workers finished. Summary written under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
