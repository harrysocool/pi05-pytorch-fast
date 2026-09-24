#!/usr/bin/env python3
"""LIBERO closed-loop eval — SnapFlow 1-NFE PyTorch (W4A4 + torch.compile).

  source env.sh
  python scripts/libero_eval.py --tasks libero_10 --episodes 2 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:512")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

def _default_libero_root() -> Path:
    env = os.environ.get("LIBERO_ROOT")
    if env:
        return Path(env).expanduser()
    pkg = Path(__file__).resolve().parents[1]
    for cand in (
        Path.home() / "openpi" / "third_party" / "libero",
        Path.home() / "model_data" / "libero",
        pkg / ".vendor" / "libero",
    ):
        if (cand / "libero").is_dir():
            return cand
    return Path.home() / "model_data" / "libero"


_LIBERO_ROOT = _default_libero_root()
if _LIBERO_ROOT.is_dir():
    sys.path.insert(0, str(_LIBERO_ROOT))

DEFAULT_POLICY = Path.home() / "model_data" / "pi05_snapflow_1nfe"


def _parse_task_ids(raw: str | None):
    if raw is None:
        return None
    s = raw.strip().lstrip("[").rstrip("]").strip()
    if not s:
        return None
    return sorted({int(tok) for tok in s.split(",") if tok.strip() != ""})


def parse_args():
    p = argparse.ArgumentParser(description="LIBERO eval for pi05-pytorch-fast")
    p.add_argument("--policy-id", default=None, help="Checkpoint dir (default: ~/model_data/pi05_snapflow_1nfe)")
    p.add_argument("--tasks", default="libero_10")
    p.add_argument("--task-ids", default=None, help="e.g. 0,1,2 (default: all tasks in suite)")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--n-action-steps", type=int, default=10)
    p.add_argument("--n-inference-steps", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=Path("eval_out/libero_snapflow"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path(os.environ.get("PALIGEMMA_TOKENIZER_PATH", Path.home() / "model_data/paligemma2-3b-pt-224")),
    )
    return p.parse_args()


def _build_eval_argv(args, policy_id: str, *, suite: str | None = None, task_id: int | None = None):
    import torch

    tasks = suite if suite is not None else args.tasks
    argv = [
        "libero_eval",
        f"--output_dir={args.output_dir}",
        f"--policy.path={policy_id}",
        f"--policy.n_action_steps={args.n_action_steps}",
        "--env.type=libero",
        f"--env.task={tasks}",
        "--env.max_parallel_tasks=1",
        "--eval.batch_size=1",
        f"--eval.n_episodes={args.episodes}",
        f"--policy.num_inference_steps={args.n_inference_steps}",
    ]
    task_ids = [task_id] if task_id is not None else _parse_task_ids(args.task_ids)
    if task_ids is not None:
        argv.append("--env.task_ids=[" + ",".join(str(t) for t in task_ids) + "]")
    if not torch.cuda.is_available():
        argv.append("--policy.device=cpu")
    if args.seed is not None:
        argv.append(f"--seed={args.seed}")
    return argv


def _policy_path_for_eval(policy_id: str) -> str:
    from pi05_fast.checkpoint_compat import sanitize_snapflow_checkpoint

    src = Path(policy_id).expanduser()
    teacher = Path(os.environ.get("SNAPFLOW_TEACHER_POLICY", Path.home() / "model_data/pi05_libero_v044"))
    for note in sanitize_snapflow_checkpoint(src, teacher):
        print(f"policy checkpoint: {note}", flush=True)
    return str(src)


def _coerce_gym_final_info(info):
    import numpy as np

    if not isinstance(info, dict) or "final_info" not in info:
        return info
    fi = info["final_info"]
    if isinstance(fi, dict):
        return info
    items = list(fi) if fi is not None else []
    succ = []
    for item in items:
        if not isinstance(item, dict):
            succ.append(False)
            continue
        s = item.get("is_success", item.get("success", False))
        if hasattr(s, "item"):
            s = s.item()
        succ.append(bool(s))
    out = dict(info)
    out["final_info"] = {"is_success": np.asarray(succ, dtype=bool)}
    return out


def _patch_lerobot_eval_gymnasium():
    import lerobot.scripts.lerobot_eval as ev

    if getattr(ev, "_pi05_final_info_patched", False):
        return

    class _InfoCoerceEnv:
        def __init__(self, env):
            object.__setattr__(self, "_env", env)

        def __getattr__(self, name):
            return getattr(self._env, name)

        def step(self, action):
            obs, reward, terminated, truncated, info = self._env.step(action)
            return obs, reward, terminated, truncated, _coerce_gym_final_info(info)

    orig = ev.eval_policy

    def wrapped_eval_policy(env, *args, **kwargs):
        return orig(_InfoCoerceEnv(env), *args, **kwargs)

    ev.eval_policy = wrapped_eval_policy
    ev._pi05_final_info_patched = True
    print("eval: coerce gymnasium<1.0 final_info for LeRobot rollout", flush=True)


def _run_eval(args, policy_id: str):
    from lerobot.scripts.lerobot_eval import eval_main
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.utils import init_logging

    init_logging()
    register_third_party_plugins()
    _patch_lerobot_eval_gymnasium()
    sys.argv = _build_eval_argv(args, policy_id)
    print(f"lerobot-eval argv: {sys.argv[1:]}", flush=True)
    t0 = time.perf_counter()
    eval_main()
    wall_s = time.perf_counter() - t0
    info_path = args.output_dir / "eval_info.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    return info, wall_s


def _suite_n_tasks(suite: str) -> int:
    from libero.libero import benchmark

    ts = benchmark.get_benchmark_dict()[suite]()
    n = getattr(ts, "n_tasks", None) or len(getattr(ts, "tasks", []))
    if not n:
        raise RuntimeError(f"could not determine task count for {suite!r}")
    return int(n)


def _worklist(args):
    suites = [s.strip() for s in args.tasks.split(",") if s.strip()]
    tids = _parse_task_ids(args.task_ids)
    work = []
    for suite in suites:
        ids = tids if tids is not None else range(_suite_n_tasks(suite))
        for tid in ids:
            work.append((suite, int(tid)))
    return work


def _eval_one_task(args, policy_id: str, suite: str, tid: int, out_dir: Path):
    from lerobot.scripts.lerobot_eval import eval_main

    out_dir.mkdir(parents=True, exist_ok=True)
    sys.argv = _build_eval_argv(args, policy_id, suite=suite, task_id=tid)
    print(f"lerobot-eval argv: {sys.argv[1:]}", flush=True)
    eval_main()
    info_path = out_dir / "eval_info.json"
    return json.loads(info_path.read_text()) if info_path.is_file() else {}


def _run_eval_resumable(args, policy_id: str):
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.utils import init_logging

    init_logging()
    register_third_party_plugins()
    _patch_lerobot_eval_gymnasium()
    work = _worklist(args)
    ckpt = args.output_dir / "by_task"
    print(f"[resume] {len(work)} task(s); checkpoints under {ckpt}", flush=True)
    per_task = []
    t0 = time.perf_counter()
    for suite, tid in work:
        tag = f"{suite}__task{tid}"
        out_dir = ckpt / tag
        res = out_dir / "eval_info.json"
        if res.is_file():
            print(f"[resume] skip {tag} (already done)", flush=True)
            per_task.append((tag, json.loads(res.read_text())))
            continue
        print(f"[resume] running {tag} ...", flush=True)
        per_task.append((tag, _eval_one_task(args, policy_id, suite, tid, out_dir)))
    wall_s = time.perf_counter() - t0
    tot_succ, tot_eps = 0.0, 0
    by_task = {}
    for tag, info in per_task:
        ov = info.get("overall") or {}
        pc, ne = ov.get("pc_success"), ov.get("n_episodes")
        by_task[tag] = ov
        if pc is not None and ne:
            tot_succ += (pc / 100.0) * ne
            tot_eps += ne
    overall = {
        "pc_success": (100.0 * tot_succ / tot_eps) if tot_eps else None,
        "n_episodes": tot_eps,
    }
    return {"overall": overall, "by_task": by_task}, wall_s


def _print_summary(info: dict, wall_s: float) -> None:
    print(f"\n== eval done in {wall_s / 60:.1f} min ==")
    overall = info.get("overall")
    if overall:
        print(
            f"  LIBERO overall : {overall.get('pc_success', float('nan')):.1f}% "
            f"success over {overall.get('n_episodes', 0)} episodes"
        )


def _patch_paligemma_tokenizer(local_path: Path | None) -> None:
    if not local_path or not local_path.is_dir():
        return
    from transformers import AutoTokenizer

    orig = AutoTokenizer.from_pretrained

    def _from_pretrained(name_or_path, *args, **kwargs):
        name = str(name_or_path)
        if "paligemma" in name:
            return orig(str(local_path), *args, **kwargs)
        return orig(name_or_path, *args, **kwargs)

    AutoTokenizer.from_pretrained = _from_pretrained  # type: ignore[method-assign]
    print(f"tokenizer: using local {local_path}", flush=True)


def _check_deps() -> None:
    try:
        from lerobot.policies.pi05 import PI05Policy  # noqa: F401
    except ImportError as exc:
        sys.exit(f"lerobot PI05Policy missing — run scripts/setup.sh ({exc})")
    try:
        import libero.libero.benchmark  # noqa: F401
    except ImportError as exc:
        sys.exit(
            f"LIBERO missing.\n  export LIBERO_ROOT={_LIBERO_ROOT}\n  pip install -e $LIBERO_ROOT\n  ({exc})"
        )


def main() -> None:
    args = parse_args()
    _check_deps()
    policy_id = str(Path(args.policy_id or DEFAULT_POLICY).expanduser())
    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    src_policy = policy_id
    policy_id = _policy_path_for_eval(policy_id)

    from pi05_fast.rocm_pi05_optim import install_rocm_pi05_eval_hooks
    from pi05_fast.snapflow_pi05 import checkpoint_has_snapflow_mlp, install_pi05_snapflow_eval_hooks

    if checkpoint_has_snapflow_mlp(src_policy) or checkpoint_has_snapflow_mlp(policy_id):
        install_pi05_snapflow_eval_hooks()
    install_rocm_pi05_eval_hooks()
    _patch_paligemma_tokenizer(args.tokenizer_path.expanduser())

    print(f"policy           : {policy_id}")
    print(f"tasks            : {args.tasks}")
    print(f"episodes/task    : {args.episodes}")
    print(f"n_inference_steps: {args.n_inference_steps}")
    print(f"output           : {args.output_dir.resolve()}")

    import torch

    if not torch.cuda.is_available():
        print("warn: torch.cuda not available — will run on CPU", flush=True)

    info, wall_s = _run_eval_resumable(args, policy_id) if args.resume else _run_eval(args, policy_id)
    _print_summary(info, wall_s)
    overall = info.get("overall", {})
    summary = {
        "recipe": "pi05-pytorch-fast",
        "policy_id": policy_id,
        "tasks": args.tasks,
        "episodes_per_task": args.episodes,
        "n_inference_steps": args.n_inference_steps,
        "n_action_steps": args.n_action_steps,
        "pc_success": overall.get("pc_success"),
        "n_episodes": overall.get("n_episodes"),
        "wall_s": wall_s,
    }
    if "by_task" in info:
        summary["by_task"] = info["by_task"]
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
