"""Run a grid of conditions, appending one JSON line per finished run.

Finished conditions are skipped, so an interrupted grid can simply be restarted.
Within one split, fusions reuse the single-modality models already trained.

    python run_grid.py --dataset prid2011 --data-root D:/data/prid2011 \\
        --arms real_aug real_synth --views estimated=D:/work/prid/views_estimated \\
        --k 1 2 5 10 --splits 0 1 2 3 4 5 6 7 8 9 --out results/q1_prid2011.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

from reid3d import data as D
from reid3d.train import ARMS, FUSIONS, SINGLE_MODALITIES, SYNTH_ARMS, Config, run_condition


def parse_views(items: list[str]) -> dict[str, str]:
    """Turn ['estimated=path', ...] into {'estimated': 'path', ...}."""
    views = {}
    for item in items:
        tag, separator, path = item.partition("=")
        if not separator or not tag or not path:
            raise ValueError(f"--views expects tag=path, got '{item}'")
        views[tag] = path
    return views


def finished_keys(out: Path) -> set[str]:
    """Names of the runs already saved. This only reads: a line damaged by an interrupted save is ignored."""
    if not out.exists():
        return set()
    keys = set()
    for line in out.read_text(encoding="utf8").splitlines():
        try:
            keys.add(json.loads(line)["key"])
        except (ValueError, KeyError, TypeError):
            print(f"Ignoring a damaged result line in {out.name}", flush=True)
    return keys


def append_result(out: Path, record: dict) -> None:
    """Add one result to the end of the file on its own line. Existing lines are never changed."""
    existing = out.read_text(encoding="utf8") if out.exists() else ""
    start = "\n" if existing and not existing.endswith("\n") else ""
    with out.open("a", encoding="utf8") as handle:
        handle.write(start + json.dumps(record) + "\n")


def conditions(modalities: list[str], arms: list[str], views: dict[str, str]):
    """Yield (modality, arm, view tag, view path) for every requested condition."""
    for modality, arm in itertools.product(modalities, arms):
        if arm in SYNTH_ARMS:
            for tag, path in views.items():
                yield modality, arm, tag, path
        else:
            yield modality, arm, None, None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(D.DATASETS))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--modalities", nargs="+", default=["rgb"], choices=[*SINGLE_MODALITIES, *FUSIONS])
    parser.add_argument("--arms", nargs="+", default=["real", "real_aug"], choices=sorted(ARMS))
    parser.add_argument("--k", nargs="+", default=["1", "2", "5", "10"], help="images per identity, or 'all'")
    parser.add_argument("--splits", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--id-loss", choices=["ce", "arcface"], default="ce")
    parser.add_argument("--query-frames", type=int, default=None, help="1 for the single-shot test protocol")
    parser.add_argument("--eval-frames", type=int, default=16, help="frames per tracklet at test time")
    parser.add_argument("--views", nargs="*", default=[], help="rendered view sources, as tag=path")
    parser.add_argument("--est-root", default=None, help="output folder of estimate_depth.py")
    parser.add_argument("--low-res", type=int, default=None, help="simulated camera height in pixels")
    parser.add_argument("--stop-at", type=float, default=None, help="Unix time after which no new model is started")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "dinov2_s"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    views = parse_views(args.views)
    if set(args.arms) & SYNTH_ARMS and not views:
        parser.error("the arms synth and real_synth need --views")
    if any("est_depth" in FUSIONS.get(m, (m,)) for m in args.modalities) and not args.est_root:
        parser.error("the modalities est_depth and rgbe need --est-root")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = finished_keys(out)
    tracklets = D.DATASETS[args.dataset](args.data_root)
    print(f"{args.dataset}: {len({t.pid for t in tracklets})} identities, {len(tracklets)} tracklets", flush=True)
    height, width = D.IMAGE_SIZE[args.dataset]

    for split in args.splits:
        cache: dict = {}
        for k_text, (modality, arm, tag, path) in itertools.product(args.k, conditions(args.modalities, args.arms, views)):
            cfg = Config(dataset=args.dataset, data_root=args.data_root, arm=arm, modality=modality,
                         k=None if k_text == "all" else int(k_text), split_seed=split, iterations=args.iterations,
                         id_loss=args.id_loss, query_frames=args.query_frames, eval_frames=args.eval_frames,
                         synth_root=path, synth_tag=tag, est_root=args.est_root, low_res=args.low_res,
                         height=height, width=width, pretrained=not args.no_pretrained, device=args.device,
                         backbone=args.backbone)
            if cfg.key() in done:
                continue
            if args.stop_at is not None and time.time() >= args.stop_at:
                print("Time limit reached. Run again to continue from here.", flush=True)
                return
            record = run_condition(cfg, tracklets, cache)
            append_result(out, record)
            print(f"split={split} k={k_text:>3} {modality:<9} {arm:<10} {tag or '-':<10} R1={record['rank1']:5.1f} "
                  f"R5={record['rank5']:5.1f} mAP={record['mAP']:5.1f} n={record['num_queries']} "
                  f"train={record['train_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
