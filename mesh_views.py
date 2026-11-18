"""Bridge to full 3D-mesh views made by the 3D-VITAL generation pipeline (Hunyuan3D-2).

That pipeline is run separately, on a machine with a large GPU, so this tool only
moves data in and out of it:

    export   copy the training frames to one folder with unique file names
    import   convert the generated views into the layout training expects

Unique names are needed because datasets such as PRID-2011 reuse names like
0001.png for every person, while the pipeline matches views to frames by name.

    python mesh_views.py export --dataset prid2011 --data-root D:/data/prid --out D:/work/prid/mesh_input
    (run third_party/3D-VITAL/generation on D:/work/prid/mesh_input/bounding_box_train)
    python mesh_views.py import --generated D:/work/prid/mesh_generated --frames D:/work/prid/mesh_input/frames.json \\
        --out D:/work/prid/views_mesh --keep-list D:/work/prid/mesh_generated/train_cvcf.txt
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from PIL import Image

from reid3d import data as D

VIEW_NAME = re.compile(r"(?P<source>.+)_(?P<view>\d\d)")


def flat_name(relative: Path) -> str:
    """A unique file stem that keeps the frame traceable and avoids the pipeline's 'C0' aerial marker."""
    return "-".join(relative.with_suffix("").parts).replace("C0", "c0")


def export_frames(args: argparse.Namespace) -> None:
    data_root, out = Path(args.data_root), Path(args.out)
    tracklets = D.DATASETS[args.dataset](data_root)
    frames = sorted(D.training_frames(tracklets, args.splits, args.k_max))
    images = out / "bounding_box_train"
    images.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for frame in frames:
        relative = frame.relative_to(data_root)
        name = flat_name(relative)
        if name in mapping:
            raise ValueError(f"two frames share the exported name {name}")
        mapping[name] = relative.as_posix()
        shutil.copyfile(frame, images / f"{name}{frame.suffix.lower()}")
    (out / "frames.json").write_text(json.dumps(mapping, indent=1), encoding="utf8")
    print(f"Exported {len(mapping)} frames to {images}")


def import_views(args: argparse.Namespace) -> None:
    generated, out = Path(args.generated), Path(args.out)
    mapping: dict[str, str] = json.loads(Path(args.frames).read_text(encoding="utf8"))
    keep = None
    if args.keep_list:
        keep = {Path(line.strip()).stem for line in Path(args.keep_list).read_text(encoding="utf8").splitlines() if line.strip()}
    written = rejected = 0
    frames_with_views: set[str] = set()
    for path in sorted(generated.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"} or path.stem.endswith("_visibility"):
            continue
        match = VIEW_NAME.fullmatch(path.stem)
        if match is None or match["source"] not in mapping:
            continue
        if keep is not None and path.stem not in keep:
            rejected += 1
            continue
        relative = Path(mapping[match["source"]])
        target = out / relative.with_suffix("") / f"view_{match['view']}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            image.convert("RGB").save(target, compress_level=1)
        written += 1
        frames_with_views.add(match["source"])
    if not written:
        raise ValueError(f"no generated views under {generated} matched the exported frames")
    manifest = {"geometry": "mesh", "views_written": written, "views_rejected": rejected,
                "frames_with_views": len(frames_with_views), "frames_exported": len(mapping)}
    if args.ms_per_image is not None:
        manifest["total_ms_per_image"] = args.ms_per_image
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(json.dumps(manifest, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--dataset", required=True, choices=sorted(D.DATASETS))
    export.add_argument("--data-root", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--splits", nargs="+", type=int, default=list(range(10)))
    export.add_argument("--k-max", type=int, default=10)
    export.set_defaults(handler=export_frames)
    imported = commands.add_parser("import")
    imported.add_argument("--generated", required=True, help="folder holding the generated views")
    imported.add_argument("--frames", required=True, help="frames.json written by export")
    imported.add_argument("--out", required=True)
    imported.add_argument("--keep-list", help="optional train_cvcf.txt listing the views to keep")
    imported.add_argument("--ms-per-image", type=float, help="measured generation cost, recorded for the cost table")
    imported.set_defaults(handler=import_views)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
