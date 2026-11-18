"""Paper figure: training images and their views from a flat plane and from estimated depth.

Each person gets one row per view source. The first column is the real training
image and the others are its views at every angle that was rendered. A missing
tile means the view was dropped because too little of it was observed.

    python make_example_views.py --dataset ilidsvid --data-root data\\ilids --work work --out figures
    python make_example_views.py --dataset prid2011 --data-root data\\prid --work work --out figures --suffix _a45

Nothing is replaced: if the figure already exists, a numbered name is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from reid3d import data as D

LABEL_WIDTH, HEADER_HEIGHT, GAP = 90, 28, 4


def unused_path(path: Path) -> Path:
    candidate, number = path, 1
    while candidate.exists():
        number += 1
        candidate = path.with_name(f"{path.stem}_{number}{path.suffix}")
    return candidate


def view_tiles(frame: Path, data_root: Path, views_root: Path, size: tuple[int, int]) -> tuple[list[str], list[Image.Image | None]]:
    """Column titles and tiles for every angle in this view folder, None where a view was dropped."""
    angles = json.loads((views_root / "manifest.json").read_text(encoding="utf8"))["angles"]
    folder = D.view_folder(frame, data_root, views_root)
    titles, tiles = [], []
    for index, (yaw, pitch) in enumerate(angles):
        titles.append(f"yaw {yaw:+.0f}" if not pitch else f"yaw {yaw:+.0f} pitch {pitch:+.0f}")
        path = folder / f"view_{index:02d}.png"
        tiles.append(Image.open(path).convert("RGB").resize(size) if path.exists() else None)
    return titles, tiles


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(D.DATASETS))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sources", nargs="+", default=["flat", "estimated"])
    parser.add_argument("--suffix", default="", help="angle-set suffix of the view folders, such as _a45")
    parser.add_argument("--people", type=int, default=3)
    parser.add_argument("--split", type=int, default=0)
    args = parser.parse_args(argv)

    height, width = D.IMAGE_SIZE[args.dataset]
    size = (width, height)
    tracklets = D.DATASETS[args.dataset](args.data_root)
    split = D.make_split(tracklets, args.split)
    frames = D.sample_images(tracklets, split.train_pids[:args.people], 1, args.split)

    rows: list[tuple[str, list[Image.Image | None]]] = []
    titles: list[str] = []
    for frame, pid, _ in frames:
        original = Image.open(frame).convert("RGB").resize(size)
        for source in args.sources:
            views_root = args.work / args.dataset / f"views_{source}{args.suffix}"
            titles, tiles = view_tiles(frame, args.data_root, views_root, size)
            rows.append((f"ID {pid}\n{source}", [original, *tiles]))

    columns = 1 + len(titles)
    canvas = Image.new("RGB", (LABEL_WIDTH + columns * (width + GAP), HEADER_HEIGHT + len(rows) * (height + GAP)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(["real image", *titles]):
        draw.text((LABEL_WIDTH + column * (width + GAP) + 4, 8), title, fill="black")
    for row, (label, tiles) in enumerate(rows):
        top = HEADER_HEIGHT + row * (height + GAP)
        draw.multiline_text((6, top + height // 2 - 12), label, fill="black")
        for column, tile in enumerate(tiles):
            left = LABEL_WIDTH + column * (width + GAP)
            if tile is None:
                draw.rectangle([left, top, left + width - 1, top + height - 1], outline="grey")
                draw.text((left + 8, top + height // 2), "dropped", fill="grey")
            else:
                canvas.paste(tile, (left, top))

    args.out.mkdir(parents=True, exist_ok=True)
    target = unused_path(args.out / f"example_views_{args.dataset}{args.suffix}.png")
    canvas.save(target)
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
