"""Combine the example-view panels into one Figure 1 image for submission (CMC: one image, <= 6.5 in wide, >= 600 dpi).

    python make_figure1.py figures/example_views_prid2011.png figures/example_views_ilidsvid.png
        --out figures --crop 0:flat,flatm,shuffled,estimated 1:flat,flatm,shuffled,estimated

With --crop, each panel keeps one person's rows and is relabeled in a font that stays at
8 pt or more at the final width. Row labels are vertical (rotated 90°) unless --label-layout horizontal.
Without --crop, the panels are joined as they are.
All text is Times New Roman; the script stops if the font file is missing.
Nothing is replaced: if the figure already exists, a numbered name is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from make_result_figures import unused_path

WIDTH_IN, DPI, GAP = 6.5, 600, 24
MIN_POINTS = 8
# Layout written by make_example_views.py: a header strip, a label column, then one row per person and view source.
HEADER, LABEL_WIDTH, ROW_GAP = 28, 90, 4
READABLE = {"flat": "center-pivot flat", "flatm": "matched flat", "shuffled": "shuffled depth",
            "estimated": "estimated depth"}
# Label column width for vertical row labels (one rotated line) and for horizontal ones (two lines).
NEW_HEADER, NEW_LABEL_WIDTH = 50, {"vertical": 52, "horizontal": 190}
FONT_FILES = ("times.ttf", r"C:\Windows\Fonts\times.ttf", "Times New Roman.ttf",
              "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")


def font(size: int) -> ImageFont.FreeTypeFont:
    """Times New Roman at `size` pixels; no silent fallback to another font."""
    for name in FONT_FILES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise RuntimeError("Times New Roman (times.ttf) not found; Figure 1 needs it")


def vertical_text(text: str, size: int) -> Image.Image:
    """A mask of `text` rotated 90° counter-clockwise (reads bottom to top), cropped to the ink."""
    face = font(size)
    left, top, right, bottom = face.getbbox(text)
    mask = Image.new("L", (right - left + 4, bottom - top + 4), 0)
    ImageDraw.Draw(mask).text((2 - left, 2 - top), text, fill=255, font=face)
    return mask.rotate(90, expand=True)


def crop_person(panel: Image.Image, person: int, sources: list[str], titles: list[str], tile_height: int,
                size: int, layout: str = "vertical") -> Image.Image:
    """One person's rows of a panel, with new column titles and readable row labels in a `size`-pixel font."""
    pitch = tile_height + ROW_GAP
    top = HEADER + person * len(sources) * pitch
    tiles = panel.crop((LABEL_WIDTH, top, panel.width, top + len(sources) * pitch - ROW_GAP))
    label_width = NEW_LABEL_WIDTH[layout]
    out = Image.new("RGB", (label_width + tiles.width, NEW_HEADER + tiles.height), "white")
    out.paste(tiles, (label_width, NEW_HEADER))
    draw = ImageDraw.Draw(out)
    column_width = tiles.width / len(titles)
    for index, title in enumerate(titles):
        centre = label_width + (index + 0.5) * column_width
        draw.text((centre, NEW_HEADER // 2), title, fill="black", font=font(size), anchor="mm")
    for index, source in enumerate(sources):
        middle = NEW_HEADER + index * pitch + tile_height // 2
        label = READABLE.get(source, source)
        if layout == "vertical":
            mask = vertical_text(label, size)
            if mask.height > tile_height:
                raise ValueError(f"row label {label!r} is taller than its row; lower the font size")
            # Centered on the row, and horizontally in the label column.
            out.paste("black", ((label_width - mask.width) // 2, middle - mask.height // 2), mask)
        else:
            draw.multiline_text((8, middle), label.replace(" ", "\n", 1), fill="black", font=font(size),
                                anchor="lm", spacing=4)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("panels", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--crop", nargs="+", metavar="PERSON:SOURCES",
                        help="one per panel, such as 0:flat,flatm,shuffled,estimated: keep that person's rows only")
    parser.add_argument("--titles", default="real,−30°,−15°,+15°,+30°", help="column titles of the cropped panels")
    parser.add_argument("--tile-height", type=int, default=256)
    parser.add_argument("--label-layout", choices=["vertical", "horizontal"], default="vertical",
                        help="row labels of the cropped panels: rotated 90° (default) or horizontal on two lines")
    args = parser.parse_args(argv)
    panels = [Image.open(p).convert("RGB") for p in args.panels]
    label_h = 70
    if args.crop:
        if len(args.crop) != len(panels):
            parser.error("give one --crop entry per panel")
        # Choose the label font so it is at least MIN_POINTS after scaling to the final width.
        label_width = NEW_LABEL_WIDTH[args.label_layout]
        width = sum(label_width + p.width - LABEL_WIDTH for p in panels) + GAP * (len(panels) - 1)
        scale = WIDTH_IN * DPI / width
        size = round(MIN_POINTS / 72 * DPI / scale) + 1
        titles = args.titles.split(",")
        panels = [crop_person(panel, int(spec.split(":")[0]), spec.split(":")[1].split(","), titles,
                              args.tile_height, size, args.label_layout) for panel, spec in zip(panels, args.crop)]
        print(f"Label font {size} px before scaling = {size * scale / DPI * 72:.1f} pt at {WIDTH_IN} in wide")
        letter_size = round(size * 1.3)
    else:
        height = max(p.height for p in panels)
        panels = [p.resize((round(p.width * height / p.height), height), Image.LANCZOS) for p in panels]
        letter_size = 40
    height = max(p.height for p in panels)
    canvas = Image.new("RGB", (sum(p.width for p in panels) + GAP * (len(panels) - 1), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    left = 0
    for letter, panel in zip("abcdefgh", panels):
        # CMC places panel letters in the top-left corner of each panel.
        draw.text((left + 8, label_h // 2), f"({letter})", fill="black", font=font(letter_size), anchor="lm")
        canvas.paste(panel, (left, label_h))
        left += panel.width + GAP
    target_w = round(WIDTH_IN * DPI)
    canvas = canvas.resize((target_w, round(canvas.height * target_w / canvas.width)), Image.LANCZOS)
    args.out.mkdir(parents=True, exist_ok=True)
    target = unused_path(args.out / "Fig1_example_views.tif")
    canvas.save(target, dpi=(DPI, DPI), compression="tiff_lzw")
    canvas.resize((canvas.width // 3, canvas.height // 3), Image.LANCZOS).save(target.with_suffix(".png"))
    print(f"Wrote {target} ({canvas.width}x{canvas.height} px, {canvas.width / DPI:.2f} x {canvas.height / DPI:.2f} in)")


if __name__ == "__main__":
    main()
