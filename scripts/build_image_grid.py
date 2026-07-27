from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from PIL import Image, ImageOps


def build_grid(image_paths: list[Path], output_path: Path, *, cols: int = 2, padding: int = 12, bg: str = "#111827") -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)

    tiles = []
    for img in images:
        canvas = Image.new("RGB", (max_w, max_h), color=bg)
        fitted = ImageOps.contain(img, (max_w, max_h))
        x = (max_w - fitted.width) // 2
        y = (max_h - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        tiles.append(canvas)

    rows = math.ceil(len(tiles) / cols)
    grid = Image.new(
        "RGB",
        (
            cols * max_w + (cols + 1) * padding,
            rows * max_h + (rows + 1) * padding,
        ),
        color=bg,
    )

    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        x = padding + col * (max_w + padding)
        y = padding + row * (max_h + padding)
        grid.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simple PNG grid from generated sample images.")
    parser.add_argument("--input-dir", required=True, help="Directory containing PNG/JPG/WebP images.")
    parser.add_argument("--output", required=True, help="Output grid image path.")
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--padding", type=int, default=12)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    image_paths = sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
    )
    if not image_paths:
        raise SystemExit(f"No images found in {input_dir}")

    build_grid(image_paths, Path(args.output), cols=args.cols, padding=args.padding)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
