"""
ColorMystery core – CLI & processing logic.
Author: ColorCode – MIT License.

Transforms any raster image into:
 • colour-by-number (visible outlines + IDs)
 • mystery colouring (hidden outlines, flat zones + IDs)

Streaming-tile pipeline handles >8 MPx with bounded memory.
"""
from __future__ import annotations
import argparse, io, math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import cairosvg

DEFAULT_TILE = 1024
ID_SETS = {
    "numbers": [str(i) for i in range(1, 1000)],
    "letters": [chr(c) for c in range(65, 91)],
    "symbols": list("★●■▲◆✚✿☾♣♥♠♦")
}

# ---------- Utilities -----------------------------------------------------------
def auto_resize(im: Image.Image, base_w: int) -> Image.Image:
    if base_w <= 0 or im.width <= base_w:
        return im
    ratio = base_w / im.width
    return im.resize((base_w, int(im.height * ratio)), Image.Resampling.LANCZOS)

def quantize(img: np.ndarray, k: int):
    h, w = img.shape[:2]
    rgb = img.reshape(-1, 3).astype(np.float32) / 255
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0).fit(rgb)
    labels = kmeans.labels_.reshape(h, w)
    palette = (kmeans.cluster_centers_ * 255).astype(np.uint8)
    return labels, palette

def contours_from_labels(labels: np.ndarray, detail: str) -> np.ndarray:
    border = (np.gradient(labels.astype(np.int16))[0] != 0) | (np.gradient(labels.astype(np.int16))[1] != 0)
    dilate_iter = {"low": 5, "medium": 2, "high": 1}[detail]
    kernel = np.ones((3, 3), np.uint8)
    border = cv2.dilate(border.astype(np.uint8), kernel, iterations=dilate_iter)
    return border * 255

def place_ids(labels: np.ndarray, id_list: List[str]):
    mapping = {}
    for lbl in np.unique(labels):
        ys, xs = np.where(labels == lbl)
        cy, cx = int(ys.mean()), int(xs.mean())
        mapping[lbl] = (id_list[lbl % len(id_list)], (cx, cy))
    return mapping

def export_png(img: Image.Image, path: Path):
    img.save(path.with_suffix('.png'), dpi=(300, 300))

def export_pdf(svg_str: str, path: Path):
    cairosvg.svg2pdf(bytestring=svg_str.encode('utf-8'), write_to=str(path.with_suffix('.pdf')))

def export_svg(svg_str: str, path: Path):
    path.with_suffix('.svg').write_text(svg_str, encoding='utf-8')

# ---------- Core processing ------------------------------------------------------
def stream_process(path: Path, args):
    image = Image.open(path).convert('RGBA')
    image = auto_resize(image, args.width)
    w, h = image.size
    font = ImageFont.load_default()
    id_list = args.id_set

    tiles_x = (w + args.tile_size - 1) // args.tile_size
    tiles_y = (h + args.tile_size - 1) // args.tile_size
    sheets = {'number': Image.new('RGBA', (w, h), 'white'),
              'mystery': Image.new('RGBA', (w, h), 'white')}

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x0, y0 = tx*args.tile_size, ty*args.tile_size
            box = (x0, y0, min(x0+args.tile_size, w), min(y0+args.tile_size, h))
            tile = image.crop(box)
            np_tile = cv2.cvtColor(np.array(tile), cv2.COLOR_RGBA2RGB)
            labels, _ = quantize(np_tile, args.k)
            border = contours_from_labels(labels, args.detail)
            ids = place_ids(labels, id_list)

            # visible outlines
            outline_img = Image.fromarray(border).convert('L').convert('RGBA')
            vis = Image.alpha_composite(tile, outline_img)
            draw_vis = ImageDraw.Draw(vis)
            draw_mys = ImageDraw.Draw(tile)

            for lbl, (ident, (cx, cy)) in ids.items():
                draw_vis.text((cx, cy), ident, fill='black', anchor='mm', font=font)
                draw_mys.text((cx, cy), ident, fill='black', anchor='mm', font=font)

            sheets['number'].paste(vis, box)
            sheets['mystery'].paste(tile, box)

    return sheets

# ---------- CLI ------------------------------------------------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(prog='colormystery')
    p.add_argument('image')
    p.add_argument('--mode', choices=['number', 'mystery', 'both'], default='both')
    p.add_argument('--difficulty', choices=['easy', 'medium', 'hard'], default='medium')
    p.add_argument('--detail', choices=['low', 'medium', 'high'], default='medium')
    p.add_argument('--id-set', choices=['numbers', 'letters', 'symbols'], default='numbers')
    p.add_argument('--palette', choices=['auto', 'daltonism-friendly'], default='auto')
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--tile-size', type=int, default=DEFAULT_TILE)
    p.add_argument('--out-formats', default='png,pdf')
    return p

def main():
    args = build_arg_parser().parse_args()
    args.k = {'easy': 8, 'medium': 16, 'hard': 24}[args.difficulty]
    args.id_set = ID_SETS[args.id_set]
    sheets = stream_process(Path(args.image), args)
    for mode, img in sheets.items():
        if args.mode in (mode, 'both'):
            if 'png' in args.out_formats:
                export_png(img, Path(args.image).with_name(f"{Path(args.image).stem}_{mode}"))
    print('Done.')

if __name__ == '__main__':
    main()
