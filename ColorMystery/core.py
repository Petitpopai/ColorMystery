"""
ColorNumber – générateur simplifié de coloriages à numéros
Option --simplify pour limiter les traits redondants.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from skimage.segmentation import find_boundaries
from skimage.measure import label, regionprops

FONT = ImageFont.load_default()

SIMPLIFY_PRESET = {
    "low":    dict(blur=1,  min_area=300,  contour_iter=0),
    "medium": dict(blur=2,  min_area=600,  contour_iter=1),
    "high":   dict(blur=3,  min_area=1200, contour_iter=2),
}

# ---------- traitement principal -----------------------------------------------------
def quantize(img: np.ndarray, k: int, min_area: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    flat = img.reshape(-1, 3).astype(np.float32) / 255
    km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(flat)
    labels = km.labels_.reshape(h, w)

    # fusion des petites régions
    lab = label(labels, connectivity=1)
    for r in regionprops(lab):
        if r.area < min_area:
            y, x = r.coords[0]
            neighbour = labels[max(0, y - 1), x]
            labels[lab == r.label] = neighbour

    palette = (km.cluster_centers_ * 255).astype(np.uint8)
    return labels, palette


def generate_number_sheet(
    image: Image.Image, k: int, simplify: str, detail: str
) -> Image.Image:
    """Retourne l’image finale : A) contours noirs, B) numéros, C) palette."""
    w, h = image.size
    cfg = SIMPLIFY_PRESET[simplify]

    # ① pré-flou
    np_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)
    if cfg["blur"] > 0:
        np_img = cv2.GaussianBlur(np_img, (0, 0), cfg["blur"])

    # ② quantisation + nettoyage
    labels, palette = quantize(np_img, k, cfg["min_area"])
    unique = np.unique(labels)
    mapping = {lbl: idx + 1 for idx, lbl in enumerate(unique)}
    remap_palette = [palette[lbl] for lbl in unique]

    # ③ contours
    border = find_boundaries(labels, mode="thick").astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    border = cv2.dilate(border, kernel, iterations=cfg["contour_iter"])

    sheet = Image.new("RGB", (w, h), "white")
    sheet.paste("black", mask=Image.fromarray(border).convert("L"))
    draw = ImageDraw.Draw(sheet)

    for lbl in unique:
        ys, xs = np.where(labels == lbl)
        draw.text(
            (int(xs.mean()), int(ys.mean())),
            str(mapping[lbl]),
            fill="black",
            anchor="mm",
            font=FONT,
        )

    # ④ palette en bas
    strip_h = 60
    final = Image.new("RGB", (w, h + strip_h), "white")
    final.paste(sheet, (0, 0))
    draw_strip = ImageDraw.Draw(final)
    spacing = max(50, w // max(8, len(unique)))
    for i, col in enumerate(remap_palette):
        x = 10 + i * spacing
        y = h + 10
        draw_strip.rectangle([x, y, x + 40, y + 40], fill=tuple(int(c) for c in col))
        draw_strip.text((x + 20, y + 20), str(i + 1), fill="black", anchor="mm", font=FONT)

    return final


def main():
    p = argparse.ArgumentParser(prog="colornumber")
    p.add_argument("image")
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    p.add_argument("--simplify", choices=["low", "medium", "high"], default="medium")
    p.add_argument("--detail", choices=["low", "medium", "high"], default="medium")  # garde pour compatibilité
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--output", default="sheet.png")
    args = p.parse_args()

    img = Image.open(args.image).convert("RGBA")
    if args.width > 0 and img.width > args.width:
        ratio = args.width / img.width
        img = img.resize((args.width, int(img.height * ratio)), Image.Resampling.LANCZOS)

    k = {"easy": 8, "medium": 16, "hard": 24}[args.difficulty]
    sheet = generate_number_sheet(img, k, args.simplify, args.detail)
    sheet.save(args.output, dpi=(300, 300))
    print("Saved", args.output)


if __name__ == "__main__":
    main()
