# core_simple.py
"""
ColorNumber – générateur de “coloriage à numéros”

• Contours noirs sur fond blanc (skeleton 1 px + épaisseur réglée)
• Numéros dans chaque zone (sauf blanc/fond), police réduite si mini-zone
• Palette lisible : texte blanc sur couleurs sombres, noir sinon
• Fond (et vrais blancs) exclus de la palette
• CLI : --difficulty (8/16/24 couleurs)  --simplify (low/medium/high)  --width
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

# ============================================================================
# police
FONT = ImageFont.load_default()
try:
    SMALL_FONT = ImageFont.truetype(FONT.path, 6)
except Exception:
    SMALL_FONT = FONT

# simplification preset
SIMPLIFY_PRESET = {
    "low":    dict(blur=1,  min_area=300,  contour_iter=0),
    "medium": dict(blur=2,  min_area=600,  contour_iter=1),
    "high":   dict(blur=3,  min_area=1200, contour_iter=2),
}

# ============================================================================
# 1) quantisation + fusion de micro-zones
def quantize(img: np.ndarray, k: int, min_area: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(
        img.reshape(-1, 3).astype(np.float32) / 255
    )
    labels = km.labels_.reshape(h, w)

    # fuse petites taches
    lab = label(labels, connectivity=1)
    for r in regionprops(lab):
        if r.area < min_area:
            y, x = r.coords[0]
            neighbour = labels[max(0, y - 1), x]
            labels[lab == r.label] = neighbour

    palette = (km.cluster_centers_ * 255).astype(np.uint8)
    return labels, palette


# ============================================================================
# 2) planche complète
def generate_number_sheet(
    image: Image.Image,
    k: int = 16,
    simplify: str = "medium",
) -> Image.Image:
    w, h = image.size
    cfg = SIMPLIFY_PRESET[simplify]

    # ---------- pré-flou ----------
    np_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)
    if cfg["blur"]:
        np_img = cv2.GaussianBlur(np_img, (0, 0), cfg["blur"])

    # ---------- palette + labels ----------
    labels, palette = quantize(np_img, k, cfg["min_area"])
    unique = np.unique(labels)

    # ---------- whitelist/skip : blanc & fond ----------
    skip_labels = set()
    for lbl in unique:
        r, g, b = palette[lbl]
        if r > 240 and g > 240 and b > 240:        # near-white
            skip_labels.add(lbl)

    surface = h * w
    lab_full = label(labels, connectivity=1)
    for r in regionprops(lab_full):
        if r.area / surface >= 0.10:               # ≥10 % surface & touche bord
            if r.bbox[0] == 0 or r.bbox[1] == 0 or r.bbox[2] == h or r.bbox[3] == w:
                lbl = labels[r.coords[0][0], r.coords[0][1]]
                skip_labels.add(lbl)

    usable = [lbl for lbl in unique if lbl not in skip_labels]
    mapping = {lbl: idx + 1 for idx, lbl in enumerate(usable)}
    palette_ordered = [palette[lbl] for lbl in usable]

    # ---------- contours ----------
    border = skeletonize(find_boundaries(labels, mode="thick")).astype(np.uint8) * 255
    if cfg["contour_iter"]:
        border = cv2.dilate(border, np.ones((3, 3), np.uint8), iterations=cfg["contour_iter"])

    sheet = Image.new("RGB", (w, h), "white")
    sheet.paste("black", mask=Image.fromarray(border).convert("L"))
    draw = ImageDraw.Draw(sheet)
    occ = np.zeros((h, w), dtype=bool)            # évite chevauchement

    # ---------- placement chiffres ----------
    def place_number(mask: np.ndarray, num: str, font: ImageFont.ImageFont):
        dist = distance_transform_edt(mask & (border == 0))
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        if dist[y, x] < 3:
            return
        if not occ[y, x]:
            draw.text((x, y), num, fill="black", anchor="mm", font=font)
            occ[max(0, y - 3): y + 4, max(0, x - 3): x + 4] = True

    lab_conn = label(labels, connectivity=1)
    for region in regionprops(lab_conn):
        lbl = labels[region.coords[0][0], region.coords[0][1]]
        if lbl in skip_labels:
            continue
        area = region.area
        if area < 50:
            continue
        mask_region = lab_conn == region.label
        font = FONT if area >= 100 else SMALL_FONT
        place_number(mask_region, str(mapping[lbl]), font)

    # ---------- palette ----------
    strip_h = 60
    final = Image.new("RGB", (w, h + strip_h), "white")
    final.paste(sheet, (0, 0))
    draw_p = ImageDraw.Draw(final)
    spacing = max(50, w // max(8, len(usable)))
    for i, col in enumerate(palette_ordered):
        x = 10 + i * spacing
        y = h + 10
        rgb = tuple(int(c) for c in col)
        draw_p.rectangle([x, y, x + 40, y + 40], fill=rgb)

        # contraste : blanc sur sombre, noir sur clair
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        txt_color = "white" if lum < 150 else "black"
        draw_p.text((x + 20, y + 20), str(i + 1), fill=txt_color, anchor="mm", font=FONT)

    return final


# ============================================================================
# 3) CLI
def main():
    p = argparse.ArgumentParser(prog="colornumber")
    p.add_argument("image")
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    p.add_argument("--simplify", choices=["low", "medium", "high"], default="medium")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--output", default="sheet.png")
    args = p.parse_args()

    diff_map = {"easy": 8, "medium": 16, "hard": 24}
    img = Image.open(args.image).convert("RGBA")
    if args.width > 0 and img.width > args.width:
        ratio = args.width / img.width
        img = img.resize((args.width, int(img.height * ratio)), Image.Resampling.LANCZOS)

    sheet = generate_number_sheet(img, diff_map[args.difficulty], args.simplify)
    sheet.save(args.output, dpi=(300, 300))
    print(f"✅  Saved {args.output}")


if __name__ == "__main__":
    main()
