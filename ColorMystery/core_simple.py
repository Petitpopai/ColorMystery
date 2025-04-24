# core_simple.py
"""
ColorNumber – générateur de « coloriage à numéros »
• Contours noirs sur fond blanc (1 px skeleton), police mini pour petites zones
• Numéros 1…N (blanc/fond exclus), palette sous l’image (texte contrasté)
• CLI : --difficulty (8/16/24) --simplify (low/medium/high) --width --output
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

# ─── Fonts ─────────────────────────────────────────────────────────────
FONT = ImageFont.load_default()
try:
    SMALL_FONT = ImageFont.truetype(FONT.path, 6)
except Exception:
    SMALL_FONT = FONT

# ─── Simplification presets ────────────────────────────────────────────
SIMPLIFY_PRESET = {
    "low":    dict(blur=0, min_area=50,   contour_iter=0),
    "medium": dict(blur=2, min_area=600,  contour_iter=1),
    "high":   dict(blur=3, min_area=1200, contour_iter=2),
}

# ─── 1) Quantisation + suppression des micro-zones ────────────────────
def quantize(img: np.ndarray, k: int, min_area: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    flat = img.reshape(-1, 3).astype(np.float32) / 255
    km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(flat)
    labels = km.labels_.reshape(h, w)
    lab = label(labels, connectivity=1)
    for r in regionprops(lab):
        if r.area < min_area:
            y, x = r.coords[0]
            labels[lab == r.label] = labels[max(0, y-1), x]
    palette = (km.cluster_centers_ * 255).astype(np.uint8)
    return labels, palette

# ─── 2) Génération de la planche ───────────────────────────────────────
def generate_number_sheet(
    image: Image.Image,
    k: int = 16,
    simplify: str = "medium",
) -> Image.Image:
    w, h = image.size
    cfg = SIMPLIFY_PRESET[simplify]

    # --- pré-flou pour lisser le bruit si demandé
    np_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)
    if cfg["blur"] > 0:
        np_img = cv2.GaussianBlur(np_img, (0, 0), cfg["blur"])

    # --- quantisation et fusion
    labels, palette = quantize(np_img, k, cfg["min_area"])
    unique = np.unique(labels)

    # --- repérer les labels à ignorer (blanc = papier + fond global) ---
    skip = set()
    # 1) near-white
    for lbl in unique:
        r, g, b = palette[lbl]
        if r > 230 and g > 230 and b > 230:
            skip.add(lbl)
    # 2) grande région de fond (≥10% surface & touche un bord)
    surface = w * h
    lab_full = label(labels, connectivity=1)
    for r in regionprops(lab_full):
        if r.area / surface >= 0.10:
            if 0 in (r.bbox[0], r.bbox[1]) or r.bbox[2] == h or r.bbox[3] == w:
                skip.add(labels[r.coords[0][0], r.coords[0][1]])

    # --- construire mapping & palette sans ces labels ---
    usable = [lbl for lbl in unique if lbl not in skip]
    mapping = {lbl: i+1 for i, lbl in enumerate(usable)}
    palette_ordered = [palette[lbl] for lbl in usable]

   # contours vectoriels optimisés
    raw = (find_boundaries(labels, mode="thick") > 0).astype(np.uint8)
    # 1) contours externes, pas trop détaillés
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    outline = Image.new("L", (w, h), 0)
    draw_outline = ImageDraw.Draw(outline)
    for cnt in contours:
        # 2) on ignore les micro-contours
        if cv2.contourArea(cnt) < 50:
            continue
        # 3) epsilon proportionnel à la longueur pour un lissage adaptatif
        arc = cv2.arcLength(cnt, True)
        factor = {"low": 0.002, "medium": 0.005, "high": 0.01}[simplify]
        eps = factor * arc
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = [tuple(pt[0]) for pt in approx]
        if len(pts) > 1:
            draw_outline.line(pts, fill=255, width=1)

# 4) on récupère le masque final des traits
border = np.array(outline)


    # --- fond blanc + traits noirs ---
    sheet = Image.new("RGB", (w, h), "white")
    sheet.paste("black", mask=Image.fromarray(border).convert("L"))
    draw = ImageDraw.Draw(sheet)
    occ = np.zeros((h, w), dtype=bool)

    # --- placement des numéros au point le + éloigné des bords ---
    def place_number(mask, num, font):
        dist = distance_transform_edt(mask & (border == 0))
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        if dist[y, x] < 3 or occ[y, x]:
            return
        draw.text((x, y), num, fill="black", anchor="mm", font=font)
        occ[max(0,y-3):y+4, max(0,x-3):x+4] = True

    lab_conn = label(labels, connectivity=1)
    for region in regionprops(lab_conn):
        lbl = labels[region.coords[0][0], region.coords[0][1]]
        if lbl not in mapping:
            continue
        area = region.area
        if area < 30:
            continue
        mask_region = lab_conn == region.label
        font = FONT if area >= 100 else SMALL_FONT
        place_number(mask_region, str(mapping[lbl]), font)

    # --- palette en bas avec contraste automatique ---
    strip_h = 60
    final = Image.new("RGB", (w, h+strip_h), "white")
    final.paste(sheet, (0,0))
    dp = ImageDraw.Draw(final)
    spacing = max(50, w // max(1, len(usable)))
    for i, col in enumerate(palette_ordered):
        x0 = 10 + i*spacing
        y0 = h + 10
        rgb = tuple(int(c) for c in col)
        dp.rectangle([x0, y0, x0+40, y0+40], fill=rgb)
        lum = 0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]
        txt_color = "white" if lum < 150 else "black"
        dp.text((x0+20, y0+20), str(i+1), fill=txt_color, anchor="mm", font=FONT)

    return final

# ─── 3) CLI ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(prog="colornumber")
    p.add_argument("image", help="Fichier PNG/JPG d'entrée")
    p.add_argument("--difficulty", choices=["easy","medium","hard"], default="medium")
    p.add_argument("--simplify", choices=["low","medium","high"], default="medium")
    p.add_argument("--width", type=int, default=1024, help="Max largeur (0=pas redim.)")
    p.add_argument("--output", default="sheet.png", help="PNG résultat")
    args = p.parse_args()

    dm = {"easy":8,"medium":16,"hard":24}
    img = Image.open(args.image).convert("RGBA")
    if args.width>0 and img.width>args.width:
        r = args.width/img.width
        img = img.resize((args.width,int(img.height*r)), Image.Resampling.LANCZOS)

    sheet = generate_number_sheet(img, dm[args.difficulty], args.simplify)
    sheet.save(args.output, dpi=(300,300))
    print(f"✅ Saved {args.output}")

if __name__=="__main__":
    main()
