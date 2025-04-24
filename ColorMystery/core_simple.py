# core_simple.py
"""
ColorNumber – générateur « coloriage à numéros » tout-en-un
• Contours noirs sur fond blanc
• Numéros placés sans chevauchement ni superposition de trait
• Palette de couleurs en bas de page
• Paramètres : nombre de couleurs (--difficulty) et simplification (--simplify)

Auteur : ColorCode – MIT License
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, List
from scipy.ndimage import distance_transform_edt


import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from skimage.segmentation import find_boundaries
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

FONT = ImageFont.load_default()

# ──────────────────────────────────────────────────────────────────────────────
#  Paramètres de simplification
#    • blur        : flou gaussien pré-traitement (px)
#    • min_area    : zone mini conservée (px²)  → fusionne les taches
#    • contour_iter: épaississement facultatif (dilate) après skeleton
SIMPLIFY_PRESET = {
    "low":    dict(blur=1,  min_area=300,  contour_iter=0),
    "medium": dict(blur=2,  min_area=600,  contour_iter=1),
    "high":   dict(blur=3,  min_area=1200, contour_iter=2),
}


# ──────────────────────────────────────────────────────────────────────────────
#  1) Quantisation + fusion des micro-zones
def quantize(img: np.ndarray, k: int, min_area: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    flat = img.reshape(-1, 3).astype(np.float32) / 255
    km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(flat)
    labels = km.labels_.reshape(h, w)

    # Fusionne les régions trop petites
    lab = label(labels, connectivity=1)
    for r in regionprops(lab):
        if r.area < min_area:
            y, x = r.coords[0]
            neighbour = labels[max(0, y - 1), x]
            labels[lab == r.label] = neighbour

    palette = (km.cluster_centers_ * 255).astype(np.uint8)  # RGB de 0-255
    return labels, palette


# ──────────────────────────────────────────────────────────────────────────────
#  2) Génération complète d’une planche
def generate_number_sheet(
    image: Image.Image,
    k: int = 16,
    simplify: str = "medium",
) -> Image.Image:
    """
    Retourne l’image finale (contours + numéros + palette).

    Args:
        image: PIL.Image RGBA ou RGB
        k:     nombre de couleurs (8, 16, 24…)
        simplify: 'low' (détaillé) · 'medium' · 'high' (très épuré)
    """
    w, h = image.size
    cfg = SIMPLIFY_PRESET[simplify]

    # -- flou optionnel pour réduire le bruit couleur
    np_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)
    if cfg["blur"]:
        np_img = cv2.GaussianBlur(np_img, (0, 0), cfg["blur"])

    # -- palette + labels nettoyés
    labels, palette = quantize(np_img, k, cfg["min_area"])
    unique = np.unique(labels)
    mapping = {lbl: idx + 1 for idx, lbl in enumerate(unique)}
    palette_ordered = [palette[lbl] for lbl in unique]

    # -- contours fins via skeleton
    border = find_boundaries(labels, mode="thick")           # booléen
    border = skeletonize(border).astype(np.uint8) * 255      # 1 px
    if cfg["contour_iter"]:
        border = cv2.dilate(border, np.ones((3, 3), np.uint8), iterations=cfg["contour_iter"])

    sheet = Image.new("RGB", (w, h), "white")
    sheet.paste("black", mask=Image.fromarray(border).convert("L"))
    draw = ImageDraw.Draw(sheet)

    # -- masque d’occupation pour éviter chevauchements
    occ = np.zeros((h, w), dtype=bool)

    def place_number(region_mask: np.ndarray, num: str):
        ys, xs = np.where(region_mask)
        if xs.size == 0:
            return
        cx, cy = int(xs.mean()), int(ys.mean())
        # spirale jusqu'à 15 px autour du centre
        for r in range(0, 15):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    x, y = cx + dx, cy + dy
                    if 0 <= x < w and 0 <= y < h:
                        if border[y, x] == 0 and not occ[y, x]:
                            draw.text((x, y), num, fill="black", anchor="mm", font=FONT)
                            occ[max(0, y - 3): y + 4, max(0, x - 3): x + 4] = True
                            return

    # — placer un numéro dans CHAQUE région de la même couleur — #
    lab_conn = label(labels, connectivity=1)
    for region in regionprops(lab_conn):
        lbl = labels[region.coords[0][0], region.coords[0][1]]  # couleur de la région
        if region.area < 200:      # zone trop petite pour un chiffre lisible
            continue
        mask_region = lab_conn == region.label
        place_number(mask_region, str(mapping[lbl]))


    # -- palette sous l'image
    strip_h = 60
    final = Image.new("RGB", (w, h + strip_h), "white")
    final.paste(sheet, (0, 0))
    draw_p = ImageDraw.Draw(final)
    spacing = max(50, w // max(8, len(unique)))
    for i, col in enumerate(palette_ordered):
        x = 10 + i * spacing
        y = h + 10
        draw_p.rectangle([x, y, x + 40, y + 40], fill=tuple(int(c) for c in col))
        draw_p.text((x + 20, y + 20), str(i + 1), fill="black", anchor="mm", font=FONT)

    return final


# ──────────────────────────────────────────────────────────────────────────────
#  3) Ligne de commande
def main():
    parser = argparse.ArgumentParser(prog="colornumber")
    parser.add_argument("image", help="Image d'entrée (PNG/JPG)")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium",
                        help="easy=8 couleurs, medium=16, hard=24")
    parser.add_argument("--simplify", choices=["low", "medium", "high"], default="medium",
                        help="Niveau de simplification des traits")
    parser.add_argument("--width", type=int, default=1024,
                        help="Largeur max redimensionnement (0 = garde taille)")
    parser.add_argument("--output", default="sheet.png", help="Nom du fichier PNG de sortie")
    args = parser.parse_args()

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
