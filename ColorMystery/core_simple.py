# core_simple.py
"""
ColorNumber – colour-by-number ultra-rapide
• Contours noirs (find_boundaries + dilate)
• Numéros au centre de chaque région (barycentre + petit décalage)
• Palette sans blanc, texte contrasté
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

# Fonts
FONT = ImageFont.load_default()
try:
    SMALL_FONT = ImageFont.truetype(FONT.path, 6)
except Exception:
    SMALL_FONT = FONT

# Presets
SIMPLIFY_PRESET = {
    "low":    dict(blur=0, min_area=30,  dilate_iter=1),
    "medium": dict(blur=1, min_area=100, dilate_iter=2),
    "high":   dict(blur=2, min_area=200, dilate_iter=3),
}

def quantize(img: np.ndarray, k: int, min_area: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(img.reshape(-1,3)/255.0)
    labels = km.labels_.reshape(h, w)
    # fusion des petites taches
    lab = label(labels, connectivity=1)
    for r in regionprops(lab):
        if r.area < min_area:
            labels[lab == r.label] = labels[tuple(r.coords[0])]
    palette = (km.cluster_centers_ * 255).astype(np.uint8)
    return labels, palette

def generate_number_sheet(image: Image.Image, k: int = 16, simplify: str = "medium") -> Image.Image:
    w, h = image.size
    cfg = SIMPLIFY_PRESET[simplify]

    # 1) Pré-flou (optionnel)
    arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)
    if cfg["blur"]>0:
        arr = cv2.GaussianBlur(arr, (0,0), cfg["blur"])

    # 2) Quantize & palette
    labels, palette = quantize(arr, k, cfg["min_area"])
    unique = np.unique(labels)

    # 3) Skip blanc & fond
    skip = set()
    # blanc (R,G,B>230)
    for lbl in unique:
        r,g,b = palette[lbl]
        if r>230 and g>230 and b>230:
            skip.add(lbl)
    # fond (grosse région au bord)
    surf = w*h
    lf = label(labels, connectivity=1)
    for r in regionprops(lf):
        if r.area/surf>0.1 and (0 in (r.bbox[0],r.bbox[1]) or r.bbox[2]==h or r.bbox[3]==w):
            skip.add(labels[tuple(r.coords[0])])

    usable = [lbl for lbl in unique if lbl not in skip]
    mapping = {lbl:i+1 for i,lbl in enumerate(usable)}
    palette_ordered = [palette[lbl] for lbl in usable]

    # 4) Contours
    edge = find_boundaries(labels, mode="thick").astype(np.uint8)*255
    kernel = np.ones((3,3),np.uint8)
    edge = cv2.dilate(edge, kernel, iterations=cfg["dilate_iter"])

    # 5) Créer la feuille
    sheet = Image.new("RGB",(w,h),"white")
    sheet.paste("black", mask=Image.fromarray(edge))
    draw = ImageDraw.Draw(sheet)
    occ = np.zeros((h,w),bool)

    # 6) Placement des numéros au barycentre (+ petit recul si sur le contour)
    lf = label(labels, connectivity=1)
    for r in regionprops(lf):
        lbl = labels[tuple(r.coords[0])]
        if lbl not in mapping or r.area< cfg["min_area"]:
            continue
        cy, cx = map(int, r.centroid)
        # si c’est un trait, reculer jusqu’à pixel blanc
        for dx in (0, -1,1,0):
            for dy in (0,0,-1,1):
                x,y = cx+dx, cy+dy
                if 0<=x<w and 0<=y<h and edge[y,x]==0:
                    cx,cy = x,y
                    break
        font = FONT if r.area>100 else SMALL_FONT
        if not occ[cy,cx]:
            draw.text((cx,cy), str(mapping[lbl]), fill="black", anchor="mm", font=font)
            occ[max(0,cy-3):cy+4, max(0,cx-3):cx+4]=True

    # 7) Palette
    strip_h=60
    final = Image.new("RGB",(w,h+strip_h),"white")
    final.paste(sheet,(0,0))
    dp = ImageDraw.Draw(final)
    spacing = max(50, w//max(1,len(usable)))
    for i,col in enumerate(palette_ordered):
        x0,y0 = 10+i*spacing, h+10
        rgb=tuple(int(c) for c in col)
        dp.rectangle([x0,y0,x0+40,y0+40], fill=rgb)
        lum=0.2126*rgb[0]+0.7152*rgb[1]+0.0722*rgb[2]
        tc="white" if lum<150 else "black"
        dp.text((x0+20,y0+20), str(i+1), fill=tc, anchor="mm", font=FONT)

    return final

def main():
    p=argparse.ArgumentParser(prog="colornumber")
    p.add_argument("image")
    p.add_argument("--difficulty",choices=["easy","medium","hard"],default="medium")
    p.add_argument("--simplify",choices=["low","medium","high"],default="medium")
    p.add_argument("--width",type=int,default=512)
    p.add_argument("--output",default="sheet.png")
    args=p.parse_args()

    dm={"easy":8,"medium":16,"hard":24}
    img=Image.open(args.image).convert("RGBA")
    if args.width>0 and img.width>args.width:
        r=args.width/img.width
        img=img.resize((args.width,int(img.height*r)),Image.Resampling.LANCZOS)

    sheet=generate_number_sheet(img, dm[args.difficulty], args.simplify)
    sheet.save(args.output,dpi=(300,300))
    print("Saved",args.output)

if __name__=="__main__":
    main()
