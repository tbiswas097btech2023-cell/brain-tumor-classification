"""Detect near-duplicate images (incl. flipped copies) shared between Training and Testing.

Usage: python check_leakage.py "C:\\path\\to\\dataset"      (needs Training/ and Testing/ sub-folders)
Uses only Pillow + NumPy. Reports how many Testing images have a near-identical twin in Training.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def dhash(img):
    """64-bit difference hash of a PIL image -> uint8 bit array of length 64."""
    g = ImageOps.exif_transpose(img).convert("L").resize((9, 8), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def hashes(folder):
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS)
    H, HF = [], []
    for p in files:
        with Image.open(p) as im:
            H.append(dhash(im))
            HF.append(dhash(ImageOps.mirror(im)))     # horizontally flipped version
    return files, np.array(H), np.array(HF)


def main(root):
    root = Path(root)
    tr_files, tr, _ = hashes(root / "Training")
    te_files, te, te_flip = hashes(root / "Testing")
    print(f"Training images: {len(tr)}   Testing images: {len(te)}")
    # Hamming distance matrix (test x train)
    d = (te[:, None, :] != tr[None, :, :]).sum(2)
    df = (te_flip[:, None, :] != tr[None, :, :]).sum(2)
    best = np.minimum(d.min(1), df.min(1))
    for thr in (0, 4, 8):
        n = int((best <= thr).sum())
        print(f"Testing images with a near-identical Training twin (distance <= {thr}): {n}  ({100*n/len(te):.1f}%)")
    pct = 100 * (best <= 8).sum() / len(te)
    print("\nVERDICT:", "LEAKAGE LIKELY - accuracy would be inflated." if pct > 5 else
          "Little overlap found - split looks reasonably clean.")


if __name__ == "__main__":
    main(sys.argv[1])
