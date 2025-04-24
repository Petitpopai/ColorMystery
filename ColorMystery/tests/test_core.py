import numpy as np
from PIL import Image
from core import quantize, contours_from_labels, place_ids

def test_quantize():
    img = np.zeros((10,10,3), dtype=np.uint8)
    labels, palette = quantize(img, 2)
    assert labels.shape == (10,10)
    assert palette.shape == (2,3)

def test_contours():
    labels = np.zeros((5,5), dtype=int)
    labels[:,3:] = 1
    border = contours_from_labels(labels, 'medium')
    assert border.sum() > 0

def test_place_ids():
    labels = np.tile(np.array([[0,1],[2,3]]), (2,2))
    ids = place_ids(labels, ['A','B','C','D'])
    assert len(ids) == 4
