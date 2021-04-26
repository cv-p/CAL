import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.exposure import histogram

coins = data.coins()


from skimage.feature import canny

edges = canny(coins)

# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(edges, cmap=plt.cm.gray)
# ax.set_title('Canny detector')
# ax.axis('off')

from scipy import ndimage as ndi

fill_coins = ndi.binary_fill_holes(edges)

# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(fill_coins, cmap=plt.cm.gray)
# ax.set_title('filling the holes')
# ax.axis('off')

from skimage import morphology

coins_cleaned = morphology.remove_small_objects(fill_coins, 21)

# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(coins_cleaned, cmap=plt.cm.gray)
# ax.set_title('removing small objects')
# ax.axis('off')

from skimage.filters import sobel

elevation_map = sobel(coins)

# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(elevation_map, cmap=plt.cm.gray)
# ax.set_title('elevation map')
# ax.axis('off')


markers = np.zeros_like(coins)
markers[coins < 30] = 1
markers[coins > 150] = 2

# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(markers, cmap=plt.cm.nipy_spectral)
# ax.set_title('markers')
# ax.axis('off')

from skimage import segmentation

segmentation_coins = segmentation.watershed(elevation_map, markers)

# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(segmentation_coins, cmap=plt.cm.gray)
# ax.set_title('segmentation')
# ax.axis('off')

from skimage.color import label2rgb

segmentation_coins = ndi.binary_fill_holes(segmentation_coins - 1)
# labeled_coins, _ = ndi.label(segmentation_coins)
# image_label_overlay = label2rgb(labeled_coins, image=coins, bg_label=0)
for y in segmentation_coins:
    for x in segmentation_coins[y]:
        print(x)

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
axes[0].imshow(coins, cmap=plt.cm.gray)
axes[0].contour(segmentation_coins, [0.5], linewidths=1.2, colors='y')
# axes[1].imshow(image_label_overlay)

for a in axes:
    a.axis('off')

plt.tight_layout()

plt.show()
