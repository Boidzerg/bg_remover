'''
Following the background image removal tutorial at: https://flothesof.github.io/removing-background-scikit-image.html
'''
# import necessary libraries
from skimage import filters, io as skio, morphology
from scipy import ndimage as ndi
from PIL import Image
import matplotlib.pyplot as plt, numpy as np

# load our source image
img = skio.imread('./images/tutorial_girl.jpg')

# apply sobel filters to find image outlines
sobel = filters.sobel(img)

# change the colors to black and gray
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.dpi'] = 200

# process the image
plt.imshow(sobel)

# save the output to our output directory
plt.savefig('./output/outlines.png')

# blur the image to make the lines thicker
blurred = filters.gaussian(sobel, sigma=2.0)
plt.imshow(blurred)

# save the blurred output to our output directory
plt.savefig('./output/outlines_blurred.png')

# begin to seperate back and foreground by identifying light spots
light_spots = np.array((img > 245).nonzero()).T
light_spots.shape

# output?
# (1432, 2)

# build new image with title and locations of lightspots
plt.plot(light_spots[:, 1], light_spots[:, 0], 'o')
plt.imshow(img)
plt.title('light spots in image')

# save the lightspots output to our output directory
plt.savefig('./output/outlines_lightspots.png')

# continue to seperate back and foreground by identifying dark spots
dark_spots = np.array((img < 3).nonzero()).T
dark_spots.shape

# output?
# (1402, 2)

# build new image with title and locations of darkspots
plt.plot(dark_spots[:, 1], dark_spots[:, 0], 'o')
plt.imshow(img)
plt.title('dark spots in image')

# save the darkspots output to our output directory
plt.savefig('./output/outlines_darkspots.png')

# making a seed mask
bool_mask = np.zeros(img.shape, dtype=np.bool)
bool_mask[tuple(light_spots.T)] = True
bool_mask[tuple(dark_spots.T)] = True
seed_mask, num_seeds = ndi.label(bool_mask)
num_seeds

# output?
# 672

# apply the watershed
ws = morphology.watershed(blurred, seed_mask)
plt.imshow(ws)

# save the watershed output to our output directory
plt.savefig('./output/outlines_watershed.png')