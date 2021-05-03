'''
Following the background image removal tutorial at: https://flothesof.github.io/removing-background-scikit-image.html
'''
# import necessary libraries
from skimage import filters, io as skio, morphology, segmentation, data
from skimage.color import rgb2gray
from scipy import ndimage as ndi
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt, numpy as np

import matplotlib.image as mpimg

# load our source image
# img = skio.imread('./images/buss.jpg')
img = skio.imread('./images/image.jpg', as_gray=True)
# img = skio.imread('./images/tutorial_girl.jpg')

print (img.shape)

# apply sobel filters to find image outlines
sobel = filters.sobel(img)

# change the colors to black and gray
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.dpi'] = 200



# process the image
plt.imshow(sobel)
# plt.show()

# save the output to our output directory
# plt.savefig('./output/outlines.png')

# blur the image to make the lines thicker
blurred = filters.gaussian(sobel, sigma=2.0, multichannel=False)
plt.imshow(blurred)

# save the blurred output to our output directory
# plt.savefig('./output/outlines_blurred.png')

# begin to seperate back and foreground by identifying light spots
light_spots = np.array((img > .9).nonzero()).T
light_spots.shape

# build new image with title and locations of lightspots
plt.plot(light_spots[:, 1], light_spots[:, 0], 'o')
plt.imshow(img)
plt.title('light spots in image')

# save the lightspots output to our output directory
# plt.savefig('./output/outlines_lightspots.png')
plt.show()

# continue to seperate back and foreground by identifying dark spots
dark_spots = np.array((img < .1).nonzero()).T
dark_spots.shape

# build new image with title and locations of darkspots
plt.plot(dark_spots[:, 1], dark_spots[:, 0], 'o')
plt.imshow(img)
plt.title('dark spots in image')

# save the darkspots output to our output directory
# plt.savefig('./output/outlines_darkspots.png')
plt.show()

# making a seed mask
bool_mask = np.zeros(img.shape, dtype=np.bool_)
bool_mask[tuple(light_spots.T)] = True
bool_mask[tuple(dark_spots.T)] = True
seed_mask, num_seeds = ndi.label(bool_mask)

# apply the watershed
# ws = morphology.watershed(blurred, seed_mask)
ws = segmentation.watershed(blurred, seed_mask)
plt.imshow(ws)

# save the watershed output to our output directory
# plt.savefig('./output/outlines_watershed.png')
plt.show()

# remove the class with the most pixels in the image
background = max(set(ws.ravel()), key=lambda g: np.sum(ws == g))
# print (background, set(ws.ravel()), ws.shape)
background_mask = (ws == background)
# print (background_mask.shape)

# show current state of plt
plt.imshow(~background_mask)
plt.show()

# how does it look??
cleaned = img * ~background_mask
plt.imshow(cleaned)
plt.show()
# plt.savefig('./output/bg_removed')

# what was removed
plt.imshow(cleaned, cmap='gray')
bms = background_mask.shape
print (bms)
plt.imshow(background_mask.reshape(background_mask.shape + (1,)) * np.array([1, 0, 0, 1]))
plt.show()

# reuse previous processing steps as a function
def draw_group_as_background(ax, group, watershed_result, original_image):
    "Draws a group from the watershed result as red background."
    background_mask = (watershed_result == group)
    cleaned = original_image * ~background_mask
    ax.imshow(cleaned, cmap='gray')
    ax.imshow(background_mask.reshape(background_mask.shape + (1,)) * np.array([1, 0, 0, 1]))

background_candidates = sorted(set(ws.ravel()), key=lambda g: np.sum(ws == g), reverse=True)

N = 3
fig, axes = plt.subplots(N, N, figsize=(6, 8))
for i in range(N*N):
    draw_group_as_background(axes.ravel()[i], background_candidates[i], ws, img)
plt.tight_layout()
plt.show()

# selecting background and foreground seeds automatically
seed_mask = np.zeros(img.shape, dtype=int)
seed_mask[0, 0] = 1 # background
seed_mask[186, 186] = 2 # foreground

# perform the watershed again
# ws = morphology.watershed(blurred, seed_mask)
ws = segmentation.watershed(blurred, seed_mask)
plt.imshow(ws)

fig, ax = plt.subplots()
draw_group_as_background(ax, 1, ws, img)

seed_mask = np.zeros(img.shape, dtype=int)
seed_mask[0, 0] = 1 # background
seed_mask[186, 186] = 2 # foreground
# seed_mask[1000, 150] = 2 # left arm

# ws = morphology.watershed(blurred, seed_mask)
ws = segmentation.watershed(blurred, seed_mask)
plt.imshow(ws)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
draw_group_as_background(ax[1], 1, ws, img)

plt.savefig('./output/bg_removed')