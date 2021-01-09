# usage: `python ~/Desktop/hough_by_hand.py 1994-654-12_v02.tif`
# output is a plot that pops up

import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import math
import pdb

# Loading image
if len(sys.argv) == 2:
  filename = sys.argv[1]
else:
  print("No input image given")

img_orig = cv2.imread("a.jpeg",)
img_orig = cv2.resize(img_orig, (50, 50))
img = img_orig[:,:,::-1] # color channel plotting mess http://stackoverflow.com/a/15074748/2256243

# http://nabinsharma.wordpress.com/2012/12/26/linear-hough-transform-using-python/
def hough_transform(img_bin, theta_res=1, rho_res=1):
  nR,nC = img_bin.shape
  theta = np.arange(0, 180.0, 45.0)

  D = np.sqrt((nR - 1)**2 + (nC - 1)**2)
  q = np.ceil(D/rho_res)
  nrho = 2*q + 1
  rho = np.linspace(-q*rho_res, q*rho_res, nrho)
  H = np.zeros((len(rho), len(theta)))
  for rowIdx in range(nR):
    for colIdx in range(nC):
      if img_bin[rowIdx, colIdx]:
        for thIdx in range(len(theta)):
          rhoVal = colIdx*np.cos(theta[thIdx]*np.pi/180.0) + \
              rowIdx*np.sin(theta[thIdx]*np.pi/180)
          rhoIdx = np.nonzero(np.abs(rho-rhoVal) == np.min(np.abs(rho-rhoVal)))[0]
          H[rhoIdx[0], thIdx] += 1
  return rho, theta, H


bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(bw, threshold1 = 0, threshold2 = 50, apertureSize = 3)

map = [
    [1, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 1],
]
map = np.array(map)


rhos, thetas, H = hough_transform(map)

def show(img_name, img, waitKey=0):
    cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
    cv2.imshow(img_name, img)
    cv2.waitKey(waitKey)
accumulator = H
accumulator = accumulator.astype(np.float32)
accumulator = accumulator / accumulator.max()
show('accumulator', accumulator)

rho_theta_pairs, x_y_pairs = top_n_rho_theta_pairs(H, 22, rhos, thetas)
im_w_lines = img.copy()
draw_rho_theta_pairs(im_w_lines, rho_theta_pairs)

# also going to draw circles in the accumulator matrix
for i in range(0, len(x_y_pairs), 1):
  x, y = x_y_pairs[i]
  cv2.circle(img = H, center = (x, y), radius = 12, color=(0,0,0), thickness = 1)

plt.subplot(141),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(edges,cmap = 'gray')
plt.title('Image Edges'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(H)
plt.title('Hough Transform Accumulator'), plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(im_w_lines)
plt.title('Detected Lines'), plt.xticks([]), plt.yticks([])

plt.show()