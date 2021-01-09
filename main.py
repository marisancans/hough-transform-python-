import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def hough_line(img, thetas):
  # Rho and Theta ranges
  width, height = img.shape
  diag_len = np.ceil(np.sqrt(width * width + height * height))   # max_dist
  rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

  # Cache some resuable values
  cos_t = np.cos(thetas)
  sin_t = np.sin(thetas)
  num_thetas = len(thetas)

  # Hough accumulator array of theta vs rho
  accumulator = np.zeros((int(2 * diag_len), num_thetas), dtype=np.uint64)

  img_inverted = img.copy()
  img_inverted += 1
  y_idxs, x_idxs = np.nonzero(img_inverted)  # (row, col) indexes to edges

  # Vote in the hough accumulator
  for i in range(len(x_idxs)):
    x = x_idxs[i]
    y = y_idxs[i]

    for t_idx in range(num_thetas):
      # Calculate rho. diag_len is added for a positive index
      rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
      accumulator[int(rho), t_idx] += 1

  return accumulator, rhos


def show(img_name, img, waitKey=0):
    cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
    cv2.imshow(img_name, img)
    cv2.waitKey(waitKey)

def get_spectre(map, thetas):
    accumulator, rhos = hough_line(map, thetas)

    spectre = np.sum(accumulator ** 2, axis=0)
    spectre = spectre / spectre.max()

    return spectre, accumulator

def translate_X(map, position):
    if position == 0:
        return map

    if position > 0:
        direction = 1
    else:
        direction = -1

    for i in range(abs(position)):
        map = np.roll(map, direction, axis=1)

        # dumb way to roll without warping
        if direction == 1: # determine which way 
            map[:, 0] = 1.0
        else:
            map[:, -1] = 1.0

    return map


def translate_Y(map, position):
    if position == 0:
        return map

    if position > 0:
        direction = 1
    else:
        direction = -1

    for i in range(abs(position)):
        map = np.roll(map, direction, axis=0)

        # dumb way to roll without warping
        if direction == 1: # determine which way 
            map[0, :] = 1.0
        else:
            map[-1, :] = 1.0
    return map


def copy_map(map):
    map_copy = map.copy()
    map_copy = map_copy.astype(np.float32) / 255.0
    return map_copy

def calc_correlation(series_1, series_2):
    cors = []
    sp_cpy = series_2.copy()
    for i in range(len(series_1)):
        x = series_1 * sp_cpy
        c = np.sum(x)
        sp_cpy = np.roll(sp_cpy, -1, axis=0)

        cors.append(c)
    return np.array(cors)


def calc_correlation_for_translation(series_1, series_2, position):
    # position = 2
    # series_1 = np.array([6,10,5,4])
    # series_2 = np.array([10,3,2,4])
    sp_cpy = series_2.copy()

    if position == 0:
        x = series_1 * sp_cpy
        c = np.sum(x)
        return c

    if position > 0:
        direction = 1
    else:
        direction = -1

    for i in range(abs(position)):
        sp_cpy = np.roll(sp_cpy, direction, axis=0)

        # dumb way to roll without warping
        if direction == 1: # determine which way 
            sp_cpy[0] = 0
        else:
            sp_cpy[-1] = 0

    x = series_1 * sp_cpy
    c = np.sum(x)

    return c


def calc_translate(map_1, map_2, pos):
    spectre_1_X = np.sum(map_1, axis=0)
    spectre_2_X = np.sum(map_2, axis=0)

    X = calc_correlation_for_translation(spectre_1_X, spectre_2_X, pos)
    X = np.sum(X)

    spectre_1_Y = np.sum(map_1, axis=1)
    spectre_2_Y = np.sum(map_2, axis=1)

    Y = calc_correlation_for_translation(spectre_1_Y, spectre_2_Y, pos)
    Y = np.sum(Y)

    return X, Y

def translate_maps(map_1, map_2_rotated):
    map_1_copy = map_1.copy()   
    map_1_copy = map_1_copy.astype(np.float32) / 255.0


    tvx = []
    tvy = []
    # Try to translate the map with -10 and +10 directions

    # show("map_1_copy", map_1_copy, 1)

    for x in range(-10, 10, 1):
        map_2_rotated_copy = copy_map(map_2_rotated)
        map_2_rotated_copy = translate_X(map_2_rotated_copy, x)
        # show("map_2_rotated_copy", map_2_rotated_copy, 0)
        X, Y = calc_translate(map_1_copy, map_2_rotated_copy, x)
        tvx.append(X) # translated values x


    for y in range(-10, 10, 1):
        map_2_rotated_copy = copy_map(map_2_rotated)
        map_2_rotated_copy = translate_Y(map_2_rotated_copy, y)
        # show("map_2_rotated_copy", map_2_rotated_copy, 0)
        X, Y = calc_translate(map_1_copy, map_2_rotated_copy, y)
        tvy.append(Y)

    return tvx, tvy





# Generate tethas with a step of one 
thetas = np.deg2rad(np.arange(0, 180.0, 1.0))


# Map one and rotated
map_1 = cv2.imread('map_1.bmp', cv2.IMREAD_GRAYSCALE)
# show('map_1', map_1, 1)
spectre_1, accumulator_1 = get_spectre(map_1, thetas)

# Map two
map_2 = cv2.imread('map_2.bmp', cv2.IMREAD_GRAYSCALE)
spectre_2, accumulator_2 = get_spectre(map_2, thetas)
# show('map_2', map_2, 1)

diff = cv2.absdiff(accumulator_1 / accumulator_1.max(), accumulator_2 / accumulator_2.max())
# show('diff', diff, 0)
# show('spectre 1', accumulator_1 / accumulator_1.max(), 0)
# show('spectre 2', accumulator_2 / accumulator_2.max(), 0)


# Debug puroses
# spectre_1 = np.array([1, 0.7, 0.5, 0.2, 1, 0.7, 0.5, 0.2])
# spectre_2 = np.array([0.5, 1, 0.6, 0.8, 0.5, 1, 0.6, 0.8])


cors_np = calc_correlation(spectre_1, spectre_2)

min_v = min(np.min(spectre_2), np.min(spectre_1))
max_v = max(np.max(spectre_2), np.max(spectre_1))

scaler = MinMaxScaler(feature_range=(min_v, max_v))
cors_2d = np.expand_dims(cors_np, -1)
scaler.fit(cors_2d)
cors_scaled = scaler.transform(cors_2d)
cors_scaled = cors_scaled.flatten()

ymax = np.max(cors_scaled)
xpos = np.argmax(cors_scaled)
xmax = thetas[xpos]


degrees = np.rad2deg(xmax)
print(f'Max corelation degrees: {degrees}')

fig = plt.figure()
ax = fig.add_subplot(111)

# ax.plot(spectre_1)
# ax.plot(spectre_2)

# ax.plot(cors_scaled)
# ax.autoscale(enable=True) 
# plt.show()

# rotate the map 90 dagrees counter-clockwise
# This is because 90 degrees is what is the maximum correlation value
map_2_rotated = np.rot90(map_2)
# show("map_2_rotated", map_2_rotated, 0)

# Map translation

# move map 10 steps on X to right and Y down
tvx, tvy = translate_maps(map_1, map_2_rotated) 

stepx = np.arange(len(tvx)) - 10

tvx_np = np.array(tvx)
tvy_np = np.array(tvy)

tvx_np = tvx_np / np.max(tvx_np)
tvy_np = tvy_np / np.max(tvy_np)

ax.plot(stepx, tvx_np)
ax.plot(stepx, tvy_np)
plt.show()

# ax.plot(stepx, tvy_map_1)
# ax.plot(stepx, tvy_map_2)
# plt.show()

# https://www.mathsisfun.com/algebra/line-parallel-perpendicular.html

# https://www.varsitytutors.com/algebra_1-help/how-to-find-out-if-a-point-is-on-a-line-with-an-equation
# https://stackoverflow.com/questions/18059793/to-determine-if-a-point-lies-on-a-line

x = 0








