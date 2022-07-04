import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import os
from matplotlib import pyplot as plt
import scipy as sp
from scipy.ndimage.interpolation import shift
from scipy import ndimage


TRAIN_PATH = "data/train/"
VAl_PATH = "data/val/"

DIGITS = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]

PICSIZE = (64, 64)


def load_digit_imgs(digit: str, train=True) -> list:
    """
    Loads the images of a digit
    :param digit: the digit to load its images
    :param train: load from train folder or validation folder
    :return: list of the digit images
    """
    path = TRAIN_PATH if train else VAl_PATH
    path = path + digit

    images = []
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f):
            image = Image.open(f)
            newsize = PICSIZE
            image = image.resize(newsize)
            image = ImageOps.invert(image)
            image = np.asarray(image)
            images.append(image)

    return images


def plot_imgs(imgs: list, in_row=10, title=None, figsize=(10, 10), picsize=PICSIZE, inv=True, factor=0):
    """
    Plot the images in a matplotlib graph
    :param imgs: the images to plot
    :param in_row: how many images to plot in a row
    :param title: title of graph
    :param figsize: size of figure
    :param picsize: picsize
    :param inv: if to inverse the image
    :param factor: if to factor the image
    """
    n = len(imgs)
    in_row = n if in_row >= n else in_row
    empty = in_row - n % in_row  if n % in_row > 0 else 0
    imgs = imgs + [np.zeros(picsize).astype(np.uint8)]*empty
    n = len(imgs)
    rows = n // in_row
    fig, axs = plt.subplots(rows, in_row, figsize=figsize)

    for i in range(rows):
        for j in range(in_row):
            img = imgs[i * in_row + j] if rows > 1 else imgs[j]
            img = img.reshape(picsize)
            img = Image.fromarray(img)
            img = img.convert("L")
            if inv:
                img = ImageOps.invert(img)
            if factor > 0:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(factor)
            axsi = axs
            if rows > 1:
                axsi = axsi[i]
            if in_row > 1:
                axsi = axsi[j]
            axsi.imshow(img, cmap="gray", vmin=0, vmax=255)
            axsi.get_xaxis().set_visible(False)
            axsi.get_yaxis().set_visible(False)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# Rotate image at random angle
def rotate_random(img):
    angle = 0  # Random angle between -20 and 20 degrees
    while abs(angle) < 8:
        angle = (np.random.rand(1) * 40 - 20)[0]
    return sp.ndimage.rotate(img, angle, reshape=False)


# Shift image in a random direction
def shift_random(img):
    dir = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    direction = dir[np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])]
    size = int(PICSIZE[0] * 10/100)
    move = [direction[0]*size, direction[1]*size]
    return shift(img, move, cval=0, mode="constant")


# Zoom image randomly in and out
def zoom(img):
    height, width = img.shape
    random_z = np.random.uniform(0.6, 3)  # Currently Does Zoom
    transform = [[random_z, 0, 0],
                 [0, random_z, 0],
                 [0, 0, 1]]
    zoomed = ndimage.affine_transform(img, transform, output_shape=(height, width))
    return zoomed


# Noise image
def add_noise(img):
    noise = np.random.normal(0, 0.05, img.shape)
    # print(noise.max(), noise.min())
    return img + 255*noise


# Reduce image
def multi_noise(img):
    multi = np.random.uniform(0, 2, img.shape)
    return img*multi


def show_random_image_results():
    random_digit = np.random.choice(DIGITS, 1)[0]

    digit_imgs = load_digit_imgs(random_digit)

    random_img = np.random.randint(0, len(digit_imgs))

    img = digit_imgs[random_img]

    to_plot = [img, rotate_random(img), shift_random(img), zoom(img), add_noise(img), multi_noise(img)]
    # plot_imgs(to_plot, in_row=5, title=f"Random image from digit {random_digit}")
    plot_imgs(to_plot, in_row=3)


def make_empty_folders():
    # Make new data folders
    path = "data_new"
    os.mkdir(path)
    os.mkdir(path + "/train")
    os.mkdir(path + "/val")
    for digit in ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]:
        for part in ["train", "val"]:
            os.mkdir(path + f"/{part}/" + digit)


def make_new_data():

    filter_funcs = [("org", lambda x: x), ("rotate", rotate_random), ("shift", shift_random), ("zoom", zoom), ("noise1", add_noise),
                    ("noise2", multi_noise)]

    make_empty_folders()

    count = 0

    # Create new train and validation folders from train
    for digit in DIGITS:
        print("CURRENT DIGIT:", digit)
        digit_imgs = load_digit_imgs(digit)

        for index, img in enumerate(digit_imgs):
            p1 = np.random.uniform(0, 1)
            if p1 < 0.25:
                for filter_name, filter_func in filter_funcs:
                    tmp_img = filter_func(img)
                    tmp_img = Image.fromarray(tmp_img)
                    tmp_img = tmp_img.convert("L")
                    tmp_img = ImageOps.invert(tmp_img)
                    newsize = PICSIZE
                    tmp_img = tmp_img.resize(newsize)
                    tmp_img.save("data_new/val/" + digit +
                                 f"/{index}_{filter_name}.png")
                    count += 1
            else:
                for filter_name, filter_func in filter_funcs:
                    pf = np.random.uniform(0, 1)
                    if pf < 0.75:
                        tmp_img = filter_func(img)
                        tmp_img = Image.fromarray(tmp_img)
                        tmp_img = tmp_img.convert("L")
                        tmp_img = ImageOps.invert(tmp_img)
                        newsize = PICSIZE
                        tmp_img = tmp_img.resize(newsize)
                        tmp_img.save("data_new/train/" + digit +
                                     f"/{index}_{filter_name}.png")
                        count += 1
                    else:
                        continue

    print("DONE")
    print("New Data Size = ", count)


if __name__ == "__main__":
    show_random_image_results()
    make_new_data()
