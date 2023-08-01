import numpy as np
from typing import List
import matplotlib.pyplot as plt
import math


def display_images(img_list: List[np.array], titles=None, cols=2) -> None:
    if titles is not None and len(titles) != len(img_list):
        titles = None

    num_images = len(img_list)
    num_rows = math.ceil(num_images / cols)

    for i in range(len(img_list)):
        image = img_list[i]
        plt.subplot(num_rows, cols, i + 1)

        if titles is not None:
            plt.title(titles[i])

        plt.imshow(image)

    plt.show()


def save_images(save_path, img_list: List[np.array], titles=None, cols=2) -> None:
    if titles is not None and len(titles) != len(img_list):
        titles = None

    num_images = len(img_list)
    num_rows = math.ceil(num_images / cols)

    for i in range(len(img_list)):
        image = img_list[i]
        plt.subplot(num_rows, cols, i + 1)

        if titles is not None:
            plt.title(titles[i])

        plt.imshow(image)
        plt.axis('off')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=400)

