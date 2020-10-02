"""
A local neighborhood are groups of neighboring pixels.
These local neighborhoods have a height, a width and a stride, very
similar to kernels of convolutions in a neural network.
more information: https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
Adapt the method below that is supposed to take an image and optimizes this image, s.t. each pixel in every local
neighborhood of the image converges to the mean color of its respective neighborhood.
Only adapt the objective. Generate a docstring. Test your method.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import image
from tqdm import tqdm


def kernel_idx(img, row, column, w, h):
    img_w = img.shape[0]
    img_h = img.shape[1]
    range_w = (int((w - 1) / 2), int((w - 1) / 2) + 1)
    if (w - 1) % 2 != 0:
        range_w = (range_w[0] + 1, range_w[1])

    range_h = (int((h - 1) / 2), int((h - 1) / 2) + 1)
    if (h - 1) % 2 != 0:
        range_h = (range_h[0] + 1, range_h[1])

    row_idx = np.array(range(row - range_w[0], row + range_w[1]))
    column_idx = np.array(range(column - range_h[0], column + range_h[1]))

    # For those kernels that go over the edge we use a standard image processing
    # technique, where you "mirror" the image and "place" it next to the edge.
    # So if our image is:
    # [ [1, 2, 3, 4],
    #   [1, 2, 3, 4],
    #   [1, 2, 3, 4],
    # ]
    # and we want to extract the kernel at index (1,3) with kernel_width = 4, kernel_height = 3 we and up with:
    # [ [3, 4, 4, 3]
    #   [3, 4, 4, 3],
    #   [3, 4, 4, 3],
    # ]
    row_min_mask = row_idx < 0
    column_min_mask = column_idx < 0
    row_max_mask = row_idx >= img_w
    column_max_mask = column_idx >= img_h

    row_idx[row_min_mask] = 0 - row_idx[row_min_mask]
    row_idx[row_max_mask] = img_w - (row_idx[row_max_mask] - img_w + 1)

    column_idx[column_min_mask] = 0 - column_idx[column_min_mask]
    column_idx[column_max_mask] = img_h - (column_idx[column_max_mask] - img_h + 1)

    # Now we need all combinations of row_idx and column_idx
    mesh = np.array(np.meshgrid(row_idx, column_idx))
    combinations = mesh.T.reshape(-1, 2)

    return combinations[:, 0], combinations[:, 1]

def image_smoother(input_image: np.ndarray,
                   kernel_height=5,
                   kernel_width=5,
                   stride_height=1,
                   stride_width=1) -> np.ndarray:
    assert input_image.ndim == 4
    assert input_image.shape[0] == 1 and input_image.shape[-1] == 3

    input_node = tf.placeholder(
        dtype=tf.float32,
        shape=input_image.shape
    )

    # This objective is the only part you need to change.
    # try to use as less for-loops as possible
    # Note: because of the assertion on line 52 I assume that the function
    #       works on single images.
    kernel_center_row = list(range(0, img[0].shape[0], stride_width))
    kernel_center_column = list(range(0, img[0].shape[1], stride_height))
    kernel_center = np.array(np.meshgrid(kernel_center_row, kernel_center_column)).T.reshape(-1, 2)

    indexes = np.array([kernel_idx(img[0], row, column, kernel_width, kernel_height) for row, column in kernel_center])
    kernels = np.array([input_image[0][row, column, :] for row, column in indexes])
    mean_kernel_numpy = tf.Session().run(tf.reduce_mean(kernels,
                                       axis=(1),
                                       keepdims=True))

    mean_kernel = np.zeros(input_image.shape)
    for i, (row, column) in enumerate(indexes):
        mean_kernel[0][row, column, :] = mean_kernel_numpy[i]

    objective = tf.square(input_node - mean_kernel)
    # Please note that this objective would only be correct if
    # (kernel_height, kernel_width) == input_image.shape[1:3]
    # your approach must be flexible under kernel size and stride

    gradient_node = tf.gradients(ys=objective, xs=input_node)

    with tf.Session() as session:
        for _ in tqdm(range(1000), desc="optimize image"):
            _gradient = session.run(
                gradient_node,
                feed_dict={
                    input_node: input_image
                }
            )
            gradient_step = np.sign(_gradient[0]) * (1 / 255)
            # in the pdf code a_min and a_max were missing, but according
            # to numpy docs at least one of them has to be provided
            input_image = np.clip(input_image - gradient_step, np.min(input_image), np.max(input_image))

    return input_image

img = image.imread("img.jpg")
img = img[400:600, 400:600, :]
img = np.array([img / 255.0], dtype="float32")
plt.subplot(121)
plt.imshow(img[0])

new_img = image_smoother(img, 10, 10, 3, 3)
plt.subplot(122)
plt.imshow(new_img[0])

plt.show()
#
# ar = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9],
# ])
#
# idx = [
#     (0, 1),
#     (2,  2)
# ]
#
# row_idx = [0, 2]
# clm_idx = [1, 2]
#
# mesh = np.array(np.meshgrid(row_idx, clm_idx)).T.reshape(-1, 2)
# print(mesh)
#
# print(ar)
# print("---------")
# # print(ar[idx])
# # print(ar[row_idx, clm_idx])
# ar[mesh[:, 0], mesh[:, 1]] = 0
# print(ar)