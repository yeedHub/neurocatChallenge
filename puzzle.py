"""
A local neighborhood are groups of neighboring pixels.
These local neighborhoods have a height, a width and a stride, very
similar to kernels of convolutions in a neural network.
more information: https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
Adapt the method below that is supposed to take an image and optimizes this image, s.t. each pixel in every local
neighborhood of the image converges to the mean color of its respective neighborhood.
Only adapt the objective. Generate a docstring. Test your method.
"""
import numpy as np
import tensorflow as tf

from tqdm import tqdm


def kernel_index(img_shape: tuple, row_index: int, column_index: int, kernel_width: int, kernel_height: int) -> tuple:
    """
    Calculates all indices for a kernel with kernel center (row_index, column_index),
    kernel_width width and kernel_height height.

    Args:
        img_shape: shape of the input image
        row_index: row index of the kernel center
        column_index: column index of the kernel center
        kernel_width: width of the kernel
        kernel_height: height of the kernel

    Returns: tuple in the form of (row_indices, column_indices) that make up the kernel

    """
    img_w = img_shape[0]
    img_h = img_shape[1]
    range_w = (int((kernel_width - 1) / 2), int((kernel_width - 1) / 2) + 1)
    if (kernel_width - 1) % 2 != 0:
        range_w = (range_w[0] + 1, range_w[1])

    range_h = (int((kernel_height - 1) / 2), int((kernel_height - 1) / 2) + 1)
    if (kernel_height - 1) % 2 != 0:
        range_h = (range_h[0] + 1, range_h[1])

    row_indices = np.arange(row_index - range_w[0], row_index + range_w[1])
    column_indices = np.arange(column_index - range_h[0], column_index + range_h[1])

    # For those kernels that go over the edge we use a standard image processing
    # technique, where you "mirror" the image and "place" it next to the edge.
    # So if our image is:
    # [ [1, 2, 3, 4],
    #   [1, 2, 3, 4],
    #   [1, 2, 3, 4],
    # ]
    # and we want to extract the kernel at index (1,3) with kernel_width = 4, kernel_height = 3 we end up with:
    # [ [3, 4, 4, 3]
    #   [3, 4, 4, 3],
    #   [3, 4, 4, 3],
    # ]
    row_min_mask = row_indices < 0
    column_min_mask = column_indices < 0
    row_max_mask = row_indices >= img_w
    column_max_mask = column_indices >= img_h

    row_indices[row_min_mask] = 0 - row_indices[row_min_mask]
    row_indices[row_max_mask] = img_w - (row_indices[row_max_mask] - img_w + 1)

    column_indices[column_min_mask] = 0 - column_indices[column_min_mask]
    column_indices[column_max_mask] = img_h - (column_indices[column_max_mask] - img_h + 1)

    # Now we need all combinations of row_idx and column_idx
    combinations = np.array(np.meshgrid(row_indices, column_indices)).T.reshape(-1, 2)

    return combinations[:, 0], combinations[:, 1]


def compute_kernel_centers(img_shape: tuple, row_stride: int, column_stride: int) -> np.ndarray:
    """
    Computes all the kernel center indices with the given strides.

    Args:
        img_shape: shape of the input image
        row_stride: stride of the rows
        column_stride: stride of the columns

    Returns: kernel center indices

    """
    kernel_centers_row = np.arange(0, img_shape[0], row_stride)
    kernel_centers_clm = np.arange(0, img_shape[1], column_stride)

    return np.array(np.meshgrid(kernel_centers_row, kernel_centers_clm)).T.reshape(-1, 2)


def compute_kernels(input_image: np.ndarray, kernel_centers: np.ndarray, kernel_width: int,
                    kernel_height: int) -> tuple:
    """
    Computes the kernels for the given kernel_centers, with given kernel_width and kernel_height.

    Args:
        input_image: the input image
        kernel_centers: the kernel center indices
        kernel_width: the kernel width
        kernel_height: the kernel height

    Returns: a tuple in the form of (kernels, indices), where indices are the indices of the kernel in the input image

    """
    indices = []
    kernels = []
    for row, column in kernel_centers:
        index = kernel_index(input_image.shape, row, column, kernel_width, kernel_height)

        indices.append(index)
        kernels.append(input_image[index[0], index[1], :])

    return np.array(kernels), np.array(indices)


def compute_mean_kernel_image(img_shape: tuple, kernels: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Constructs an image of all the kernels, where each of the kernels elements is the mean of the kernel.

    Args:
        img_shape: shape of the input image
        kernels: array of kernels
        indices: array of indices

    Returns: the constructed image

    """
    mean_kernel_numpy = tf.Session().run(tf.reduce_mean(kernels,
                                                        axis=1,
                                                        keepdims=True))
    mean_kernel_img = np.zeros(img_shape)

    for i, (row, column) in enumerate(indices):
        mean_kernel_img[row, column, :] = mean_kernel_numpy[i]

    return np.array([mean_kernel_img])


def image_smoother(input_image: np.ndarray,
                   kernel_height=5,
                   kernel_width=5,
                   stride_height=1,
                   stride_width=1) -> np.ndarray:
    """
    Takes an image and smoothes it.

    Args:
        input_image: the input image
        kernel_height: the kernel height
        kernel_width: the kernel width
        stride_height: the kernel stride for columns
        stride_width: the kernel stride for rows

    Returns: the smoothed image

    """
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
    kernel_centers = compute_kernel_centers(input_image[0].shape, stride_width, stride_height)
    kernels, indices = compute_kernels(input_image[0], kernel_centers, kernel_width, kernel_height)
    mean_kernel_img = compute_mean_kernel_image(input_image[0].shape, kernels, indices)

    objective = tf.square(input_node - mean_kernel_img)
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
