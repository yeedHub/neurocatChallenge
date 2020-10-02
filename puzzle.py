"""
A local neighborhood are groups of neighboring pixels.
These local neighborhoods have a height, a width and a stride, very
similar to kernels of convolutions in a neural network.
more information: https://d2l.ai/chapter_convolutional-neural-networks/paddin g-and-strides.html
Adapt the method below that is supposed to take an image and optimizes this image, s.t. each pixel in every local
neighborhood of the image converges to the mean color of its respective neighborhood.
Only adapt the objective. Generate a docstring. Test your method.
"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm


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
    kernels = input_image  # this is just a placeholder ...

    mean_kernel = tf.reduce_mean(kernels,
                                 axis=(1, 2),
                                 keepdims=True)

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
            input_image = np.clip(input_image - gradient_step)

    return input_image
