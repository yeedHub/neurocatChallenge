import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np

from puzzle import image_smoother

img = image.imread("img.jpg")
img = np.array([img / 255.0], dtype="float32")
plt.subplot(121)
plt.imshow(img[0])

new_img = image_smoother(img, 10, 10, 3, 3)
plt.subplot(122)
plt.imshow(new_img[0])

plt.show()
