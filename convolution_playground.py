import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import asarray
from scipy import signal

image = Image.open('test_image_bw.jpg')
image_data = asarray(image)

filter_kernel = np.array([[1, 0, 0, 0, -1],
                          [1, 0, 0, 0, -1],
                          [1, 0, 0, 0, -1],
                          [1, 0, 0, 0, -1],
                          [1, 0, 0, 0, -1]])

# filter_kernel = filter_kernel.transpose()

processed_image_data = signal.convolve2d(image_data, filter_kernel, mode='same', boundary='symm')

images = [image_data, processed_image_data]

for i in range(2):
    plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    plt.show()
