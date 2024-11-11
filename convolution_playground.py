import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import asarray
from scipy import signal

image = Image.open('test_me_bw.jpg')
image_data = asarray(image)

a = np.random.rand(3, 6)
b = -a
filter_kernel = np.concatenate((a, b)) * 2
np.random.shuffle(filter_kernel)

processed_image_data = image_data

for i in range(1):
    processed_image_data = signal.convolve2d(processed_image_data, filter_kernel, mode='same', boundary='symm')

plt.imshow(processed_image_data, cmap='gray', vmin=0, vmax=255)
plt.show()
