import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("Numpy/output.png")
plt.imshow(img)
#plt.show()

imggray = img.convert('LA')
plt.imshow(imggray)
#plt.show()


#Convert data into numpy matrix

import numpy as np

imgmat = np.array(list(imggray.getdata  (band=0)), float)
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.matrix(imgmat)
#plt.imshow(imgmat, cmap='gray')
#plt.show()

U, sigma, V = np.linalg.svd(imgmat)
#print(U)


# this code was about eigen values in Descending order.  SVD
reconstimg = np.matrix(U[:,:1]) * np.diag(sigma[:1]) * np.matrix(V[:1, :])           # we can use 50 components for better image quality instead of 1component..
plt.imshow(reconstimg, cmap='gray')
plt.show()

#Additional Singular Vector improve the image quality

for i in [2,4,8,16,32,64]:
    reconstimg = np.matrix(U[:,:i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
    plt.imshow(reconstimg, cmap='gray')
    title = "n = %s" % i
    plt.title(title)
    plt.show()