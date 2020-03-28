import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# pixel werte auf 0 bis 1 skalieren
train_images = train_images / 255.0
test_images = test_images / 255.0

test_image = train_images[0]

plt.gray()
plt.imshow(test_image)
plt.show()

print(test_image.shape)

x0 = test_image.shape[0] / 2
y0 = test_image.shape[1] / 2
u0 = test_image.shape[0] / 2
v0 = test_image.shape[1] / 2
d = 50
fx = 100
fy = 100
transforms = np.asarray([d/fx,0,-u0*d/fx+x0,0,d/fy,-v0*d/fy+y0,0,0],dtype=np.float32)

test_image = tfa.image.transform(
    test_image,
    transforms,
    interpolation='BILINEAR',
)


plt.imshow(test_image)
plt.show()