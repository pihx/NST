# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision

#__future__ allows introducing new behavior into the language 
# that would be backwards-incompatible.
from __future__ import print_function, division

from builtins import range, input
# Note: you may need to update your version of future



# In this script, we will focus on generating an image

# that attempts to match the content of one input image
# and the style of another input image.
#
# We accomplish this by balancing the content loss
# and style loss simultaneously.

from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
#from keras.preprocessing import image
from keras_preprocessing import image

from skimage.transform import resize

import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

from style_transfer1 import VGG16_AvgPool, VGG16_AvgPool_CutOff, unpreprocess, scale_img
from style_transfer2 import gram_matrix, style_loss, minimize
from scipy.optimize import fmin_l_bfgs_b


# load the content image
def load_img_and_preprocess(path, shape=None):
    img = image.load_img(path, target_size=shape)

    # convert keras image object to np array and preprocess for vgg
    x = image.img_to_array(img)
    #x.shape(397,635,3)
    x = np.expand_dims(x, axis=0)
     #x.shape(1,397,635,3)
    x = preprocess_input(x)

    return x


content_img = load_img_and_preprocess(
    #'content/elephant.jpg',
    'content/sydney.jpg',
    # (225, 300),
)
#content_img: array([[[[149.061, 34.221, -24.68],]]])

# resize the content image
# since we don't care too much about warping it
h, w = content_img.shape[1:3]

style_img = load_img_and_preprocess(
    'styles/starrynight.jpg',
    #'styles/flowercarrier.jpg',
    #'styles/monalisa.jpg',
    #'styles/lesdemoisellesdavignon.jpg',
    (h, w)
)
#style_img: array([[[[40.060, -12.778, -47.68],]]]),shape(1,397,635,3), max:144.061, min:-123.68, size:756285


# we'll use this throughout the rest of the script
batch_shape = content_img.shape
#batch_shape:(1,397,635,3)
shape = content_img.shape[1:]
#shape:(397,635,3)

# we want to make only 1 VGG here
# as you'll see later, the final model needs
# to have a common input
vgg = VGG16_AvgPool(shape)


# create the content model
# we only want 1 output
# remember you can call vgg.summary() to see a list of layers
# 1,2,4,5,7-9,11-13,15-17
#vgg.summary()
# original content_model = Model(vgg.input, vgg.layers[13].get_output_at(0))
#

content_model = Model(vgg.input, vgg.layers[12].get_output_at(0))
#content_model = Model(vgg.input, vgg.layers[13].get_output_at(0))

content_target = K.variable(content_model.predict(content_img))
# create the style model
# we want multiple outputs
# we will take the same approach as in style_transfer2.py
symbolic_conv_outputs = [
    layer.get_output_at(1) for layer in vgg.layers
    if layer.name.endswith('conv1')
]
# make a big model that outputs multiple layers' outputs
style_model = Model(vgg.input, symbolic_conv_outputs)
# calculate the targets that are output at each layer
style_layers_outputs = [K.variable(y) for y in style_model.predict(style_img)]

# we will assume the weight of the content loss is 1
# and only weight the style losses
style_weights = [0.2, 0.4, 0.3, 0.5, 0.2]
#style_weights = [0.6,1, 0.6, 1, 0.6]
#style_weights = [1,1,1,1,1]

# create the total loss which is the sum of content + style loss
loss = K.mean(K.square(content_model.output - content_target))

for w, symbolic, actual in zip(style_weights, symbolic_conv_outputs, style_layers_outputs):
    # gram_matrix() expects a (H, W, C) as input
    loss += w * style_loss(symbolic[0], actual[0])


# once again, create the gradients and loss + grads function
# note: it doesn't matter which model's input you use
# they are both pointing to the same keras Input layer in memory
grads = K.gradients(loss, vgg.input)

# just like theano.function x = vgg.input
get_loss_and_grads = K.function(
    inputs=[vgg.input],
    outputs=[loss] + grads
)


def get_loss_and_grads_wrapper(x_vec):
    # l, g = get_loss_and_grads(img)
    # where img = [x_vec.reshape(*batch_shape)], img.shape == (1,H,W,3)
    # l is a scalar, g.shape == (1,H,W,3)
    
    l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])

    return l.astype(np.float64), g.flatten().astype(np.float64)


final_img = minimize(get_loss_and_grads_wrapper, 6, batch_shape)

plt.imsave('.\\results\\styleTransfer3\\nst.jpg',scale_img(final_img))
plt.imshow(scale_img(final_img))
plt.show()

