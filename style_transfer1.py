# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

# In this script, we will focus on generating the content
# E.g. given an image, can we recreate the same image


from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin_l_bfgs_b

import tensorflow as tf
if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()
 
from keras_preprocessing import image

def VGG16_AvgPool(shape):
    # we want to account for features across the entire image
    # so get rid of the maxpool which throws away information
    vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)

    i = vgg.input
    # i: <tf.Tensor 'input_1:0' shape=(None, 397, 635, 3) dtype=float32>
    x = i
    
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            # replace it with average pooling
            x = AveragePooling2D()(x)
        else:
            x = layer(x)
    
    return Model(i, x)


def VGG16_AvgPool_CutOff(shape, num_convs):
    # there are 13 convolutions in total
    # we can pick any of them as the "output"
    # of our content model

    if num_convs < 1 or num_convs > 13:
        print("num_convs must be in the range [1, 13]")
        return None

    model = VGG16_AvgPool(shape)
   
    n = 0
    output = None
    for layer in model.layers:
        if layer.__class__ == Conv2D:
            n += 1
        if n >= num_convs:
            output = layer.output
            break

    return Model(model.input, output)


def unpreprocess(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img


def scale_img(x):
    x = x - x.min()
    x = x / x.max()
    return x


if __name__ == '__main__':

    # open an image
    # feel free to try your own
    #path = 'content/elephant.jpg'
    path = 'content/sydney.jpg'
    
    img = image.load_img(path)
    # Using Keras func to load image. img is an image object

    # convert image to numpy array and preprocess for vgg
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # we'll use this throughout the rest of the script
    batch_shape = x.shape
    shape = x.shape[1:]

    # see the image
    # plt.imshow(img)
    # plt.show()

    # make a content model
    # try different cutoffs to see the images that result
    # content_model = VGG16_AvgPool_CutOff(shape, 10)
    content_model = VGG16_AvgPool_CutOff(shape, 11)
    
    # we make the target, which is the model output given x
    target = K.variable(content_model.predict(x))

    # try to match the image

    # define our loss in keras: mean square error of symbolic content model output and target
    loss = K.mean(K.square(target - content_model.output))

    # gradients which are needed by the optimizer. Symbolic variables.
    grads = K.gradients(loss, content_model.input)

    # just like theano.function Symbolic
    get_loss_and_grads = K.function(
        inputs=[content_model.input],
        outputs=[loss] + grads
    )
    
    # Input x_vec must be numpy array:
    # return a numpy array using these symbolic variables just build.
    def get_loss_and_grads_wrapper(x_vec):
        # scipy's minimizer allows us to pass back
        # function value f(x) and its gradient f'(x)
        # simultaneously, rather than using the fprime arg
        #
        # we cannot use get_loss_and_grads() directly
        # input to minimizer func must be a 1-D array
        # input to get_loss_and_grads must be [batch_of_images]
        # 
        # gradient must also be a 1-D array
        # and both loss and gradient must be np.float64
        # will get an error otherwise
       
        #1-D vector reshaped into an image, call get_loss_and_grads()
        l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
        # flatten the grads back to 1-D vector and cast both to float64
        return l.astype(np.float64), g.flatten().astype(np.float64)

    from datetime import datetime
    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))
    #range(i), where i is epochs/iterations
    for i in range(10):
        # optimize the image
        x, l, _ = fmin_l_bfgs_b(
            func=get_loss_and_grads_wrapper,
            x0=x,
            # bounds=[[-127, 127]]*len(x.flatten()),
            maxfun=20
        )
        x = np.clip(x, -127, 127)
        # print("min:", x.min(), "max:", x.max())
        print("iter=%s, loss=%s" % (i, l))
        losses.append(l)

    print("duration:", datetime.now() - t0)
    plt.plot(losses)
    plt.show()
    
    #reshape the vector x back to an image
    newimg = x.reshape(*batch_shape)
    #reverse the preprocessing, to be able to display it
    final_img = unpreprocess(newimg)
    #scale the image before plotting it.
    plt.imsave('only_content.jpg',scale_img(final_img[0]))
    plt.imshow(scale_img(final_img[0]))
    plt.show()
    
