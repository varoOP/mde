
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.backend import int_shape


from PIL import Image, ImageOps

def model_predict(img, model):
    prediction = model.predict(img)
    return prediction

def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)
    w1 = 1.0
    w2 = 1.0
    w3 = theta
    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))


# Model 
def res_unet(filter_root=4, layers=8, n_class=1, input_size=(384, 1280, 3), activation='relu', batch_norm=True, final_activation='sigmoid'):
    inputs = Input(input_size)
    x = inputs
    long_connection_store = {}

    # Down sampling
    for i in range(layers):
        out_channel = 2**i * filter_root
        # print("out_channel downsampling: {}".format(out_channel))

        # Residual/Skip connection
        res = Conv2D(out_channel, kernel_size=1,
                     padding='same', use_bias=False, name="Identity{}_1".format(i))(x)

        # First Conv Block with Conv, BN and activation
        conv1 = Conv2D(out_channel, kernel_size=3, padding='same', name="Conv{}_1".format(i))(x)
        if batch_norm:
            conv1 = BatchNormalization(name="BN{}_1".format(i))(conv1)
        act1 = Activation(activation, name="Act{}_1".format(i))(conv1)

        # Second Conv block with Conv and BN only
        conv2 = Conv2D(out_channel, kernel_size=3, padding='same', name="Conv{}_2".format(i))(act1)
        if batch_norm:
            conv2 = BatchNormalization(name="BN{}_2".format(i))(conv2)

        resconnection = Add(name="Add{}_1".format(i))([res, conv2])

        act2 = Activation(activation, name="Act{}_2".format(i))(resconnection)

        # Max pooling
        if i < layers - 1:
            long_connection_store[str(i)] = act2
            x = MaxPooling2D(padding='same', name="MaxPooling{}_1".format(i))(act2)
        else:
            x = act2
    # print("\n")
    # Upsampling
    for i in range(layers - 2, -1, -1):
        # print("i upsampling: {}".format(i))

        out_channel = 2**(i) * filter_root
        # print("out_channel upsampling: {}".format(out_channel))

        # long connection from down sampling path.
        long_connection = long_connection_store[str(i)]
        # print("long_connection: {}".format(long_connection))

        up1 = UpSampling2D(name="UpSampling{}_1".format(i))(x)
        up_conv1 = Conv2D(out_channel, 2, activation='relu', padding='same', name="upsamplingConv{}_1".format(i))(up1)
        # print("up_conv1: {}".format(up_conv1))

        crop_shape = get_crop_shape(int_shape(up_conv1), int_shape(long_connection))

        crop_connection = Cropping2D(cropping=crop_shape, name="upCrop{}_1".format(i))(long_connection)

        #  Concatenate.
        up_conc = Concatenate(axis=-1, name="upConcatenate{}_1".format(i))([up_conv1, crop_connection])

        #  Convolutions
        up_conv2 = Conv2D(out_channel, 3, padding='same', name="upConv{}_1".format(i))(up_conc)
        if batch_norm:
            up_conv2 = BatchNormalization(name="upBN{}_1".format(i))(up_conv2)
        up_act1 = Activation(activation, name="upAct{}_1".format(i))(up_conv2)

        up_conv2 = Conv2D(out_channel, 3, padding='same', name="upConv{}_2".format(i))(up_act1)
        if batch_norm:
            up_conv2 = BatchNormalization(name="upBN{}_2".format(i))(up_conv2)

        # Residual/Skip connection
        res = Conv2D(out_channel, kernel_size=1,
                     padding='same', use_bias=False, name="upIdentity{}_1".format(i))(up_conc)

        resconnection = Add(name="upAdd{}_1".format(i))([res, up_conv2])

        x = Activation(activation, name="upAct{}_2".format(i))(resconnection)

    # Final convolution
    output = Conv2D(n_class, 1, padding='same',
                    activation=final_activation, name='output')(x)

    return Model(inputs, outputs=output, name='Res-UNet')



def get_crop_shape(target, source):
    # source is coming from down sampling path.
    # target is coming from up sampling operation.
    source_height_width = np.array(source[1:-1])
    target_height_widht = np.array(target[1:-1])

    diff = (source_height_width - target_height_widht).tolist()
    diff_tup = map(lambda x: (x//2, x//2) if x%2 == 0 else (x//2, x//2 + 1), diff)
    return tuple(diff_tup)


