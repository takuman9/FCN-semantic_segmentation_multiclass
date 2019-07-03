from keras.models import Model
from keras.engine.topology import Input
from keras.layers import LeakyReLU
from keras.layers.core import Flatten, Dense, Reshape, Dropout, Activation
from keras.layers.convolutional import Conv2D, UpSampling2D,Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.initializers import Initializer
class FcnNet(object):
    def __init__(self, input_shape, class_num):
        Initializer()
        self.INPUT_SHAPE = input_shape
        self.CLASS_NUM  = class_num

        inputs = Input((self.INPUT_SHAPE), name='input')
        encodeLayer1 = self.__add_encode_layers(16, inputs, is_first=True, add_drop_layer=True)
        encodeLayer2 = self.__add_encode_layers(32, encodeLayer1, add_drop_layer=True)
        encodeLayer3 = self.__add_encode_layers(64, encodeLayer2, add_drop_layer=True)
        encodeLayer4 = self.__add_encode_layers(64, encodeLayer3, add_drop_layer=True)
        straightLayer1 = self.__add_encode_layers(64, encodeLayer4, is_straight=True, add_drop_layer=True)
        straightLayer2 = self.__add_encode_layers(64, straightLayer1, is_straight=True, add_drop_layer=True)

        decodeLayer3 = self.__add_decode_layers(
            64, straightLayer2, encodeLayer4,first_decode_layer=True, add_drop_layer=True)
        decodeLayer2 = self.__add_decode_layers(
            32, decodeLayer3, encodeLayer3, add_drop_layer=True)
        decodeLayer1 = self.__add_decode_layers(
            16, decodeLayer2, encodeLayer2, add_drop_layer=True)

        decodeLayer0 = concatenate([decodeLayer1, encodeLayer1])
        outputs = Conv2D(class_num,(3,3),strides=(1, 1),padding='same', name='output')(decodeLayer0)
        out     = Activation('softmax')(outputs)
        print(outputs.shape)

        self.MODEL = Model(inputs=[inputs], outputs=[out])

    def __add_encode_layers(self, filter_num, input_layer, is_first=False ,is_straight=False,add_drop_layer=False):
        layer = input_layer
        if is_first:
            layer = Conv2D(filter_num, 3, strides=(1, 1),padding='same', input_shape=(self.INPUT_SHAPE))(layer)
        elif is_straight:
            layer = Conv2D(filter_num, 3, strides=(1, 1), padding='same')(layer)
        else:
            layer = Conv2D(filter_num, 3, strides=(2, 2), padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = LeakyReLU(alpha=0.01)(layer)
        if add_drop_layer:
            layer = Dropout(0.5)(layer)
        print(layer.shape)
        return layer

    def __add_decode_layers(self, filter_num, input_layer, concat_layer, add_drop_layer=False,first_decode_layer=False,last_decode_layer=False):
        if first_decode_layer:
            layer = concatenate([input_layer, concat_layer])
        else:
            # layer = UpSampling2D(size=(2, 2))(input_layer)
            # layer = concatenate([layer, concat_layer])
            layer = concatenate([input_layer, concat_layer])
        layer = Conv2DTranspose(filters=filter_num,kernel_size=4, use_bias=True, strides=(2,2),padding='same')(layer)
        # layer = Conv2D(filter_num, 4,strides=(1, 1), padding='same')(layer)
        # layer = Conv2DTranspose(filters=filter_num,kernel_size=4, use_bias=False, strides=(2,2))(input_layer)
        layer = BatchNormalization()(layer)
        layer = LeakyReLU(alpha=0.01)(layer)
        if add_drop_layer:
            layer = Dropout(0.5)(layer)
        print(layer.shape)
        return layer


    def model(self):
        return self.MODEL
