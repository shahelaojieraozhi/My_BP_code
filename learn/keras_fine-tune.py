# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/8/13 16:10
@Author  : Rao Zhi
@File    : keras_fine-tune.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 
@refer: https://blog.csdn.net/Tourior/article/details/83822944
"""


from keras.layers import Dropout, Flatten, Dense
from keras.layers import BatchNormalization, Input, Conv2D
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model


def create_model(n_out):
    input_shape = (256, 256, 3)
    input_tensor = Input(shape=(256, 256, 3))
    base_model = InceptionV3(include_top=False,
                             weights='inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                             input_shape=input_shape
                             )
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Conv2D(32, kernel_size=(1, 1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)

    return model


# warm up model
model = create_model(n_out=10)
model.summary()

for layer in model.layers:
    layer.trainable = False

# model.trainable = False   # Or freeze all layers using this method

model.layers[-1].trainable = True
model.layers[-2].trainable = True
model.layers[-3].trainable = True
model.layers[-4].trainable = True
model.layers[-5].trainable = True
model.layers[-6].trainable = True

print("trainable layers is:")
print()
for x in model.trainable_weights:
    print(x.name)
print('\n')

print("don't trainable layers is:")
print()
for x in model.non_trainable_weights:
    print(x.name)

