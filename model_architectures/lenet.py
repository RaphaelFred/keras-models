from tensorflow import keras


def lenet(input_shape, output_shape, normalize=False):
    input_img = keras.Input(shape=input_shape)

    # # Normalization layer
    # # if normalize:
    # input_img = keras.layers.Rescaling(scale=1./255, input_shape=input_shape)(input_img),

    # First Block CONVOLUTION -> MAX_POOLING
    Z1 = keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid')(input_img)
    A1 = keras.layers.Activation(keras.activations.relu)(Z1)
    P1 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(A1)

    # Second Block CONVOLUTION -> MAX_POOLING
    Z2 = keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid')(P1)
    A2 = keras.layers.Activation(keras.activations.relu)(Z2)
    P2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(A2)

    # Fully-connected Layers
    F = keras.layers.Flatten()(P2)
    FC1 = keras.layers.Dense(units=400, activation='relu')(F)
    FC2 = keras.layers.Dense(units=120, activation='relu')(FC1)
    FC3 = keras.layers.Dense(units=84, activation='relu')(FC2)
    outputs = keras.layers.Dense(units=output_shape, activation='softmax')(FC3)

    model = keras.Model(inputs=input_img, outputs=outputs)

    return model
