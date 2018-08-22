from keras.activations import softmax, linear
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D, MaxPooling2D, AveragePooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.constraints import maxnorm
from keras import regularizers


def model1(x_train, y_train, baseDim = 64, activation = "softplus", padding = "valid", depth = 1):

    cnn = Sequential()

    cnn.add(Convolution2D(baseDim, (7,4),  strides = (1,1), use_bias=False, padding=padding,  activation=activation,
                          input_shape=(x_train.shape[1], x_train.shape[2], depth)))
    cnn.add(BatchNormalization())

    cnn.add(MaxPooling2D(pool_size=(3,2)))
    cnn.add(Dropout(0.5))


    cnn.add(Convolution2D(baseDim*2, (4,2),  strides = (1,1), padding=padding, use_bias=False, activation=activation))
    cnn.add(BatchNormalization())

    cnn.add(Convolution2D(baseDim*4, (4,3),  strides = (1,1), padding=padding, use_bias=False, activation=activation))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.6))
    cnn.add(MaxPooling2D(pool_size=(5,1)))


    cnn.add(Flatten())

    cnn.add(Dense(100, activation=activation))
    cnn.add(Dropout(0.75))
    cnn.add(BatchNormalization())

    cnn.add(Dense(y_train.shape[1], activation="sigmoid"))
    return cnn
    

def model2 (x_train, y_train, baseDim = 128, activation = "softplus", padding = "valid", depth = 1):
    cnn = Sequential()

    cnn.add(Convolution2D(baseDim, (6,4),  strides = (1,1), padding=padding, 
                          input_shape=(x_train.shape[1], x_train.shape[2],depth)))
    cnn.add(BatchNormalization())
    cnn.add(Activation(activation))
    

    cnn.add(Dropout(0.4))
    cnn.add(Convolution2D(baseDim, (5,3),  strides = (1,1), padding=padding))
    cnn.add(BatchNormalization())
    cnn.add(Activation(activation))
    

    cnn.add(MaxPooling2D(pool_size=(3,3)))
    cnn.add(Dropout(0.4))


    cnn.add(Convolution2D(baseDim, (3,2),  strides = (1,1), padding=padding))
    cnn.add(BatchNormalization())
    cnn.add(Activation(activation))
    
    cnn.add(Dropout(0.1))
    cnn.add(Convolution2D(baseDim*2, (4,2),  strides = (1,1), padding=padding))
    cnn.add(BatchNormalization())
    cnn.add(Activation(activation))
    

    cnn.add(MaxPooling2D(pool_size=(4,1)))
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    cnn.add(Dense(100, activation=activation))
    cnn.add(Dropout(0.6))

    cnn.add(Dense(y_train.shape[1], activation='softmax'))

    return cnn
    
def model3(x_train, y_train, baseDim = 16, activation = "softplus", padding = "same", depth = 1):
    cnn = Sequential()

    cnn.add(Convolution2D(baseDim, (4,2),  strides = (1,1), padding="valid", 
                          input_shape=(x_train.shape[1], x_train.shape[2],depth)))
    cnn.add(Activation(activation))

    cnn.add(Convolution2D(baseDim * 2, (2,2),  strides = (1,1), padding=padding))
    cnn.add(Activation(activation))


    cnn.add(Convolution2D(baseDim*4, (2,2),  strides = (1,1), padding=padding, use_bias=False))
    cnn.add(BatchNormalization())
    cnn.add(Activation(activation))
    
    cnn.add(MaxPooling2D(pool_size=(4,2)))

    cnn.add(Dropout(0.4))
    cnn.add(BatchNormalization())
    cnn.add(Convolution2D(baseDim * 2, (2,2),  strides = (1,1), padding=padding ))
    cnn.add(Activation(activation))

    #cnn.add(Dropout(0.2))
    cnn.add(Convolution2D(baseDim * 4, (2,2),  strides = (1,1), padding=padding, use_bias=False))
    cnn.add(BatchNormalization())
    cnn.add(Activation(activation))
    

    cnn.add(MaxPooling2D(pool_size=(2,2)))

    cnn.add(Dropout(0.4))
    cnn.add(BatchNormalization())
    cnn.add(Convolution2D(baseDim *8, (2,2),  strides = (1,1), padding=padding))
    cnn.add(Activation(activation))



    cnn.add(MaxPooling2D(pool_size=(4,2)))

    cnn.add(Dropout(0.3))


    cnn.add(Flatten())

    cnn.add(Dense(80, activation=activation))

    cnn.add(Dropout(0.5))
    cnn.add(BatchNormalization())

    cnn.add(Dense(y_train.shape[1], activation="softmax"))
    return cnn


def tinyDarknet(x_train, y_train, baseDim = 16, activation = "softplus", padding = "same", depth = 1, dropout = 0.1, regularizer = 0.01):
    cnn = Sequential()
    cnn.add(Dropout(dropout))
    cnn.add(Convolution2D(baseDim*2, (3,3),  strides = (1,1), use_bias=False, padding=padding, 
                          input_shape=(x_train.shape[1], x_train.shape[2],depth)))
    cnn.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    cnn.add(LeakyReLU(alpha=.1))
    #cnn.add(MaxPooling2D(pool_size=(2,2)))
    
    cnn.add(Dropout(dropout))
    cnn.add(Convolution2D(baseDim, (3,3),  strides = (1,1), use_bias=False,padding=padding))
    cnn.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    cnn.add(LeakyReLU(alpha=.1))    
    cnn.add(MaxPooling2D(pool_size=(2,2)))

    cnn.add(Dropout(dropout))
    cnn.add(Convolution2D(baseDim, (1,1),  strides = (1,1),use_bias=False, padding=padding))
    cnn.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    cnn.add(LeakyReLU(alpha=.1))
    
    cnn.add(Convolution2D(baseDim*8, (3,3),  strides = (1,1), use_bias=False, padding=padding))
    cnn.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    cnn.add(LeakyReLU(alpha=.1))
    
    cnn.add(Convolution2D(baseDim, (1,1),  strides = (1,1), use_bias=False, padding=padding))
    cnn.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    cnn.add(LeakyReLU(alpha=.1))
    
    cnn.add(Convolution2D(baseDim*8, (3,3),  strides = (1,1), use_bias=False, padding=padding))
    cnn.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    cnn.add(LeakyReLU(alpha=.1))

    
    cnn.add(MaxPooling2D(pool_size=(2,2)))

    cnn.add(Dropout(dropout))
    cnn.add(Convolution2D(baseDim*2, (1,1),  strides = (1,1), use_bias=False, padding=padding))
    cnn.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    cnn.add(LeakyReLU(alpha=.1))
    
    cnn.add(Convolution2D(baseDim*8*2, (3,3),  strides = (1,1), use_bias=False, padding=padding))
    cnn.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    cnn.add(LeakyReLU(alpha=.1))
    
    cnn.add(Convolution2D(baseDim*2, (1,1),  strides = (1,1),use_bias=False, padding=padding))
    cnn.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    cnn.add(LeakyReLU(alpha=.1))
    
    cnn.add(Convolution2D(baseDim*8*2, (3,3),  strides = (1,1), use_bias=False, padding=padding))
    cnn.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    cnn.add(LeakyReLU(alpha=.1))
    
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    
    cnn.add(Dropout(dropout))
    cnn.add(Convolution2D(baseDim*2*2, (1,1),  strides = (1,1),use_bias=False, padding=padding))
    cnn.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    cnn.add(LeakyReLU(alpha=.1))
    
    
    cnn.add(Convolution2D(baseDim*8*2*2, (3,3),  strides = (1,1), use_bias=False,padding=padding))
    cnn.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    cnn.add(LeakyReLU(alpha=.1))
    
 
    cnn.add(Convolution2D(baseDim*2*2, (1,1),  strides = (1,1), use_bias=False,padding=padding))
    cnn.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    cnn.add(LeakyReLU(alpha=.1))
    
    
    cnn.add(Convolution2D(baseDim*8*2*2, (3,3),  strides = (1,1),use_bias=False, padding=padding))
    cnn.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    cnn.add(LeakyReLU(alpha=.1))
    
    
    cnn.add(Convolution2D(baseDim*2*2, (1,1),  strides = (1,1), use_bias=False,padding=padding))
    cnn.add(BatchNormalization(epsilon=1e-05, momentum=0.1))    
    cnn.add(LeakyReLU(alpha=.1))
   
    cnn.add(Convolution2D(baseDim*62, (1,1),  strides = (1,1),use_bias=False, padding=padding))
    cnn.add(BatchNormalization(epsilon=1e-05, momentum=0.1))    
    cnn.add(Activation('linear'))
    
    cnn.add(GlobalAveragePooling2D())

    #cnn.add(Flatten())
    cnn.add(Dropout(dropout*3))
    cnn.add(Dense(200, activation=activation, kernel_regularizer=regularizers.l2(regularizer)))

    
    cnn.add(BatchNormalization(epsilon=1e-05, momentum=0.1))

    cnn.add(Dense(y_train.shape[1], activation="softmax"))
    return cnn