from keras.activations import softmax, linear
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D, MaxPooling2D, AveragePooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.constraints import maxnorm
from keras import regularizers


def model1(x_train, y_train, baseDim = 64, activation = "softplus", padding = "valid"):

    cnn = Sequential()

    cnn.add(Convolution2D(baseDim, (7,4),  strides = (1,1), use_bias=False, padding=padding,  activation=activation,
                          input_shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3]), name = 'input'))
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

    cnn.add(Dense(y_train.shape[1], activation="sigmoid", name = 'output'))
    return cnn
    
    
def model2(x_train, y_train, baseDim = 16, activation = "softplus", padding = "same"):
    cnn = Sequential()

    cnn.add(Convolution2D(baseDim, (4,2),  strides = (1,1), padding="valid", 
                              input_shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3]),  name = 'input'))
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

    cnn.add(Dense(y_train.shape[1], activation="softmax", name = 'output'))
    return cnn


def tinyDarknet(x_train, y_train, baseDim = 16, activation = "softplus", padding = "same", dropout = 0.1, regularizer = 0.01):
    cnn = Sequential()
    #cnn.add(Dropout(dropout))
    cnn.add(Convolution2D(baseDim*2, (3,3),  strides = (1,1), use_bias=False, padding=padding, 
                          input_shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3]), name = 'input'))
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

    cnn.add(Dense(y_train.shape[1], activation="softmax", name = 'output'))
    return cnn

##########################################
from keras.layers import Input, SpatialDropout2D, concatenate
from keras.models import Model

def cBN(inputLayer, filt = 64, size = (1,1), padding = 'same', activation = 'relu', regu = 0.0, dropout = 0.05, strides = (1,1)):
    cbn = Convolution2D(filt, size, padding=padding, use_bias=False, kernel_regularizer=regularizers.l2(regu), strides = (1,1))(inputLayer)
    cbn = Activation(activation)(cbn)
    #cbn = BatchNormalization(epsilon=1e-05, momentum=0.1, axis=-1)(cbn)
    return cbn

def inception(inputLayer, filt = 64, base = 3):
    tower_1 = cBN(inputLayer, filt = filt)   
    tower_1 = cBN(tower_1, size = (base,base), filt = filt)

    tower_2 = cBN(inputLayer, filt = filt)
    tower_2 = cBN(tower_2, size = (base+2,base+2), filt = filt)
    
    tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(inputLayer)
    tower_3 = cBN(tower_3, filt = filt)
    
    output = concatenate([tower_1, tower_2, tower_3], axis = 3)
    return output

def SiSoInception(x_train, y_train, baseDim = 10, activation = "softplus", dropout = 0.15):
    input_img = Input(name = 'input_input', shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]))

    output = Convolution2D(120, (3,round(x_train[0].shape[2])))(input_img)
    output = lin_inception(output, filt = 80, base = 3)

    output = MaxPooling2D(pool_size=(3,1), padding='same')(concat)
    output = lin_inception(output, filt = baseDim, base = 3)
    
    output = AveragePooling2D((11,1))(output)
    output = Flatten()(output)

    output = Dense(70)(output)
    output = Activation('relu')(output)
    output = Dropout(dropout)(output)

    output = Dense(y_train.shape[1], name = 'output', activation='softmax')(output)
    cnn = Model(inputs = [first_input, second_input, third_input], outputs = output)
    return cnn

#################################
def lin_inception(inputLayer, filt = 64, base = 3):
    tower_1 = cBN(inputLayer, filt = filt)   
    tower_1 = cBN(tower_1, size = (base,1), filt = filt)

    tower_2 = cBN(inputLayer, filt = filt)
    tower_2 = cBN(tower_2, size = (base+1,1), filt = filt)

    tower_3 = cBN(inputLayer, filt = filt)
    tower_3 = cBN(tower_3, filt = filt, size = (base+2,1))
    
    output = concatenate([tower_1, tower_2, tower_3], axis = 3)
    return output


def singleInputMFCC(x_train, name, mfcc = True):
    single_input = Input(name = name, shape = (x_train[0].shape[1], x_train[0].shape[2], 1))
    if mfcc:
        output = Convolution2D(120, (3,round(x_train[0].shape[2])))(single_input)
        output = lin_inception(output, filt = 80, base = 3)
    else:
        output = Convolution2D(16, (1,1), padding = "same")(single_input)
        output = LeakyReLU(alpha=.1)(output)
        output = Convolution2D(32, (3,3), padding = "same")(output)
        output = LeakyReLU(alpha=.1)(output)
        output = MaxPooling2D(pool_size=(2,2))(output)
        output = Convolution2D(16, (1,1), padding = "same")(output)
        output = LeakyReLU(alpha=.1)(output)
        output = Convolution2D(64, (3,3), padding = "same")(output)
        output = LeakyReLU(alpha=.1)(output)
        output = Convolution2D(16, (1,1), padding = "same")(output)
        output = LeakyReLU(alpha=.1)(output)
        output = Convolution2D(64, (3,3), padding = "same")(output)
        output = LeakyReLU(alpha=.1)(output)
        output = MaxPooling2D(pool_size=(2,2))(output)
        output = Convolution2D(16, (1,1), padding = "same")(output)
        output = LeakyReLU(alpha=.1)(output)
        output = Convolution2D(128, (3,3), padding = "same")(output)
        output = LeakyReLU(alpha=.1)(output)
        output = Convolution2D(16, (1,1), padding = "same")(output)
        output = LeakyReLU(alpha=.1)(output)
        output = Convolution2D(128, (3,3), padding = "same")(output)
        output = LeakyReLU(alpha=.1)(output)    
    return single_input, output

def firstConcat(x_train, mfcc = True):
    first, first_output = singleInputMFCC(x_train,'first', mfcc = mfcc)
    second, second_output = singleInputMFCC(x_train,'second', mfcc = mfcc)
    third, third_output = singleInputMFCC(x_train,'third', mfcc = mfcc)
    concat = concatenate([first_output, second_output, third_output], axis = 3)
    return first, second, third, concat
def MiSoInception(x_train, y_train, baseDim = 64, dropout = 0.3, mfcc = True):
    first_input, second_input, third_input, concat = firstConcat(x_train, mfcc = mfcc)

    if mfcc:
        output = MaxPooling2D(pool_size=(3,1), padding='same')(concat)
        output = lin_inception(output, filt = baseDim, base = 3)
        output = AveragePooling2D((11,1))(output)
        output = Flatten()(output)
    else:
        output = MaxPooling2D(pool_size=(5,3), padding='same')(concat)
        output = inception(output, filt = baseDim, base = 2)
        output = GlobalAveragePooling2D()(output)

    output = Dense(70)(output)
    output = Activation('relu')(output)
    output = Dropout(dropout)(output)

    output = Dense(y_train.shape[1], name = 'output', activation='softmax')(output)
    cnn = Model(inputs = [first_input, second_input, third_input], outputs = output)
    return cnn

#######################

def extraClassifier(name, inputLayer, outputShape):
    output0 = GlobalAveragePooling2D()(inputLayer)
    output0 = Dense(100, activation = 'relu')(output0)
    output0 = Dropout(0.7)(output0)
    output0 = Dense( outputShape, name = name, activation='softmax')(output0)
    return output0

def SiMoInception(x_train, y_train, baseDim = 64, dropout = 0.4):
    input_img = Input(name = 'input', shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]))

    output = cBN(input_img)
    output = cBN(output, size = (3,3))
    output = cBN(output)
    output = cBN(output, size = (3,3))

    output = MaxPooling2D(pool_size=(3,2), padding='same')(output)

    output = inception(output, filt = baseDim)
    output0 = extraClassifier('output0', output, y_train[0].shape[1])

    output = MaxPooling2D(pool_size=(3,2), padding='same')(output)

    output = inception(output, filt = baseDim)
    output1 = extraClassifier('output1',output, y_train[0].shape[1])
    output = inception(output, filt = baseDim)

    output = GlobalAveragePooling2D()(output)

    output = Dense(90, activation = 'relu')(output)
    output = Dropout(dropout)(output)
    output2 = Dense(y_train[0].shape[1], name = 'output2', activation='softmax')(output)
    cnn = Model(inputs = input_img, outputs = [output2, output1, output0])
    return cnn