import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,RobustScaler
from keras.utils import np_utils
import random as rnd
## division for train and test
from sklearn.model_selection import train_test_split
import pickle
from keras.activations import softmax
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D, MaxPooling2D, AveragePooling2D, BatchNormalization

from IPython.display import clear_output, Image, display, HTML

from sklearn.metrics import confusion_matrix
import itertools

import hashlib
import re
import os
import librosa
import random as rnd
import re

rnd.seed(1)

def stackData(inputData, augmentedData, iterate):
    if isinstance(inputData, list):
         for count, layer in enumerate(inputData):
            inputData[count] = np.vstack((inputData[count], augmentedData[count][iterate]))
    else:
        inputData = np.vstack((inputData, augmentedData[iterate]))
    return inputData

def meltData(inputData, augmentedData, inputLabel, augmentedLabel, percentage):    
    if isinstance(augmentedData, list):
        size = augmentedData[0].shape[0]
    else:
        size = augmentedData.shape[0]
    
    toAugment = round(size * percentage)
    iterate = rnd.sample(range(size),toAugment)
    
    inputData = stackData(inputData, augmentedData, iterate)
    inputLabel = stackData(inputLabel, augmentedLabel, iterate)
    return inputData, inputLabel

def getName(index):
    if index == 0:
        return 'Train'
    elif index == 1:
        return 'Test'
    elif index == 2:
        return 'Validation'
    elif index == 3:
        return 'AugmentedTrain'
    
def modelSelection(model, data, labels):
    models = ['model1', 'model2', 'tinyDarknet', 'SiSoInc', 'MiSoInc', 'SiMoInc']
    if model == models[0] or model == models[1] or model == models[2] or model == models[3]:
        validation_data=({'input_input': data['Validation']}, {'output': labels['Validation']})
        inputData = data['Train']
        inputLabel = labels['Train']
        testData = data['Test']
        testLabel = labels['Test']
        validData = data['Validation']
        validLabel = labels['Validation']
        augmentedData = data['AugmentedTrain']
        augmentedLabel = labels['AugmentedTrain']
        loss_weights={'output': 1.}
    elif model == models[4]:
        validation_data=({'first': data['Validation'][:,:,:,0, np.newaxis], 'second': data['Validation'][:,:,:,1, np.newaxis], 'third': data['Validation'][:,:,:,2, np.newaxis]}, {'output': labels['Validation']})
        inputData = [data['Train'][:,:,:,0, np.newaxis], data['Train'][:,:,:,1, np.newaxis], data['Train'][:,:,:,2, np.newaxis]]
        inputLabel = labels['Train']
        testData = [data['Test'][:,:,:,0, np.newaxis], data['Test'][:,:,:,1, np.newaxis], data['Test'][:,:,:,2, np.newaxis]]
        testLabel = labels['Test']
        validData = [data['Validation'][:,:,:,0, np.newaxis], data['Validation'][:,:,:,1, np.newaxis], data['Validation'][:,:,:,2, np.newaxis]]
        validLabel = labels['Validation']
        augmentedData = [data['AugmentedTrain'][:,:,:,0, np.newaxis], data['AugmentedTrain'][:,:,:,1, np.newaxis], data['AugmentedTrain'][:,:,:,2, np.newaxis]]
        augmentedLabel = labels['AugmentedTrain']
        loss_weights={'output': 1.}
    elif model == models[5]:
        inputData = data['Train']
        inputLabel = [labels['Train'], labels['Train'], labels['Train']]
        validation_data=({'input': data['Validation']}, {'output0': labels['Validation'], 'output1': labels['Validation'], 'output2': labels['Validation']})
        testData = data['Test']
        testLabel = [labels['Test'], labels['Test'], labels['Test']]
        validData = data['Validation']
        validLabel = [labels['Validation'], labels['Validation'], labels['Validation']]
        augmentedData = data['AugmentedTrain']
        augmentedLabel  = [labels['AugmentedTrain'], labels['AugmentedTrain'], labels['AugmentedTrain']]
        loss_weights={'output0': 1., 'output1': 1., 'output2' : 1.}
    return inputData, inputLabel, testData, testLabel, validData, validLabel, augmentedData, augmentedLabel, validation_data, loss_weights

def plotHistory(epochs, fitted, title):
    top3 = re.compile('.*top3*')
    acc = re.compile('.*acc')
    figs, axs = plt.subplots(1,3, figsize=(10, 5))
    for key in fitted[0].history:
        if top3.match(key):            
            ax = axs[0]
            ax.set_title('Top3 accuracy')
        elif acc.match(key):
            ax = axs[1]
            ax.set_title('Accuracy')
        else:
            ax = axs[2]
            ax.set_title('Loss')
        y = []
        for count, fit in enumerate(fitted):
            y.extend(fit.history[key]) 
            ax.axvline(x=len(y), color = 'red')
        ax.plot(range(1, len(y)+1),y,label=key)     
    plt.suptitle('Training history')
    for ax in axs:
        legend = ax.legend(loc='best', shadow=True)
    
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width
        
    plt.show()


def which_set(filename, validation_percentage, testing_percentage, totClass):
    """Determines which data partition the file should belong to.
    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.
    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.
    Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.
    Returns:
    String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                      (totClass + 1)) *
                     (100.0 / totClass))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result

#shift signal of #value samples padding with zero to return the same dimension
def shiftVec(signal, value):
    initial_length = signal.shape[0]
    padded = np.pad(signal, (abs(value),abs(value)), 'constant', constant_values=0)
    signal = padded[abs(value)-value:abs(value)+initial_length-value]
    return signal
#return a random noise of nSample
def noiseSelector(noise, nSample):
    length = len(noise)
    choice = rnd.randint(0, length-1)
    key = list(noise.keys())[choice]
    start = rnd.randint(0, noise[key].shape[0]-nSample-1)
    return noise[key][start:start+nSample]

## Return the word between two string starting from left
def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

# stretching the sound
def stretch(data, rate=1):
    input_length = data.shape[0]
    data = librosa.effects.time_stretch(data, rate)
    if len(data)>input_length:
        data = data[round((data.shape[0]-input_length)/2):round((data.shape[0]+input_length)/2)]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant", constant_values=0)

    return data

#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(preds, y_true, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          plot = True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    a = []
    for i in preds:
        a.append(np.argmax(i)+1)
    b = []
    for i in y_true:
        b.append(np.argmax(i)+1)
        
    cm = confusion_matrix(a, b)
    np.set_printoptions(precision=2)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    fig, ax = plt.subplots()
    
    #print(cm)
    if plot:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    return cm
    
def findScaler(x, scalerType='standard'):
    #initialize the scaler    
    scalers = []
    for i in range(x.shape[3]):
        temp = []
        if scalerType == 'robust':
            scaler = RobustScaler()
        elif scalerType == 'standard':
            scaler = StandardScaler()
        for count, sample in enumerate(x):
            tmp = np.reshape(sample[:,:,i],(sample.shape[0]*sample.shape[1])).reshape(1, -1)
            temp.append(tmp)    
        temp = np.vstack(temp)
        scaler.fit(temp)
        scalers.append(scaler)
    return scalers    

def scale(x, scalers):
    #scaling data with the trained scaler  
    for i in range(x.shape[3]):
        temp = []
        for count, sample in enumerate(x):
            tmp = np.reshape(sample[:,:,i],(sample.shape[0]*sample.shape[1])).reshape(1, -1)
            temp.append(tmp)    
        temp = np.vstack(temp)
        temp = scalers[i].transform(temp)
        for count, sample in enumerate(temp):
            x[count,:,:,i] = np.reshape(sample,(x.shape[1],x.shape[2])) 
    return x
        
def train_test_creator(data, used, unknown, depth, with_unknown = True, scalerType='standard', unknown_percentage = 0.1):
    dataset = {}
    labels = {}
    for group in data:
        X = []
        Y = []
        #create X and Y with corresponding index
        length = len(data[group])
        count = 0
        for key in used:
            tmp = data[group][key]
            label = np.array(count)
            count += 1
            label = np.resize(label, (tmp.shape[0],1))
            X.append(tmp)
            Y.append(label)
        if group != 'AugmentedTrain':
            if with_unknown:
                tot_unk = 0
                for key in unknown:
                    length = data[group][key].shape[0]        
                    toUnk = round(length*unknown_percentage)
                    tot_unk += toUnk
                    X.append(data[group][key][rnd.sample(range(length),toUnk)])
                label = np.array(count)
                label = np.resize(label, (tot_unk,1))
                Y.append(label)

        #transform X and Y (lists) in ndarray 
        X = np.vstack(X)
        Y = np.vstack(Y)
        #transform Y into 1-hot array
        if group == 'AugmentedTrain' and with_unknown:
            labels[group] = np_utils.to_categorical(Y, np.max(Y)+2)
        else:
            labels[group] = np_utils.to_categorical(Y, np.max(Y)+1)
        

        #reshape for conv2d layers
        dataset[group] = np.reshape(X, ( X.shape[0], X.shape[1], X.shape[2], depth))
    
    if scalerType == 'robust' or scalerType == 'standard' :
        scalers = findScaler(dataset['Train'], scalerType)
        #scale data
        for group in dataset:
            dataset[group] = scale(dataset[group], scalers)
    if with_unknown:
        used.append('unknown')
    #save used data for hyperas use
    with open('variables/train_test_split.pkl', 'wb') as f:  
        pickle.dump(dataset, f)
        pickle.dump(labels, f)
    with open('variables/labelList.pkl', 'wb') as f:  
        pickle.dump(used, f)
    return dataset

def load_dataset():
    #load used data
    with open('variables/train_test_split.pkl', 'rb') as f: 
        dataset = pickle.load(f)
        labels = pickle.load(f)
    return dataset, labels

       
#show net
def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

    
#Hyperas
def data():
    #load used data
    with open('variables/train_test_split.pkl', 'rb') as f: 
        dataset = pickle.load(f)
        labels = pickle.load(f)
    return dataset['Train'], labels['Train'], dataset['Test'], labels['Test']
def create_model(x_train, y_train, x_test, y_test):
    activation = 'softplus'
    minim = {{choice([8,16,20,24,32,30,46,50,64])}}
    padding = 'same'
    cnn = Sequential()

    cnn.add(Convolution2D(minim, (4,2),  strides = (1,1), padding="valid", 
                          input_shape=(x_train.shape[1], x_train.shape[2],1)))
    cnn.add(Activation(activation))

    cnn.add(Convolution2D(minim * 2, (2,2),  strides = (1,1), padding=padding))
    cnn.add(Activation(activation))


    cnn.add(Convolution2D(minim*4, (2,2),  strides = (1,1), padding=padding))
    cnn.add(Activation(activation))

    cnn.add(BatchNormalization())

    cnn.add(MaxPooling2D(pool_size=(4,2)))

    cnn.add(Dropout(0.4))
    cnn.add(BatchNormalization())
    cnn.add(Convolution2D(minim * 2, (2,2),  strides = (1,1), padding=padding ))
    cnn.add(Activation(activation))

    #cnn.add(Dropout(0.2))
    cnn.add(Convolution2D(minim * 4, (2,2),  strides = (1,1), padding=padding))
    cnn.add(Activation(activation))
    cnn.add(BatchNormalization())

    cnn.add(MaxPooling2D(pool_size=(2,2)))

    cnn.add(Dropout(0.4))
    cnn.add(BatchNormalization())
    cnn.add(Convolution2D(minim *8, (2,2),  strides = (1,1), padding=padding))
    cnn.add(Activation(activation))



    cnn.add(MaxPooling2D(pool_size=(4,2)))

    cnn.add(Dropout(0.3))


    cnn.add(Flatten())

    cnn.add(Dense(80, activation=activation))

    cnn.add(Dropout(0.5))
    cnn.add(BatchNormalization())

    cnn.add(Dense(y_train.shape[1], activation="softplus"))

    cnn.compile(loss="categorical_crossentropy", optimizer="adamax", metrics=['accuracy'])
            
    cnn.fit(x_train, y_train,
              batch_size={{choice([128, 256])}},
              epochs=30,
              verbose=2,
              validation_data=(x_test, y_test))
    score, acc = cnn.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': cnn}