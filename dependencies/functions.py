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

def plotConfusionMatrix(predictions, true_labels, labels):
    k = true_labels.shape[1]
    n = true_labels.shape[0]
    confusion_matrix = np.zeros((k,k))

    for l in range(n):
        decision = np.zeros(k)
        j = np.argmax(predictions[l])
        decision[j] = 1
        i = np.argmax(true_labels[l])
        confusion_matrix[i,j] +=1
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
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
        
def train_test_creator(dic, unknownClass, depth, with_unknown = True, test_size = 0.2, scalerType='standard', unknown_percentage = 0.1):
    X = []
    Y = []
    labelList = []
    last_class = 0
    #create X and Y with corresponding index
    if type(dic)==dict:
        length = len(dic)
        for count, key in enumerate(dic):
            tmp = dic[key]
            label = np.array(count)
            labelList.append(key)
            last_class = count
            label = np.resize(label, (tmp.shape[0],1))
            X.append(tmp)
            Y.append(label)
    else:
        return -1
    
    if with_unknown:
        tot_unk = 0
        for key in unknownClass:
            length = unknownClass[key].shape[0]        
            toUnk = round(length*unknown_percentage)
            tot_unk += toUnk
            X.append(unknownClass[key][rnd.sample(range(length),toUnk)])
        label = np.array(last_class+1)
        labelList.append('unknown')
        label = np.resize(label, (tot_unk,1))
        Y.append(label)
    
    #transform X and Y (lists) in ndarray 
    X = np.vstack(X)
    Y = np.vstack(Y)
    #transform Y into 1-hot array
    Y = np_utils.to_categorical(Y, np.max(Y)+1)
    #split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    
    #reshape for conv2d layers
    x_train = np.reshape(x_train, ( x_train.shape[0], x_train.shape[1], x_train.shape[2], depth))
    x_test = np.reshape(x_test, ( x_test.shape[0], x_test.shape[1], x_test.shape[2],depth))
    
    scalers = findScaler(x_train, scalerType)
    for i in scalers:
        print(i.get_params())
    #scale data
    scale(x_train, scalers)
    scale(x_test, scalers)
    
    #save used data for hyperas use
    with open('variables/train_test_split.pkl', 'wb') as f:  
        pickle.dump(x_train, f)
        pickle.dump(y_train, f)
        pickle.dump(x_test, f)
        pickle.dump(y_test, f) 
    with open('variables/labelList.pkl', 'wb') as f:  
        pickle.dump(labelList, f)
        
    return x_train, y_train, x_test, y_test, labelList 


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
        x_train = pickle.load(f)
        y_train = pickle.load(f)
        x_test = pickle.load(f)
        y_test = pickle.load(f) 
    return x_train, y_train, x_test, y_test 

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