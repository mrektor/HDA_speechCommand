import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,RobustScaler
from keras.utils import np_utils
import random as rnd
## division for train and test
from sklearn.model_selection import train_test_split
import pickle

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
    if scalerType == 'robust':
        scaler = RobustScaler()
    elif scalerType == 'standard':
        scaler = StandardScaler()
    temp = []
    #online fit on all data reshaped as array
    for count, sample in enumerate(x):
        sample = np.reshape(sample,(sample.shape[0]*sample.shape[1])).reshape(1, -1)
        temp.append(sample)
    temp=np.vstack(temp)
    scaler.fit(temp)
    return scaler    

def scale(x, scaler):
    #scaling data with the trained scaler  
    temp = []
    for count, sample in enumerate(x):
        sample = np.reshape(sample,(sample.shape[0]*sample.shape[1])).reshape(1, -1)
        temp.append(sample)
    temp=np.vstack(temp)
    temp = scaler.transform(temp)
    for count, sample in enumerate(temp):
        x[count] = np.reshape(sample,(x.shape[1],x.shape[2],1))   
        
def train_test_creator(dic, unknownClass, with_unknown = True, test_size = 0.2, scalerType='standard', unknown_percentage = 0.1):
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
    x_train = np.reshape(x_train, ( x_train.shape[0], x_train.shape[1], x_train.shape[2],1))
    x_test = np.reshape(x_test, ( x_test.shape[0], x_test.shape[1], x_test.shape[2],1))
    
    scaler = findScaler(x_train, scalerType)
    
    #scale data
    scale(x_train, scaler)
    scale(x_test, scaler)
    
    #save used data for hyperas use
    with open('variables/train_test_split.pkl', 'wb') as f:  
        pickle.dump(x_train, f)
        pickle.dump(y_train, f)
        pickle.dump(x_test, f)
        pickle.dump(y_test, f) 
    with open('variables/labelList.pkl', 'wb') as f:  
        pickle.dump(labelList, f)
        
    return x_train, y_train, x_test, y_test, labelList 