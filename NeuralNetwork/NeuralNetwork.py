#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    gradDescent()

    #Plotting statements
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.grid()
    plt.show()

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):
    return np.maximum(x, 0)

#Accepts matrix with rows of o vectors
def softmax(x):
    #Prevent overflow by normalizing
    x -= np.amax(x, axis=1)[:, None]
    return np.exp(x) / np.sum(np.exp(x), axis=1)[:,None]

def computeLayer(X, W, b):
    return W.dot(X) + b[:, None]

def CE(target, prediction):
    #Avoid taking unnecessary logs by doing row-wise sum of the matrix product
    pred_of_tar = np.sum(target*prediction, axis=1)
    return (-1/target.shape[0])*np.sum(np.log(pred_of_tar))

def gradCE(target, prediction):
    return softmax(prediction) - target

def accuracy(target, prediction):
    N = target.shape[0]
    return np.sum(target[np.arange(N), np.argmax(prediction, axis=1)])/N
    
def gradDescent():
    H = 1000
    K = 10
    F = 784
    EPOCHS = 200
    gamma = 0.9
    alpha = 0.1
    N = 10000
    
    Data = loadData()
    data = Data[0].reshape(N, 784)
    validData = Data[1].reshape(6000, 784)
    Targets = convertOneHot(Data[3], Data[4], Data[5])
    target = Targets[0]
    validTarget = Targets[1]

    del Data
    del Targets
    
    Wo = np.random.normal(0, np.sqrt(2/H), (K, H))
    Wh = np.random.normal(0, np.sqrt(2/F), (H, F))
    bo = np.zeros(K)
    bh = np.zeros(H)
    
    VWo = np.full((K, H), 1e-5)
    VWh = np.full((H, F), 1e-5)
    Vbo = np.full(K, 1e-5)
    Vbh = np.full(H, 1e-5)

    xpoints = np.arange(1, EPOCHS+1)
    ytrain = []
    yvalid = []

    def forward_propagation(training):
        nonlocal data, Wh, bh, Wo, bo
        if training:
            Sh = computeLayer(data.T, Wh, bh)
        else:
            Sh = computeLayer(validData.T, Wh, bh)
        Xh = relu(Sh)
        So = computeLayer(Xh, Wo, bo)
        Xo = softmax(So.T)

        return Sh, Xh, So, Xo
    
    for epoch in range(EPOCHS): 
        #Forward propagation
        Sh, Xh, So, Xo = forward_propagation(True)
        
        #Gradients/backpropagation
        grad_bo = gradCE(target, So.T)
        grad_Wo = grad_bo.T.dot(Xh.T)/N
        grad_bo = np.sum(grad_bo, axis=0)/N
        grad_bh = (Wo.T.dot(Xo.T) - Wo.T.dot(target.T))*np.array(Xh, dtype=bool)
        grad_Wh = grad_bh.dot(data)/N
        grad_bh = np.sum(grad_bh, axis=1)/N
        
        #Gradient descent with momentum
        VWo = gamma*VWo + alpha*grad_Wo
        VWh = gamma*VWh + alpha*grad_Wh
        Vbo = gamma*Vbo + alpha*grad_bo
        Vbh = gamma*Vbh + alpha*grad_bh
        
        #Update Weights and biases
        Wo -= VWo
        Wh -= VWh
        bo -= Vbo
        bh -= Vbh

        ytrain.append(accuracy(target, Xo))
        yvalid.append(accuracy(validTarget, forward_propagation(False)[3]))
        print(ytrain[epoch])

    plt.plot(xpoints, ytrain, label = "Training")
    plt.plot(xpoints, yvalid, label = "Validation")

if __name__ == '__main__':
    main()