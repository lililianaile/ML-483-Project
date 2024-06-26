import numpy as np

import sklearn.model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy
import sklearn

import torchvision

import matplotlib.pyplot as plt
from IPython import display
import time

display.set_matplotlib_formats('svg')

adevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

downloadYN = input("Do you need to download the EMNIST dataset? (y/n): ")
downloadTF = False if downloadYN == "n" else True

dataset = torchvision.datasets.EMNIST(root='emnist', split='letters', download=downloadTF) #download dataset

print(dataset.classes)
print(str(len(dataset.classes)) + ' classes') # dataset has 27 classes

print('Data size:')
print(dataset.data.shape) # data shape has 124800 letters, 28x28 images

images = dataset.data.view([124800, 1, 28, 28]).float() # convert to 4D tensor data
# 124800 letter iamges, 1 image channel, 28x28 images
print('Tensor data:')
print(images.shape)

# no data with class 'N/A'
print(torch.sum(dataset.targets==0))

torch.unique(dataset.targets)

dataset.class_to_idx

# eliminate the 1st class 'N/A'
letterCategories = dataset.classes[1:]

# re-label all the other entries, subtracting 1 to realign them
labels = copy.deepcopy(dataset.targets)-1
print(labels.shape)

print(torch.sum(labels==0))
print('labels:')
print(torch.unique(labels))

# # Need to normalize image data, right now pixel values are 0-255
# plt.hist(images[:10,:,:,:].view(1,-1).detach(),40)
# plt.title('Raw values')
# plt.show()

# Divide by max value in images
images /= torch.max(images)

# # Pixel values are now between 0-1
# plt.hist(images[:10,:,:,:].view(1,-1).detach(),40)
# plt.title('After normalization')
# plt.show()

# # show random sample images
# fig,axs = plt.subplots(3,7,figsize=(13,6))

# for i, ax in enumerate(axs.flatten()):

#     randompicture = np.random.randint(images.shape[0])

#     # extract the image and what letter it is
#     I = np.squeeze(images[randompicture,:,:])
#     letter = letterCategories[labels[randompicture]]

#     # visualize the letter
#     ax.imshow(I.T,cmap='gray')
#     ax.set_title('The letter "%s"' %letter)
#     ax.set_xticks([])
#     ax.set_yticks([])

# plt.show()

# split arrays into training/testing datasets
train_data, test_data, train_labels, test_labels = sklearn.model_selection.train_test_split(images, labels, test_size=.1)

# convert arrays into pytorch dataset
train_data = TensorDataset(train_data, train_labels)
test_data = TensorDataset(test_data, test_labels)


setbatchsize = input("Enter batch size (8, 16, 32, 64, 128) [Higher batch size = Faster, but more CPU]: ")

batchsize = int(setbatchsize)
# create DataLoader objects that handle the data, batch size, and shuffling between epochs
train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

print("trainloader shapes")
print(train_loader.dataset.tensors[0].shape)
print(train_loader.dataset.tensors[1].shape)


def createNN(printtoggle=False):

    class emnistnet(nn.Module):
        def __init__(self, printtoggle):
            super().__init__()

            self.print = printtoggle

            # convolutional layers: extract key features like edges, textures, and shapes
            self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
            self.bnorm1 = nn.BatchNorm2d(6) # batch normalization: can accelerate training convergence and leads to improved generalization performance

            self.conv2 = nn.Conv2d(6, 6, 3, padding=1)
            self.bnorm2 = nn.BatchNorm2d(6)

            # fully connected layers are responsible for taking the high-level features extracted by the convolutional layers and transforming them into the final output of the network
            self.fc1 = nn.Linear(7*7*6, 50)
            self.fc2 = nn.Linear(50, 26) # reshapes the original output down to the 26 letters we have
        
        def forward(self, x):
            if self.print: 
                print(f'Input: {list(x.shape)}')
            
            # convolutional layer -> max pool -> batch normalization -> relu
            x = F.max_pool2d(self.conv1(x), 2)
            x = F.leaky_relu(self.bnorm1(x)) # leaky relu prevents neurons from dying completely, allowing the nn to handle more complex data
            if self.print:
                print(f'First CPR block: {list(x.shape)}')
            
            x = F.max_pool2d(self.conv2(x), 2)
            x = F.leaky_relu(self.bnorm2(x))
            if self.print:
                print(f'Second CPR block: {list(x.shape)}')
            
            nUnits = x.shape.numel()/x.shape[0]
            x = x.view(-1, int(nUnits))
            if self.print:
                print(f'Vectorized: {list(x.shape)}')
            
            x = F.leaky_relu(self.fc1(x))
            x = self.fc2(x)
            if self.print:
                print(f'Final output: {list(x.shape)}')
            
            return x
        
    
    net = emnistnet(printtoggle)

    lossfunction = nn.CrossEntropyLoss() # cross entropy loss works with data classification, compares predicted values to true values and penalizes NN based on discrepancy

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001) # optimization algorithm that will adjust the learning rate during training as needed

    return net, lossfunction, optimizer

net, lossfunction, optimizer = createNN(True)

X, Y = next(iter(train_loader))
yHat = net(X)

print('\nOutput Size:')
print(yHat.shape)

loss = lossfunction(yHat, torch.squeeze(Y))
print('\nLoss:')
print(loss)

# A function that trains the model
def trainModel(epochs, printing):
    starttime = int(time.time())
    print(f"Start Time: {starttime}")

    print("training model function called...")
    device = adevice
    print("device set")
    numepochs = epochs
    print(f"set epochs: {numepochs}")

    # create new model
    print("creating model")
    net, lossfunction, optimizer = createNN()
    print("model created")

    print("directing model to device")
    net.to(device)

    trainLoss = torch.zeros(numepochs)
    testLoss = torch.zeros(numepochs)
    trainErr = torch.zeros(numepochs)
    testErr = torch.zeros(numepochs)

    # loop over epochs
    for epochi in range(numepochs):
        print(f"NEW EPOCH: {epochi}")
        net.train()
        #print("train function finished")
        batchLoss = []
        batchErr = []
        for X, Y in train_loader:
            #print("train loader for loop")
            #print(f"X: {X}, Y: {Y}")
            #print(f"train_loader: {train_loader}")
            X = X.to(device)
            Y = Y.to(device)
            #print("set to device")

            #print("1")
            yHat = net(X)
            #print("2")
            loss = lossfunction(yHat, Y)

            #print("3")
            optimizer.zero_grad()
            #print("4")
            loss.backward()
            #print("5")
            optimizer.step()

            #print("6")
            batchLoss.append(loss.item())
            #print("7")
            batchErr.append(torch.mean((torch.argmax(yHat, axis=1) != Y).float()).item())

            #print("8")

        # After each epoch, calculate train loss and error
        trainLoss[epochi] = np.mean(batchLoss)
        trainErr[epochi] = 100 * np.mean(batchErr)

        # Calculate test loss and error after each epoch
        net.eval()
        with torch.no_grad():
            X, Y = next(iter(test_loader))
            X = X.to(device)
            Y = Y.to(device)

            yHat = net(X)
            loss = lossfunction(yHat, Y)

        testLoss[epochi] = loss.item()
        testErr[epochi] = 100 * torch.mean((torch.argmax(yHat, axis=1) != Y).float()).item()

        if printing:
            print(f"Epoch: {epochi}")
            print(f"testLoss: {testLoss[epochi]}")
            print(f"testErr: {testErr[epochi]}")

    # end of epochs
    print("end of epochs")
    endtime = int(time.time())
    print(f"End Time: {endtime}")
    elapsedtime = endtime - starttime
    elapsedmin = int(elapsedtime / 60)
    print(elapsedmin)

    elapsedsec = elapsedtime % 60
    print(elapsedsec)
    print(f"Elapsed time: {elapsedmin} minutes, {elapsedsec} seconds.")

    # function output
    return trainLoss, testLoss, trainErr, testErr, net


# USAGE
numepochs = int(input("Enter number of epochs: "))
printstatementsTF = False
printstatements = input("Do you want to print statements during training? (y/n): ")
printstatementsTF = True if printstatements == "y" else False
trainLoss, testLoss, trainErr, testErr, net = trainModel(numepochs, printstatementsTF)

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

ax[0].plot(trainLoss, 's-', label="Train")
ax[0].plot(testLoss, 'o-', label="Test")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss (MSE)")
ax[0].set_title('Model loss')

ax[1].plot(trainLoss, 's-', label="Train")
ax[1].plot(testLoss, 'o-', label="Test")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Error rates (%)")
ax[1].set_title(f'Final model test error rate: {testErr[-1]:.2f}%')
ax[1].legend()

plt.show()

# Visualize some images
device = adevice
X, Y = next(iter(test_loader))
X = X.to(device)
Y = Y.to(device)
yHat = net(X)

randomexamples = np.random.choice(len(Y), size=21, replace=False)

fig, axs = plt.subplots(3, 7, figsize=(15,6))

for i, ax in enumerate(axs.flatten()):
    I = np.squeeze(X[randomexamples[i],0,:,:]).cpu()
    trueLetter = letterCategories[Y[randomexamples[i]]]
    predLetter = letterCategories[torch.argmax(yHat[randomexamples[i],:])]

    if trueLetter == predLetter:
        col = 'gray'
    else:
        col = 'hot'
    
    ax.imshow(I.T, cmap=col)
    ax.set_title(f'True {trueLetter}, predicted {predLetter}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()