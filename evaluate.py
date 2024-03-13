import argparse
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_curve
import torch.multiprocessing as mp
import os, random
from tqdm import tqdm
import torchvision
from sklearn.metrics import accuracy_score

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.manual_seed(12)

lab_path = os.getcwd()
data_folder = os.path.join(lab_path, 'audio')
labs_folder = os.path.join(lab_path, 'labels')
#data_folder = "/Users/dell/Desktop/assignments/Task4VAD/com4511/audio"
#labs_folder = "/Users/dell/Desktop/assignments/Task4VAD/com4511/labels"

train_prefixes = ["N", "V"]
valid_prefixes = ["E"]
test_prefixes = ["C"]

# dataset class for loading and preprocessing audio data for training, validation, and testing purposes.
class VoiceDataSet (Dataset):
    def __init__(self, dataset, transform=None, window_length=512):
        self.prefix = ["C"]
        # These prefixes are used later to generate file paths for training, validation, and testing data
        if dataset.lower () == "train":
            self.prefix = train_prefixes
        if dataset.lower () == "val":
            self.prefix = valid_prefixes
        if dataset.lower () == "test":
            print (dataset.lower ())
            self.prefix == ["C"]
        # generates file paths using the gen_paths method and calculates the length of the dataset
        self.paths = self.gen_paths (self.prefix)
        self.len = len (self.paths[0])
        self.win = window_length
        self.transform = transform

    # retrieves an item from the dataset based on the given index
    def __getitem__(self, idx):
        X, y = self.readpath (self.paths[0][idx], self.paths[1][idx])
        l = X.size (dim=0)
        # randomly selects a window of length
        i = random.randint (0, l - self.win)

        if self.win == 0:
            if self.transform:
                temp = X.view (1, -1, 13).permute (2, 0, 1)

                X = self.transform (temp)
                X = X.view (13, -1).permute (1, 0)
                return X, y.type (torch.float32)
            else:
                return X, y.type (torch.float32)

        X = X[i:i + self.win]
        y = y[i:i + self.win]
        # applies the specified transformation
        if self.transform:
            temp = X.view (1, -1, 13).permute (2, 0, 1)

            X = self.transform (temp)
            X = X.view (13, -1).permute (1, 0)

        return X, y.type(torch.float32)

    def __len__(self):
        return self.len

    # reads the audio and label data from the specified file paths
    def readpath(self, d_path, l_path):
        # convert it from numpy to pytorch tensosr
        with open (d_path, 'rb') as f:
            X = torch.from_numpy (np.load (f))

        with open (l_path, 'rb') as f:
            y = torch.from_numpy (np.load (f))

        return X, y

    def gen_paths(self, prefixes):
        # generates file paths for the specified prefixes by iterating over
        # the files in the data and labels folders
        os.chdir (data_folder)
        data_paths = [f"{data_folder}/{file}" for file in os.listdir () if file[0] in prefixes]

        os.chdir (labs_folder)
        labs_paths = [f"{labs_folder}/{file}" for file in os.listdir () if file[0] in prefixes]
        # return file paths containing data and labels
        return (data_paths, labs_paths)

def data_means_std(prefixes):
    ##calculates the standard deviation and mean values of the audio data
    data_paths, label_paths = VoiceDataSet.gen_paths (VoiceDataSet, prefixes=prefixes)
    data, label_data = [], []

    for d, label in zip (data_paths, label_paths):
        temp_d, temp_l = VoiceDataSet.readpath (VoiceDataSet, d, label)
        data.append (temp_d)
    # cat data
    total_data = torch.cat (data)
    # calculate mean and stds
    stds, means = torch.std_mean (total_data, dim=0)
    return stds, means

def evaluate(audio_dir, labels_dir):
    from tqdm import tqdm
    print("Test with audio directory:", audio_dir)
    print("Test with labels directory:", labels_dir)

    train_stds, train_means = data_means_std (train_prefixes)

    # creates a composition of transformations contain normalization transformation
    n_transform = transforms.Compose ([
        transforms.Normalize (train_means, train_stds)]
    )

    # it determines the size of the windows that will be randomly selected from the audio data
    window_length = 512
    b_size = 2

    # get test data
    VoiceSet = VoiceDataSet ("test", window_length=0, transform=n_transform)
    test_loader = DataLoader (VoiceSet, shuffle=True, batch_size=1)

    VoiceSet2 = VoiceDataSet ("val", window_length=0, transform=n_transform)
    val_loader = DataLoader (VoiceSet, shuffle=True, batch_size=1)

    # the recurrent neural network (RNN) with LSTM cells
    class LSTM_network (nn.Module):
        def __init__(self, hidden_size=13, hidden_layers=1, bidirectional=False):
            super (LSTM_network, self).__init__ ()
            # hidden layer size is 13, and the number of hidden layers is 1 by default
            self.lstm = nn.LSTM (input_size=13, hidden_size=hidden_size, num_layers=hidden_layers, batch_first=True)
            # The dimensional order of the input data is is (batch_size, seq_length, input_size)
            self.fc1 = nn.Linear (in_features=hidden_size, out_features=1)
            self.sigmoid = nn.Sigmoid ()
            # choose bidirection or single direction
            self.bidirectional = bidirectional

        def forward(self, x):
            # The output output is the hidden state of the LSTM at each time step, and the hidden state at the final time step
            output, _status = self.lstm (x)
            # output is passed to the linear layer self.fc1 for linear transformation
            output = self.fc1 (output)
            # activition
            output = self.sigmoid (output)

            return output

    # training løøp
    # An instance of LSTM_network is created and some parameters are passed in
    model = LSTM_network (hidden_size=26, hidden_layers=2, bidirectional=True)
    model.load_state_dict (torch.load ('LSTM_model.pth'))
    model.eval ()
    # BCELoss is a common loss function to binary classfication
    crit = nn.BCELoss ()
    # Set the weight decay parameter when initializing the optimizer
    weight_decay = 1e-5
    optimiser = optim.Adam (model.parameters (), lr=1e-4, betas=[0.9, 0.99], weight_decay=weight_decay)

    eps = 1e-7
    epochs = 10
    losses = [0.1]

    # obtain the validation dataset (val_X and val_y) for validating the model
    test_win_len = 8096
    # get next batch of data(tensor of data and labels)

    print ('LSTM')
    print ('test')
    #Loop for inference and performance evaluation on test sets
    for i, (X, y) in enumerate (test_loader):
        with torch.no_grad ():
            y_pred = model (X)
            # Use the np.squeeze() function to reduce it from a tensor of shape (1, seq_length, 1) to a one-dimensional array of shape (seq_length,).
            y_pred = np.array (y_pred.squeeze ())
            y = np.array (y.squeeze ())

            # Calculate the False Positive Rate, True Positive Rate and Threshold of the ROC curve
            fpr, tpr, threshold = roc_curve (y, y_pred, pos_label=1)
            fnr = 1 - tpr
            # Find the threshold of the Equal Error Rate based on the difference between the fnr and tpr
            eer_threshold = threshold[np.nanargmin (np.absolute ((fnr - fpr)))]
            # find the index with the smallest absolute value of the difference
            EER = fpr[np.nanargmin (np.absolute ((fnr - fpr)))]

            print (EER)

    print ('val')
    for i, (X, y) in enumerate (val_loader):
        with torch.no_grad ():
            y_pred = model (X)
            # Use the np.squeeze() function to reduce it from a tensor of shape (1, seq_length, 1) to a one-dimensional array of shape (seq_length,).
            y_pred = np.array (y_pred.squeeze ())
            y = np.array (y.squeeze ())

            # Calculate the False Positive Rate, True Positive Rate and Threshold of the ROC curve
            fpr, tpr, threshold = roc_curve (y, y_pred, pos_label=1)
            fnr = 1 - tpr
            # Find the threshold of the Equal Error Rate based on the difference between the fnr and tpr
            eer_threshold = threshold[np.nanargmin (np.absolute ((fnr - fpr)))]
            # find the index with the smallest absolute value of the difference
            EER = fpr[np.nanargmin (np.absolute ((fnr - fpr)))]

            print (EER)

    # Creating feed-forward neural networks
    class FFNN_network (nn.Module):
        def __init__(self, hidden_dims=[13], input_size=13, out_size=1):
            super (FFNN_network, self).__init__ ()
            self.layers = nn.ModuleList ()
            self.layers.append (nn.Linear (input_size, hidden_dims[0]))
            # Activation functions
            self.act = nn.ReLU ()
            # output activation function
            self.sigmoid = nn.Sigmoid ()

            # hidden layers generation
            # By dynamically creating the network structure in the constructor, it is possible to build feedforward neural networks with different number of layers and different dimensions in a flexible way
            lay_dim = hidden_dims[0]
            for l in hidden_dims:
                self.layers.append (nn.Linear (lay_dim, l))
                lay_dim = l

                # Add linear layer between the last hidden layer and the output layer to layers
            self.layers.append (nn.Linear (lay_dim, out_size))

        def forward(self, x):
            # loop over all but last, ReLu not applied to last
            for layer in self.layers[:-1]:
                x = self.act (layer (x))
            out = self.sigmoid (self.layers[-1] (x))
            return out

    ### FFNN
    # define model and loss function and optimiser
    model = FFNN_network (hidden_dims=[28, 26])
    model.load_state_dict (torch.load ('FFNN_model.pth'))
    model.eval ()
    crit = nn.BCELoss ()
    # Set the weight decay parameter when initializing the optimizer
    weight_decay = 1e-5
    optimiser = optim.Adam (model.parameters (), lr=1e-4, betas=[0.9, 0.99], weight_decay=weight_decay)

    # set hyperparameters
    eps = 1e-8
    epochs = 500

    val_losses = []
    tr_losses = []
    preds = []

    # Obtained a sample validation set for loss calculation
    test_win_len = 32768
    # change is used to determine the termination conditions for model training
    change = 0
    print ('FFNN')
    print ('test')
    for i, (X, y) in enumerate (test_loader):
        with torch.no_grad ():
            y_pred = model (X)

            y_pred = np.array (y_pred.squeeze ())
            y = np.array (y.squeeze ())

            fpr, tpr, threshold = roc_curve (y, y_pred, pos_label=1)
            fnr = 1 - tpr
            eer_threshold = threshold[np.nanargmin (np.absolute ((fnr - fpr)))]
            EER = fpr[np.nanargmin (np.absolute ((fnr - fpr)))]
            print (EER)

    print ('val')
    for i, (X, y) in enumerate (val_loader):
        with torch.no_grad ():
            y_pred = model (X)

            y_pred = np.array (y_pred.squeeze ())
            y = np.array (y.squeeze ())

            fpr, tpr, threshold = roc_curve (y, y_pred, pos_label=1)
            fnr = 1 - tpr
            eer_threshold = threshold[np.nanargmin (np.absolute ((fnr - fpr)))]
            EER = fpr[np.nanargmin (np.absolute ((fnr - fpr)))]
            print (EER)


    print ("Number of data points: {}".format (len (VoiceSet)))
    print ('test accuracy')
    # 循环遍历测试集进行推理和性能评估
    for i, (X, y) in enumerate (test_loader):
        with torch.no_grad ():
            y_pred = model (X)
            # 将输出从形状为(1, seq_length, 1)的张量减少到形状为(seq_length,)的一维数组，使用np.squeeze()函数
            y_pred = np.array (y_pred.squeeze ())
            y = np.array (y.squeeze ())

            # 计算准确率
            accuracy = accuracy_score (y, np.round (y_pred))
            print ("Accuracy:", accuracy)

    print ('val accuracy')
    # 循环遍历测试集进行推理和性能评估
    for i, (X, y) in enumerate (val_loader):
        with torch.no_grad ():
            y_pred = model (X)
            # 将输出从形状为(1, seq_length, 1)的张量减少到形状为(seq_length,)的一维数组，使用np.squeeze()函数
            y_pred = np.array (y_pred.squeeze ())
            y = np.array (y.squeeze ())

            # 计算准确率
            accuracy = accuracy_score (y, np.round (y_pred))
            print ("Accuracy:", accuracy)
    '''
    # Loop through test sets for inference and performance evaluation
    for i, (X, y) in enumerate (test_loader):
        with torch.no_grad ():
            y_pred = model (X)
            y_pred = np.array (y_pred.squeeze ())
            y = np.array (y.squeeze ())

            # 计算DET曲线
            fpr, fnr, _ = roc_curve (y, y_pred, pos_label=1)
            plt.plot (fpr, fnr)
            plt.xlabel ("False Positive Rate")
            plt.ylabel ("False Negative Rate")
            plt.title ("test DET Curve")
            plt.show ()
    '''
    # Loop through val sets for inference and performance evaluation
    for i, (X, y) in enumerate (val_loader):
        with torch.no_grad ():
            y_pred = model (X)
            y_pred = np.array (y_pred.squeeze ())
            y = np.array (y.squeeze ())

            # 计算DET曲线
            fpr, fnr, _ = roc_curve (y, y_pred, pos_label=1)
            plt.plot (fpr, fnr)
            plt.xlabel ("False Positive Rate")
            plt.ylabel ("False Negative Rate")
            plt.title ("val DET Curve")
            plt.show ()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('audio_dir', type=str, help='Directory containing training speech parameter vector sequences')
    parser.add_argument('labels_dir', type=str, help='Directory containing training label sequences')


    args = parser.parse_args()

    evaluate(args.audio_dir, args.labels_dir)