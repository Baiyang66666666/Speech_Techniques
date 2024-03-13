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
from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.manual_seed(12)

lab_path = os.getcwd()
data_folder = os.path.join(lab_path, 'audio')
labs_folder = os.path.join(lab_path, 'labels')

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

def train(audio_dir, labels_dir):
    from tqdm import tqdm

    # 例如：
    print("Training with audio directory:", audio_dir)
    print("Training with labels directory:", labels_dir)


    train_stds, train_means = data_means_std (train_prefixes)

    # creates a composition of transformations contain normalization transformation
    n_transform = transforms.Compose ([
        transforms.Normalize (train_means, train_stds)]
    )

    # it determines the size of the windows that will be randomly selected from the audio data
    window_length = 512

    VoiceSet = VoiceDataSet ("train", window_length=window_length, transform=n_transform)
    VoiceSet2 = VoiceDataSet ("val", window_length=window_length, transform=n_transform)
    b_size = 8

    # creates the data loader for the training dataset.
    train_loader = DataLoader(VoiceSet, shuffle=True, batch_size=b_size)
    val_loader = DataLoader (VoiceSet2, shuffle=True, batch_size=b_size)

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
    # BCELoss is a common loss function to binary classfication
    crit = nn.BCELoss ()
    # Set the weight decay parameter when initializing the optimizer
    weight_decay = 1e-5
    optimiser = optim.Adam (model.parameters (), lr=1e-4, betas=[0.9, 0.99], weight_decay=weight_decay)

    eps = 1e-7
    epochs = 20
    losses = [0.1]

    # Adjusting the window size of data in the data loader
    train_loader.dataset.win = 5000
    dat_len = len (train_loader.dataset)
    # dat_len=8096

    # obtain the validation dataset (val_X and val_y) for validating the model
    test_win_len = 8096
    # get next batch of data(tensor of data and labels)
    val_X, val_y = next (iter (DataLoader (VoiceDataSet ("val", transform=n_transform, window_length=8096))))

    # Slicing operation to remove data from the last time step
    val_x = val_X[0, 0:-1, :]
    val_y = val_y[0, 0:-1, :]
    train_loader.dataset.win = window_length

    t = tqdm(range (epochs))
    for e in t:
        epoch_loss = 0
        for i, (X, y) in enumerate (train_loader):
            # forward pass
            y_pred = model (X)
            # Gradient to zero
            optimiser.zero_grad ()
            # calculate loss
            loss = crit (y_pred, y)
            # backward pass and update parameters
            loss.backward ()
            optimiser.step ()

            # Perform dimensional expansion, 2 dim->3 dim, (1, seq_length, input_size)
            val_pred = model (val_x.unsqueeze (0))

            # Record the total loss value for the current round
            epoch_loss += crit (val_pred.squeeze (0), val_y)

        if e % 1 == 0:
            epoch_loss /= dat_len
            losses.append (epoch_loss)
            t.set_description (desc=f"Epoch: {e + 1}, loss: {round (epoch_loss.item (), 5)}")

        if abs (losses[-1] - losses[-2]) < eps:
            print ("Converged!")
            break
    torch.save (model.state_dict (), 'LSTM_model.pth')

    if losses[0] == 0.1: losses.remove (0.1)
    plt.plot ([l.item () for l in losses])
    plt.title('train')
    # plt.yscale('log')



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
    crit = nn.BCELoss ()
    # Set the weight decay parameter when initializing the optimizer
    weight_decay = 1e-5
    optimiser = optim.Adam (model.parameters (), lr=1e-4, betas=[0.9, 0.99], weight_decay=weight_decay)

    # set hyperparameters
    eps = 1e-8
    epochs = 300

    val_losses = []
    tr_losses = []
    preds = []

    # length of window
    train_loader.dataset.win = 5000
    dat_len = len (train_loader.dataset)

    # Obtained a sample validation set for loss calculation
    test_win_len = 32768
    val_X, val_y = next (iter (DataLoader (VoiceDataSet ("val", transform=n_transform,
                                                         window_length=test_win_len))))  # acquire dev sample for loss calculation
    val_x = val_X[0, 0:-1, :]
    val_y = val_y[0, 0:-1, :]

    # Set to twice the window length to ensure that the training data covers the complete input window of the model
    train_loader.dataset.win = 8096 * 2
    # change is used to determine the termination conditions for model training
    change = 0

    t = tqdm (range (epochs))
    # Iterate over the training data for each batch
    for e in t:
        epoch_tr_loss = 0
        epoch_val_loss = 0
        for i, (X, y) in enumerate (train_loader):
            y_pred = model (X)

            optimiser.zero_grad ()
            loss = crit (y_pred, y)
            epoch_tr_loss += loss.item ()
            loss.backward ()
            optimiser.step ()

            val_pred = model (val_x.unsqueeze (0))

            epoch_val_loss += crit (val_pred.squeeze (0), val_y).item ()

        if e % 1 == 0:
        # Calculating the average data set loss
            epoch_val_loss /= dat_len
            epoch_tr_loss /= dat_len
            val_losses.append (epoch_val_loss)
            tr_losses.append (epoch_tr_loss)
            if len (val_losses) > 1:
                change = val_losses[-2] - val_losses[-1]
            t.set_description (desc=f"Epoch: {e}, val loss: {round (epoch_val_loss, 5)}, tr loss: {round (epoch_tr_loss, 5)}, val change: {change}")

        if len (val_losses) > 1 and abs (val_losses[-1] - val_losses[-2]) < eps:
            print ("Converged!")
            print (val_losses)
    torch.save (model.state_dict (), 'FFNN_model.pth')

    # Record the objective function values of the training and validation data during the training process and plot the objective function change curve
    train_objective_values = []
    valid_objective_values = []

    # Calculate the objective function value of the training data in the training loop and record it
    for batch_idx, (data, target) in enumerate (train_loader):
        optimiser.zero_grad ()
        output = model (data)
        loss = crit (output, target)
        loss.backward ()
        optimiser.step ()
        train_objective_values.append (loss.item ())

    # Calculate the objective function value of the validation data in the validation loop and record it
    with torch.no_grad ():
        for data, target in val_loader:
            output = model (data)
            loss = crit (output, target)
            valid_objective_values.append (loss.item ())

    # Print the objective function values and number of data points for training, validation and test data
    print ("Training data:")
    print ("Number of data points: {}".format (len (VoiceSet)))
    print ()
    print ("Validation data:")
    print ("Number of data points: {}".format (len (VoiceSet2)))
    print ()

    # Plotting the objective function change curve for training and validation data

    plt.plot (range (len (train_objective_values)), train_objective_values, label='Training')
    plt.plot (range (len (valid_objective_values)), valid_objective_values, label='Validation')
    plt.xlabel ('Epoch')
    plt.ylabel ('Objective Function Value')
    plt.title ('Change in Objective Function during Training')

    # plt.show ()

    plt.figure(figsize=(8, 8))
    plt.plot(tr_losses, label="Training Set")
    plt.plot(val_losses, label="Development Set")

    plt.legend()

    # plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('audio_dir', type=str, help='Directory containing training speech parameter vector sequences')
    parser.add_argument('labels_dir', type=str, help='Directory containing training label sequences')


    args = parser.parse_args()

    train(args.audio_dir, args.labels_dir)