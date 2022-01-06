import numpy as np
from tqdm import tqdm  # For nice progress bar!
from torch import nn  # All neural network modules
# Parameterless functions, like (some) activation functions
import torch.nn.functional as F
import torch
torch.cuda.empty_cache()

class myLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, layer_size, output_size, bidirectional=True):
        super(myLSTM, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_size, self.hidden_size, self.layer_size, self.output_size = input_size, hidden_size, layer_size, output_size
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size, hidden_size, layer_size,
                            batch_first=True, bidirectional=bidirectional)

        if bidirectional:  # we'll have 2 more layers
            self.layer = nn.Linear(hidden_size*2, output_size)
        else:
            self.layer = nn.Linear(hidden_size, output_size)

    def forward(self, images):
        #print('images shape:', images.shape)

        # Set initial states
        if self.bidirectional:
            hidden_state = torch.zeros(
                self.layer_size*2, images.size(0), self.hidden_size)
            cell_state = torch.zeros(
                self.layer_size*2, images.size(0), self.hidden_size)
        else:
            hidden_state = torch.zeros(
                self.layer_size, images.size(0), self.hidden_size)
            cell_state = torch.zeros(
                self.layer_size, images.size(0), self.hidden_size)

        # LSTM:
        #print("images shape",images.shape)
        output, (last_hidden_state, last_cell_state) = self.lstm(images)
        
        # Reshape
        output = output[:, -1, :]

        # FNN:
        output = self.layer(output)

        return output

    def train_loop(self, train_loader, model, criterion, optimizer):
        with tqdm(train_loader, desc="Train") as pbar:
            total_loss = 0.0
            model = model.train()
            # for data, targets in enumerate(tqdm(train_loader)):
            for frame, targets in pbar:
                frame, targets = frame.cuda(), targets.cuda()
                #frame, targets = frame.cuda().float(), targets.cuda().float()
                optimizer.zero_grad()
                # Get to correct shape
                scores = model(frame)
                # print(targets.shape)
                # print(scores.shape)
                loss = criterion(scores, targets)
                # backward
                loss.backward()
                # gradient descent or adam step
                optimizer.step()
                total_loss += loss.item() / len(train_loader)
                pbar.set_postfix(loss=total_loss)
    # Check accuracy on training & test to see how good our model

    def test_loop(self, loader, model, criterion):
        num_correct = 0
        num_samples = 0
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                # x = x.reshape(x.shape[0], -1)
                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
        model.train()
        return num_correct//num_samples