#!pip install opencv-python mediapipe

import numpy as np
import onnx
import os
import time
import mediapipe as mp
import onnxruntime as ort
import torch
from slr_project_mirror.video import IntelVideoReader
from slr_project_mirror.dataset import CustomImageDataset
from slr_project_mirror.LSTM import myLSTM
from slr_project_mirror.preprocess import Preprocess
from slr_project_mirror.pthToOnnx import export_to_onnx
from slr_project_mirror.test_onnx import TestOnnx
from slr_project_mirror.test import Test
from slr_project_mirror.tuto import Tuto
# Gives easier dataset managment by creating mini batches etc.
from torch.utils.data import DataLoader
from torch import nn  # All neural network modules
from torch import optim  # For optimizers like SGD, Adam, etc.
from tqdm import tqdm  # For nice progress bar!


def launch_LSTM(output_size, train, weights_type, make_data_augmentation):
    # Set device cuda for GPU if it's available otherwise run on the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameters of our neural network which depends on the dataset, and
    # also just experimenting to see what works well (learning rate for example).

    learning_rate = 0.001  # how much to update models parameters at each batch/epoch
    batch_size = 32  # number of data samples propagated through the network before the parameters are updated
    NUM_WORKERS = 4
    num_epochs = 100  # number times to iterate over the dataset
    DECAY = 1e-4
    hidden_size = 128  # number of features in the hidden state h
    num_layers = 2

    train_preprocess = Preprocess(
        actions, DATA_PATH_TRAIN, nb_sequences_train, sequence_length, make_data_augmentation)
    valid_preprocess = Preprocess(
        actions, DATA_PATH_VALID, nb_sequences_valid, sequence_length, False)
    test_preprocess = Preprocess(
        actions, DATA_PATH_TEST, nb_sequences_test, sequence_length, False)

    input_size = train_preprocess.get_data_length()

    train_loader = DataLoader(train_preprocess, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)

    test_loader = DataLoader(test_preprocess, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                             pin_memory=True)

    valid_loader = DataLoader(valid_preprocess, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)
    # Initialize network
    model = myLSTM(input_size,  hidden_size,
                   num_layers, output_size).to(device)

    if(train):  # On verifie si on souhaite reentrainer le modele

        model = train_launch(model, output_size, learning_rate, DECAY,
                             num_epochs, train_loader, test_loader, valid_loader)
    else:
        if(weights_type == "onnx"):
            try:
                ort.InferenceSession("/home/adrlfv/Documents/ESILV/A4/AI/SLR_project_mirror/models/slr_"+str(output_size)+".onnx", providers=[
                    'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
                print("Found valid onnx model")
            except Exception as e:
                print("Onnx model not found")

                try:
                    export_to_onnx(output_size)
                    print("Converted pth to onnx")
                except:
                    print("Unable to convert to onnx")
                    weights_type = "pth"
                # print(e)

        if(weights_type == "pth"):
            try:
                model.load_state_dict(torch.load(
                    "../models/slr_"+str(output_size)+".pth"))
                print("Found valid pth model")
            except Exception as e:
                print("Pth model not found")
                model = train_launch(model, output_size, learning_rate, DECAY,
                                     num_epochs, train_loader, test_loader, valid_loader)
            export_to_onnx(output_size)

    return model  # ,logits


def train_loop(train_loader, model, criterion, optimizer):
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


def test_loop(loader, model, criterion):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device='cuda' if torch.cuda.is_available() else 'cpu')
            y = y.to(device='cuda' if torch.cuda.is_available() else 'cpu')
            # x = x.reshape(x.shape[0], -1)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    model.train()
    return torch.div(num_correct, num_samples, rounding_mode='trunc')


def train_launch(model, output_size, learning_rate, DECAY, num_epochs, train_loader, test_loader, valid_loader):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=DECAY)
    # Train Network

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_loader, model, criterion, optimizer)

        print(
            f"Accuracy on training set: {test_loop(train_loader, model, criterion)*100:.2f}")
        print(
            f"Accuracy on test set: {test_loop(test_loader, model, criterion)*100:.2f}")
    print("Done!")

    print(
        f"Accuracy on valid set: {test_loop(valid_loader, model, criterion)*100:.2f}")

    torch.save(model.state_dict(), './models/slr_'+str(output_size)+'.pth')
    
    return model


# on crée des dossiers dans lequels stocker les positions des points que l'on va enregistrer
# Chemin pour les données
DATA_PATH_TRAIN = os.path.join('MP_Data/Train')
DATA_PATH_VALID = os.path.join('MP_Data/Valid')
DATA_PATH_TEST = os.path.join('MP_Data/Test')
RESOLUTION_Y = int(1920)  # Screen resolution in pixel
RESOLUTION_X = int(1680)
# Thirty videos worth of data
nb_sequences = 30
nb_sequences_train = int(nb_sequences*80/100)
nb_sequences_valid = int(nb_sequences*10/100)
nb_sequences_test = int(nb_sequences*10/100)
# Videos are going to be 30 frames in length
sequence_length = 30

# ===================================== Parameters to modify =====================================================
make_train = False
make_dataset = False


make_data_augmentation = True
weights_type = "onnx"  # "pth"
# =================================================================================================================

if(make_dataset):
    make_train = True
# dataset making : (ajouter des actions dans le actionsToAdd pour créer leur dataset)
actionsToAdd = np.array(["nothing", "empty", "hello", "thanks", "iloveyou"])  #

# Actions that we try to detect
actions = np.array(["nothing", "empty", "hello", "thanks", "iloveyou"])
# , "nothing" 'hello', 'thanks', 'iloveyou', "what's up", "hey", "my", "name", "nice","to meet you"

# instances de preprocess
# on crée des instances de preprocess en leur donnant le chemin d'accès ainsi que le nombre de séquences dans chaque dossier
# en fonction de si leur type de preprocess est train, valid, test.

if (make_dataset):
    CustomImageDataset(actionsToAdd, nb_sequences, sequence_length, DATA_PATH_TRAIN,
                       DATA_PATH_VALID, DATA_PATH_TEST, RESOLUTION_X, RESOLUTION_Y).__getitem__()

# Appel du modele

cap = IntelVideoReader()
#myTestOnnx = TestOnnx()

model = launch_LSTM(len(actions), make_train, weights_type, make_data_augmentation)


if(weights_type == "pth"):
    myTest = Test(model, len(actions))
if(weights_type == "onnx"):
    myTest = TestOnnx(len(actions))

myTuto = Tuto(actions, RESOLUTION_X, RESOLUTION_Y)


for action in actions:
    if (action != "nothing" and action != "empty"):
        myTuto.launch_tuto(action)
        myTest.launch_test(actions, action, cap, RESOLUTION_X, RESOLUTION_Y)
