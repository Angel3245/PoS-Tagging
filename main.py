# Copyright (C) 2024  Jose Ángel Pérez Garrido
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from src.preprocess import *
from src.posmodel import PosModel
from src.fileparser import *

from pathlib import Path
from InquirerPy import inquirer
import os
import pickle
import matplotlib.pyplot as plt

def ask_training_config(datafolder):
    # Ask user to select between supported treebanks 
    language = inquirer.select(
        message=str(f"Select treebank language for training:"),
        choices=os.listdir(str(f"{datafolder}/Datasources")),
    ).execute()

    # Ask user network topology
    # LSTM layer units
    lstm_units = inquirer.number(
        message="Select the number of units for the LSTM layer:",
    ).execute()

    # Number of TimeDistributed Dense layer
    num_dense = inquirer.number(
        message="Select the number of Time Distributed Dense layers:",
    ).execute()

    if int(num_dense) > 0:
        # TimeDistributed Dense layer units
        dense_units = inquirer.number(
            message="Select the number of units for Dense layers:",
        ).execute()
    else:
        dense_units = 0

    # Ask user hyperparameters for training

    # Optimizer
    optimizer = inquirer.select(
        message="Select an optimizer:",
        choices=["adam", "sgd", "rmsprop", 
                "adadelta", "adagrad", "adamax",
                "adafactor", "nadam", "ftrl"],
    ).execute()

    # Number of epochs
    epochs = inquirer.number(
        message="Select a number of epochs value:",
    ).execute()

    # Batch size
    batch_size = inquirer.number(
        message="Select a batch size value:",
    ).execute()

    # Set training hyperparameters
    hyperparameters = {
        "loss" : "sparse_categorical_crossentropy",
        "optimizer" : optimizer,
        "metrics" : ["accuracy"],
        "epochs" : int(epochs),
        "batch_size" : int(batch_size)
    }

    # Set topology
    topology ={
        "lstm_units" : int(lstm_units),
        "num_dense" : int(num_dense),
        "dense_units" : int(dense_units)
    }

    return language, hyperparameters, topology

def ask_evaluation_config(datafolder):
    # Get trained models
    if not os.path.isdir(str(f"{datafolder}/Model_output")):
        raise Exception("No trained models found in ./Model_output.")
    
    model_names = os.listdir(str(f"{datafolder}/Model_output"))

    if len(model_names) == 0:
        raise Exception("No trained models found in ./Model_output.")
                  
    # Ask user to select between supported treebanks 
    language = inquirer.select(
        message=str(f"Select treebank language for evaluation:"),
        choices=os.listdir(str(f"{datafolder}/Datasources")),
    ).execute()

    # Ask user to select a previously trained model
    pos_model = inquirer.select(
        message="Select a model:",
        choices=model_names,
    ).execute()

    # Ask user the hyperparameters for evaluation
    # Batch size
    batch_size = inquirer.number(
        message="Select a batch size value:",
    ).execute()

    return language, int(batch_size), pos_model

def ask_prediction_config(datafolder):
    # Get trained models
    if not os.path.isdir(str(f"{datafolder}/Model_output")):
        raise Exception("No trained models found in ./Model_output.")
    
    model_names = os.listdir(str(f"{datafolder}/Model_output"))

    if len(model_names) == 0:
        raise Exception("No trained models found in ./Model_output.")
    
    # Ask user to select a previously trained model
    pos_model = inquirer.select(
        message="Select a model for predictions:",
        choices=model_names,
    ).execute()

    # Ask user for a sentence to compute its POS tagging
    input_sentence = inquirer.text(
        message="Write a sentence to compute its POS tagging:",
    ).execute()

    return pos_model, input_sentence

def train(datafolder):
    """
        Ask user for training configuration, train a POS model and save it as a pickle file.
    """
    language, hyperparameters, topology = ask_training_config(datafolder)

    print("Loading",language,"dataset", end="")

    # PREPROCESS INPUT SAMPLES
    parser = Conllu_parser()

    # Parse train file
    input_str = str(f"{datafolder}/Datasources/{language}/train.conllu")
    x_train, y_train = parser(input_str)

    # Parse validation file
    input_str = str(f"{datafolder}/Datasources/{language}/dev.conllu")
    x_val, y_val=parser(input_str)

    # Parse test file
    input_str = str(f"{datafolder}/Datasources/{language}/test.conllu")
    x_test, y_test=parser(input_str)

    print(" - DONE")

    # Generate a label mapping dictionary
    print("Generating label dict", end="")
    target_label_dict = generate_label_dict(y_train)
    print(" - DONE")
    print(target_label_dict)

    # Preprocess datasets
    print("Preprocessing data", end="")
    x_train, y_train = preprocess(x_train, y_train, target_label_dict)
    x_val, y_val = preprocess(x_val, y_val, target_label_dict)
    x_test, y_test = preprocess(x_test, y_test, target_label_dict)
    print(" - DONE")

    # Create POS model architecture
    print("Creating model", end="")
    model = PosModel(target_label_dict) 
    model.build_model(x_train, topology)
    print(" - DONE")

    # Train
    print("TRAINING", end="")
    history = model.train((x_train,y_train),(x_val,y_val),hyperparameters)
    print(" - DONE")

    # Evaluate
    print("EVALUATING", end="")
    print("[Loss, Accuracy] =",model.evaluate((x_test, y_test), hyperparameters["batch_size"]))
    print(" - DONE")

    # Save model as a pickle file
    print("Saving model as a pickle file", end="")
    if not os.path.exists(str(f"{datafolder}/Model_output")):
        os.makedirs(str(f"{datafolder}/Model_output"))

    with open(str(f"{datafolder}/Model_output/{language}.pickle"), "wb") as data_file:
        pickle.dump(model,data_file)
    print(" - DONE")

    # Generate training plots
    print("Generating training plots")
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()

    # Save plot
    if not os.path.exists(str(f"{datafolder}/Plots/{language}")):
        os.makedirs(str(f"{datafolder}/Plots/{language}"))
    plt.savefig(str(f"{datafolder}/Plots/{language}/Plot_accuracy.png"))
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()

    # Save plot
    plt.savefig(str(f"{datafolder}/Plots/{language}/Plot_loss.png"))
    plt.close()
    


def evaluate(datafolder):
    """
        Ask user for evaluation treebank and evaluate a POS model previously saved as a pickle file.
    """
    language_samples, batch_size, pos_model = ask_evaluation_config(datafolder)
    
    # Load model
    print("Loading pickle model", end="")
    with open(str(f"{datafolder}/Model_output/{pos_model}"), "rb") as data_file:
        model = pickle.load(data_file)
    #model = keras.models.load_model(str(f"{datafolder}/Model_output/{pos_model}"))
    print(" - DONE")

    # Preprocess test samples
    print("Preprocessing test samples", end="")
    parser = Conllu_parser()
    input_str = str(f"{datafolder}/Datasources/{language_samples}/test.conllu")
    x_test, y_test=parser(input_str)

    x_test, y_test = preprocess(x_test, y_test, model.target_label_dict)
    print(" - DONE")

    # Evaluate
    print("EVALUATING", end="")
    print("[Loss, Accuracy] =",model.evaluate((x_test, y_test), batch_size))
    print(" - DONE")

def predict(datafolder):
    """
        Compute POS tagging for a user given sentence
    """
    pos_model, input_sentence = ask_prediction_config(datafolder)

    # Load model
    print("Loading pickle model")
    with open(str(f"{datafolder}/Model_output/{pos_model}"), "rb") as data_file:
        model = pickle.load(data_file)

    # Test prediction
    print("Test prediction: ",input_sentence)
    print(model.predict([[input_sentence]]))

def main():
    datafolder=Path.cwd()

    print("POS tagger")

    # Ask user to select between the different functionalities supported by the program 
    functionality = inquirer.select(
        message="Select functionality:",
        choices=["Train a model", "Evaluate a model previously trained", "Predict with a model previously trained"],
    ).execute()

    function_map = {
        "Train a model" : train,
        "Evaluate a model previously trained" : evaluate,
        "Predict with a model previously trained" : predict
    }
    
    function_map[functionality](datafolder)


if __name__ == "__main__":
    main()