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

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def preprocess(inputs, targets, target_label_dict, max_sequence_length=128):

    # Convert target string into label
    targets = [[(target_label_dict[token] if token in target_label_dict.keys() else target_label_dict["[UNK]"]) for token in sentence] for sentence in targets]

    # Transform the desired output label into a one-hot
    # vector to encode which class the model must predict.
    # NOTE: It is not needed since we use a sparse categorical crossentropy loss
    #targets = [tf.keras.utils.to_categorical(sentence, len(target_label_dict)+1) for sentence in targets]

    # Padding
    #inputs = pad_sequences(inputs,maxlen=max_sequence_length) # cannot be done here (strings), it will be done by the TextVectorization layer 
    targets = pad_sequences(targets,maxlen=max_sequence_length, padding="post")

    # Convert inputs and targets list to a numpy array
    inputs = np.array(inputs)
    targets = np.array(targets)

    #print(inputs[10])
    #print(targets[10])
    return inputs, targets

def generate_label_dict(samples):
    # Create an empty dictionary to store the unique strings and their labels
    toret = {}
    
    # Add an UNKNOWN token 
    toret["[UNK]"] = 0

    # Counter for labeling
    label_counter = 1

    # Iterate through the list
    for sentence in samples:
        for item in sentence:
            if item not in toret:
                # If the string is not in the dictionary, add it with a label
                toret[item] = label_counter
                label_counter += 1

    return toret