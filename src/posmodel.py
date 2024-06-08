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

import keras
import tensorflow as tf
import numpy as np
from tqdm.keras import TqdmCallback
from keras.layers import Dense, Embedding, TimeDistributed, Bidirectional, LSTM
from keras.callbacks import EarlyStopping

class PosModel(object):

    def __init__(self, target_label_dict, max_sentence_length=128):
        self.model = None
        self.target_label_dict = target_label_dict
        self.max_sentence_length = max_sentence_length

    def build_model(self, X_train, topology):
       
        # Prepare the text vectorizer layer
        text_vectorizer = tf.keras.layers.TextVectorization(
            output_mode="int",
            standardize=None,
            output_sequence_length = self.max_sentence_length # pad the output to max_sentence_length
        )
        text_vectorizer.adapt(X_train)

        #print("Text vectorizer layer prepared.\n")
        #print("Text_vectorizer vocabulary:",text_vectorizer.get_vocabulary())

        # Create an Input layer
        inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
        # A TextVectorizer layer
        x = text_vectorizer(inputs)
        # An Embedding layer (input_dim+1 to take into account unknown words (because of mask_zero))
        x = Embedding(text_vectorizer.vocabulary_size()+1, topology["lstm_units"], mask_zero=True)(x)

        # An Bidirectional LSTM layer
        x = Bidirectional(LSTM(topology["lstm_units"], return_sequences=True))(x)

        # Dense layers for the computation of results
        for i in range(topology["num_dense"]):
            x = TimeDistributed(Dense(units=topology["dense_units"], activation='relu'))(x)


        # A output layer with the appropiate activation function for a multiclass classifier
        outputs = TimeDistributed(Dense(units=len(self.target_label_dict), activation='softmax'))(x)
        # Create the model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        print(self.model.summary())
        #tf.keras.utils.plot_model(self.model, show_shapes=True)

    
    def train(self, train_sets, val_sets, hyperparameters):
        # Prepare training input in batches
        train_ds = tf.data.Dataset.from_tensor_slices((train_sets[0],train_sets[1]))
        train_ds = train_ds.batch(hyperparameters["batch_size"])

        #val_ds = tf.data.Dataset.from_tensor_slices((val_sets[0],val_sets[1]))
        #val_ds = val_ds.batch(hyperparameters["batch_size"])

        # Compile the model
        self.model.compile(loss=hyperparameters["loss"],
              optimizer=hyperparameters["optimizer"],
              metrics=hyperparameters["metrics"])

        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

        # Train the model and show validation loss and validation accuracy at the end of each epoch
        history = self.model.fit(train_ds,
            epochs=hyperparameters["epochs"], validation_data=(val_sets[0],val_sets[1]),
            callbacks=[TqdmCallback(),es]
        )

        return history
        
    def evaluate(self, test_sets, batch_size):
        ds = tf.data.Dataset.from_tensor_slices((test_sets[0],test_sets[1]))
        ds = ds.batch(batch_size)
        
        return self.model.evaluate(ds)

    def predict(self, test_sets):
        # Compute the output and convert values back to POS tags
        # For each word we select the tag with the highest probability
        toret=[]
        sentence_cont = 0

        for sentence in self.model.predict(test_sets):
            output_sentence=[]
            for tag_probabilities in sentence:
                tags = list(self.target_label_dict.keys())
                #print(tag_probabilities)

                # Select the tag with the highest probability
                idx_tag = tag_probabilities.argmax()

                output_sentence.append(tags[idx_tag])

            # Remove padding
            toret.append(output_sentence[:len(test_sets[sentence_cont][0].split())])
            sentence_cont += 1

        return toret
        
        #return [self.target_label_dict[list(self.target_label_dict.values()).index(tag_probabilities.argmax())] for tag_probabilities in self.model.predict(test_sets)]
        #return self.model.predict(test_sets)