#http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
#fchollet also has a similar model at https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
#try to note differences in these as i go

#this script trains the model on letter contingencies

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
print('corpus length:', len(raw_text))

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
print('total chars:', len(chars))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
#fchollet uses shorter sequences and a step size of 3
seq_length = 100
step = 1
dataX = [] #preceding vector of letters
dataY = [] #upcoming letter
for i in range(0, n_chars - seq_length, step):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

print('Build model...')
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
#fchollet uses a different optimizer...
#from keras.optimizers import RMSprop
#optimizer=RMSprop(lr=0.01)

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, nb_epoch=20, batch_size=128, callbacks=callbacks_list)
