from keras.preprocessing.text import Tokenizer
import gzip
import os
import numpy as np

#data from http://jmcauley.ucsd.edu/data/amazon/

#Basic parameters
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 40000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
GLOVE_DIR = '/home/dan-laptop/github/ulysses/glove.6B/'

# Convolution
timesteps = 1
filter_length = 3
nb_filter = 32
pool_length = 2

# LSTM
lstm_output_size = 70

#####################set up tokenizer###################3
#generator for tokenizer
def generator_review_parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        review_dict = eval(l)
        yield review_dict['reviewText']

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)

path = './reviews_Electronics_5.json.gz'
tokenizer.fit_on_texts(generator_review_parse(path))
sequences = tokenizer.texts_to_sequences_generator(generator_review_parse(path))
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

################## generator for training neural network #################
def generator_modelData(path,batch_size=1,token_model=tokenizer):
    g = gzip.open(path, 'r')

    #here's the proportion of the ratings (10k samples of electronics)
    cat_props = [0.05293333,0.0406,0.08066667,0.20826667,0.61753333]

    count = 0
    for l in g:
        if count == 0: reviews, scores, sample_weight = [], [], []

        review_dict = eval(l)

        temp_review = np.zeros((MAX_SEQUENCE_LENGTH,))
        temp_r = token_model.texts_to_sequences(review_dict['reviewText'])
        temp_r = [x[0] for x in temp_r if len(x) > 0]
        if len(temp_r) > MAX_SEQUENCE_LENGTH:
            temp_review = temp_r[:MAX_SEQUENCE_LENGTH]
        elif len(temp_r) == 0:
            continue
        else:
            temp_review[-len(temp_r):] = np.squeeze(temp_r)
        #temp_review = np.reshape(temp_review,(1,1000))
        temp_review = np.reshape(temp_review,(1000,))

        temp_score = np.zeros((5))
        temp_score[int(review_dict['overall'])-1] = 1

        if len(temp_score) == 0: continue

        #temp_score = np.reshape(temp_score,(1,5))
        temp_score = np.reshape(temp_score,(5,))

        scores.append(temp_score)
        reviews.append(temp_review)
        sample_weight.append(1/cat_props[int(review_dict['overall'])-1])

        count += 1

        if count == batch_size:
            yield (np.array(reviews),np.array(reviews))
            count = 0
    g.close()

###############Set up embedding layer ###############################33
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

##################### Create RNN #####################################3
from keras.preprocessing import sequence
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Input, MaxPooling1D
from keras.layers import Embedding, LSTM, RepeatVector,UpSampling1D,Convolution1D

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
print('Build model...')
def run_model(training=1):
    # model = Sequential()
    # model.add(embedding_layer)
    # #model.add(Dropout(0.25))
    # #model.add(Convolution1D(nb_filter=nb_filter,
    # #                        filter_length=filter_length,
    # #                        border_mode='valid',
    # #                        activation='relu',
    # #                        subsample_length=1))
    # #model.add(MaxPooling1D(pool_length=pool_length))
    # model.add(LSTM(lstm_output_size))
    #
    # model.add(RepeatVector(timesteps))
    # model.add(LSTM(lstm_output_size, return_sequences=True))
    # #model.add(UpSampling1D(length=pool_length))
    # #model.add(Convolution1D(nb_filter=nb_filter,
    # #                        filter_length=filter_length,
    # #                        border_mode='valid',
    # #                        activation='relu',
    # #                        subsample_length=1))
    # model.add(embedding_layer)
    #
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adadelta')

    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    encoded = embedding_layer(inputs)
    #encoded = Dropout(0.25)(encoded)
    #encoded = Convolution1D(nb_filter=nb_filter,
    #                        filter_length=filter_length,
    #                        border_mode='valid',
    #                        activation='relu',
    #                        subsample_length=1)(encoded)
    #encoded = MaxPooling1D(pool_length=pool_length)(encoded)
    encoded = LSTM(lstm_output_size,)(encoded)

    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(MAX_SEQUENCE_LENGTH, return_sequences=True)(decoded)
    #decoded = UpSampling1D(pool_length=pool_length)(decoded)
    #decoded = Convolution1D(nb_filter=nb_filter,
    #                        filter_length=filter_length,
    #                        border_mode='valid',
    #                        activation='relu',
    #                        subsample_length=1)(decoded)
    decoded = embedding_layer(decoded)

    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    sequence_autoencoder.compile(optimizer='adadelta', loss='mse')

    ##################### Train RNN #####################################
    if training == 1:
        from keras.callbacks import History
        history = History()

        trials_per_epoch = 2000
        batch_size = 32
        nb_epoch = 2

        sequence_autoencoder.fit_generator(generator_modelData(path,batch_size=32), trials_per_epoch, nb_epoch=nb_epoch,
                                            validation_data=generator_modelData(path),nb_val_samples=1280,
                                            callbacks=[history])
        sequence_autoencoder.save_weights('./amazon_ratings_autoenc.h5')

        history = history.history

        import matplotlib.pyplot as plt
        plt.style.use('ggplot')

        plt.plot(history.epoch,history.history['acc'],label='training accuracy')
        plt.plot(history.epoch,history.history['val_acc'],label='test accuracy')
        plt.xlabel('epoch')
        plt.ylabel('model accuracy')
        plt.ylim((0.50,1.00))

        plt.show()
    return model

run_model()
