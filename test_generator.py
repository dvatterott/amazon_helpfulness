import numpy as np
from keras.preprocessing.text import Tokenizer
import gzip

path = './reviews_Electronics_5.json.gz'
MAX_NB_WORDS = 40000
MAX_SEQUENCE_LENGTH = 1000
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)

def generator_review_parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        review_dict = eval(l)
        yield review_dict['reviewText']

tokenizer.fit_on_texts(generator_review_parse(path))
sequences = tokenizer.texts_to_sequences_generator(generator_review_parse(path))
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


def generator_modelData(path,batch_size=1,token_model=tokenizer):
    g = gzip.open(path, 'r')
    #reviews, scores = [], []
    for l in g:
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
        temp_review = np.reshape(temp_review,(1,1000))

        temp_score = np.zeros((5))
        temp_score[int(review_dict['overall'])-1] = 1

        if len(temp_score) == 0: continue

        #scores.append(np.reshape(temp_score,(1,5)))
        #reviews.append(temp_review)

        yield (temp_review,np.reshape(temp_score,(1,5)))
    g.close()


print('begin')

gen_obj = generator_modelData(path)

score_count = np.zeros((5))
for i in range(10000):
   data = next(gen_obj)
   score_count += np.squeeze(data[1])

mean_score = score_count/np.sum(score_count)
print(mean_score)
