import pandas as pd
import numpy as np
import os, sys
from utils import *
import string

SEQ_LENGTH = 64
MAX_EPOCHS = 50
BATCH_SIZE = 200
DATA_DIR = 'training_data'
EMB_SIZE = 200

if not os.path.exists('sequences.txt'):

    comments = load_text(DATA_DIR+'/comments.txt')
    #one crazy long string
    comments = comments.replace('\n', ' endofcomment ')
    #Every Newline is now just endofcomment
    tokens = comments.split()
    #Tokes is a list of each word in order lol...

    #print("these are the tokens")
    #print(tokens)
    print('Total Tokens: ', len(tokens))
    print('Unique Tokens: ', len(set(tokens)))

    length = SEQ_LENGTH + 1
    sequences = []
    for i in range(length, len(tokens)):
        seq = tokens[i-length:i]
        line = ' '.join(seq)
        sequences.append(line)

    print("These are the sequences:")
    #print(sequences)
    #has  ever single variation of 25 words or whatever...
    save_text(sequences, 'sequences.txt')

    print('Total Sequences: ', len(sequences))

#========TOKENIZE SEQUENCES========

doc = load_text('sequences.txt')
lines = doc.split('\n')




from keras.preprocessing.text import Tokenizer
from pickle import dump, load

if not os.path.exists('tokenizer.pkl'):
    tokenizer = Tokenizer(lower = True)
    tokenizer.fit_on_texts(lines)
    dump(tokenizer, open('tokenizer.pkl', 'wb'))

tokenizer = load(open('tokenizer.pkl', 'rb'))
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
#print(sequences)
vocab_size = len(tokenizer.word_index) + 1

#========CREATE MODEL========

from keras.models import Sequential
from keras.layers import Dropout, Dense, GRU, Embedding, TimeDistributed, BatchNormalization, Input
print(SEQ_LENGTH)
model = Sequential()
#model.add(Input(shape=(SEQ_LENGTH,)))
model.add(Embedding(vocab_size, EMB_SIZE, input_length=SEQ_LENGTH))
model.add(GRU(128))
model.add(Dropout(0.2))
model.add(BatchNormalization(momentum=0.99))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_crossentropy'])

model.summary()

#========ASSEMBLING TRAINING DATA========
from keras.utils import to_categorical
sequences = np.array(sequences)
#sequences.shape = (sequences.size // SEQ_LENGTH, SEQ_LENGTH)

print("This is the sequence's shape")
print(sequences.shape)
X, y = sequences[:,:-1], sequences[:,-1]
#y = to_categorical(y, num_classes=vocab_size)

print(X.shape)
print(y.shape)

if os.path.exists('model.h5'):
    from keras.models import load_model
    model = load_model('model.h5')

model.fit(X, y, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE)
model.save('model.h5')
