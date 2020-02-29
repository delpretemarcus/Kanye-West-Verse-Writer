from utils import *
import random
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
print("hello")
SEQ_LENGTH = 64
BATCH_SIZE = 200
CONF_THRESH = 0.6

doc = load_text('sequences.txt')
lines = doc.split('\n')
#lines is an array of each line of the document's sequencing
from pickle import load

tokenizer = load(open('tokenizer.pkl', 'rb'))
sequences = tokenizer.texts_to_sequences(lines)

vocab_size = len(tokenizer.word_index) + 1

model = load_model('model.h5')
print("its started")
answers = open("50YeezyPredictedTexts.txt", "w+")
for i in range(100):
    result = []
    in_text = lines[random.randint(0,len(lines))].split()
    #intext is an array of each word of a random line of the text
    in_text[len(in_text)-1] = 'endofcomment'
    in_text = ' '.join(in_text)
    #change the last thing of intext to endofcomment and throw it all into a string
    answers.write('\n--------SAMPLE {}-------'.format(i+1))
    answers.write('\n----------Seed---------\n' + in_text)
    answers.write('\n-------Generated-------\n')
    #So now we have our seed and are ready to generate


    for _ in range(4):
        new_comment = ''
        while True:
            if len(new_comment.split()) >= SEQ_LENGTH:
                in_text += ' endofcomment'
                break
            encoded = tokenizer.texts_to_sequences([in_text])[0]
            #our seed is now tokenized as numbers
            encoded = pad_sequences([encoded], maxlen=SEQ_LENGTH, truncating='pre')
            #essentially turns encoded into a 2d numpy array
            yhat_probs = model.predict(encoded, verbose=0)[0]
            yhat = np.random.choice(len(yhat_probs), 1, p=yhat_probs)
            #yhat = model.predict_classes(encoded, verbose=0)
            out_word = ''
            for word,index in tokenizer.word_index.items():
                if index == yhat:
                    out_word = word
                    break
            #out_word is now the word we chose
            in_text += ' ' + out_word
            if out_word == 'endofcomment':
                break
            else:
                new_comment += ' ' + out_word
        #print('-'+new_comment)
        result.append(new_comment)
        answers.write(new_comment + '\n')
    answers.write('\n\n 4bar number ' + str((i + 1)) + ':\n')
    answers.write('----------Done---------')
answers.close()
