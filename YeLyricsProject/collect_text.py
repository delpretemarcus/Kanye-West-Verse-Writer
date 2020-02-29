import pandas as pd
import os, sys, string
from utils import save_text
import ast

DATA_DIR = 'training_data'


#for file in os.listdir(DATA_DIR):
 #   print(file)
#datas.append(pd.read_csv(DATA_DIR+'/thing.txt'))
#print('thing.txt')
datas = open(DATA_DIR+'/thing.txt')
clean_comments = datas.readlines()
comments = '\n'.join(clean_comments)
datas.close()

#print('Total Comments: ',len(comments))

#clean_comments = []

#for comment in comments:
   # check_comment = str(comment).lower().split()
  #  if 'robot' in check_comment or 'build' in check_comment or 'make' in check_comment:
   #     clean_comments.append(str(str(comment).encode('utf-8'))[2:-1])

print('Total Robot Comments: ',len(clean_comments))

clean_comments = set(clean_comments)

print('Unique Robot Comments: ',len(clean_comments))


def flatten(lst):
    for el in lst:
        if isinstance(el, list):
            yield from el
        else:
            yield el

cleaned = []
for comment in clean_comments:
    token = comment.split()
    token = [word.split('-') for word in token]
    tokens = flatten(token)
    table = str.maketrans('','',string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    cleaned.append(' '.join(tokens))

save_text(cleaned, DATA_DIR+'/comments.txt')


#THIS IS DONE FOR NOW
