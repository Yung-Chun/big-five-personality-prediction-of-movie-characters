import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import remove_stopwords
from pathlib import Path
import os
from os.path import getsize
import time
import json
import csv
import re
import tiktoken
import openai

msdb_path = Path('Movie-Script-Database/scripts')
charinfo_path = Path(msdb_path / 'parsed/charinfo')
dialogue_path = Path(msdb_path / 'parsed/dialogue')

os.makedirs('clean')
os.makedirs('embedding')
clean_path = Path(msdb_path / 'clean')
embedding_path = Path(msdb_path / 'embedding')


# data cleansing
def clear_text(sentence, rmpunc, rmsw):

    sentence = sentence.lower()
    sentence = re.sub("\d+", "", sentence)

    # punc
    if rmpunc == True:
        sentence = re.sub(r'[^\w\s]', ' ', sentence)
        sentence = re.sub(r'\b[a-zA-Z]\b', '', sentence)

    # stopwords
    if rmsw == True:
        sentence = remove_stopwords(sentence)

    sentence = ' '.join([w for w in sentence.split(' ') if w != ''])

    return sentence


def pick_main_char(pair):
    threshold = np.quantile(list(line_num.values()), .95)
    key, value = pair
    if value >= threshold:
        return True  # filter pair out of the dictionary
    else:
        return False  # keep pair in the filtered dictionary


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def save_clean_json_file(dialogue):
    print(f'saving {movie_name}_clean at {time.ctime()}', end='\n\n')
    jsonFile = open(clean_path / f'{movie_name}_clean.json', 'w')
    w = json.dumps(dialogue)
    jsonFile.write(w)
    jsonFile.close()


total_tokens = 0
for file in charinfo_path.glob(f'*charinfo*'):
    print(f'opening {file.stem} at {time.ctime()}')

    with open(file) as cf:
        movie_name = file.stem.split('_')[0]
        line_num = {}
        data = cf.readlines()
        for row in data:
            line = row.replace('\n', '').split(': ')
            line_num.update({line[0]: int(line[-1])})

        main_char_line_num = dict(filter(pick_main_char, line_num.items()))
        main_char_line_num = dict(
            sorted(main_char_line_num.items(),
                   key=lambda x: x[1],
                   reverse=True))

    for file in dialogue_path.glob(f'*{movie_name}_dialogue*'):
        size = os.path.getsize(file)
        if size > 1024 * 5:
            print(f'opening {file.stem} at {time.ctime()}')

            with open(file) as df:
                dialogue = {movie_name: {}}
                data = df.readlines()

                # gather raw lines
                for row in data:
                    line = row.replace('\n', '').split('=>')
                    char = line[0]
                    raw_line = line[-1]

                    if char in main_char_line_num.keys():
                        if char not in dialogue[movie_name].keys():
                            dialogue[movie_name].update({
                                char: {
                                    'raw_line': raw_line,
                                    'clean_line': {}
                                }
                            })

                        else:
                            dialogue[movie_name][char]['raw_line'] += raw_line

                # clear lines
                for char in dialogue[movie_name].keys():
                    dialogue[movie_name][char]['clean_line'] = clear_text(
                        dialogue[movie_name][char]['raw_line'], True, True)
                    token = num_tokens_from_string(
                        dialogue[movie_name][char]['clean_line'])
                    total_tokens += token

                save_clean_json_file(dialogue)

print('total_tokens:', total_tokens)

# embedding
openai.api_key = 'your api key'


def get_embedding(text):
    response = openai.Embedding.create(model="text-embedding-ada-002",
                                       input=text)
    return response['data'][0]['embedding']


for file in clean_path.glob(f'*clean*'):
    size = os.path.getsize(file)
    movie_name = file.stem.split('_')[0]
    embedding_dict = {}

    if size > 1024 * 5:
        print(f'opening {file.stem} at {time.ctime()}')

        with open(file) as f:
            data = json.load(f)
            for char in data[movie_name].keys():
                clean_lines = data[movie_name][char]['clean_line']
                response = openai.Embedding.create(
                    model='text-embedding-ada-002', input=clean_lines)
                data[movie_name][char]['clean_line_embedding'] = response[
                    'data'][0]['embedding']
                data[movie_name][char].pop('raw_line')
                data[movie_name][char].pop('clean_line')

            print(f'saving {movie_name}_embedding at {time.ctime()}',
                  end='\n\n')
            jsonFile = open(embedding_path / f'{movie_name}_embedding.json',
                            'w')
            w = json.dumps(data)
            jsonFile.write(w)
            jsonFile.close()