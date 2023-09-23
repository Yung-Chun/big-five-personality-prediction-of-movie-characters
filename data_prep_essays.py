import pandas as pd
import re
from gensim.parsing.preprocessing import remove_stopwords
import openai
import warnings

warnings.filterwarnings('ignore')

# import labeled data
essays = pd.read_csv('essays.csv', encoding='cp1252')

# data processing

# data cleansing
for col in essays.columns[2:7]:
    essays[col] = essays[col].replace('n', '0')
    essays[col] = essays[col].replace('y', '1')


def clear_text(sentences, rmpunc, rmsw):

    clean_sentences = []
    for sentence in sentences:
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
        clean_sentences.append(sentence)

    file_name = 'essays_embedding'
    if rmpunc == True:
        file_name += '_rmpunc'
    if rmsw == True:
        file_name += '_rmsw'

    return file_name, clean_sentences


# embedding
openai.api_key = 'your api key'


def get_embedding(text):
    response = openai.Embedding.create(model="text-embedding-ada-002",
                                       input=text)
    return response['data'][0]['embedding']


file_name, essays['clean_text'] = clear_text(essays['TEXT'],
                                             rmpunc=True,
                                             rmsw=True)
essays['clean_text_embedding'] = [
    get_embedding(ct) for ct in essays.clean_text
]

essays.to_csv(f'{file_name}.csv', encoding='cp1252', index=False)