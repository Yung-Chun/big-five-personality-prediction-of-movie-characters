import pandas as pd
from pathlib import Path
import time
import yaml
import json
import numpy as np
from sklearn.svm import SVC
import warnings

warnings.filterwarnings('ignore')

embedding_path = Path('Movie-Script-Database/scripts/embedding')
with open('the_best_params.yaml', 'r') as f:
    best_params = yaml.safe_load(f)

# import data
file_name = 'essays_embedding_rmpunc_rmsw'
# file_name = 'essays_embedding_rmpunc'
# file_name = 'essays_embedding'
# file_name = 'essays_embedding_rmsw'

# model training

essays_embedding = pd.read_csv(f'essays_embedding.csv', encoding='cp1252')
essays_embedding = essays_embedding.rename(
    columns={
        'cOPN': 'Openness',
        'cCON': 'Conscientiousness',
        'cEXT': 'Extraversion',
        'cAGR': 'Agreeableness',
        'cNEU': 'Neuroticism',
    })

y_cols = [
    'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness',
    'Neuroticism'
]
essays_embedding['clean_text_embedding'] = [
    np.array([float(n) for n in x[1:-2].split(', ')])
    for x in essays_embedding['clean_text_embedding']
]


def predict_personality(y_col, kernel, gamma):
    svc = SVC(kernel=kernel,
              gamma=gamma,
              random_state=42,
              C=best_params[y_col],
              probability=True)
    svc.fit(X_train, y_train)
    y_proba = svc.predict_proba(X_test)[:, 1]

    return y_proba


movie_embedding = []
for file in embedding_path.glob(f'*embedding*'):
    with open(file) as ef:
        data = json.load(ef)
        movie_name = list(data.keys())[0]

        for char in data[movie_name].keys():
            clean_line_embedding = data[movie_name][char][
                'clean_line_embedding']
            movie_embedding.append([movie_name, char, clean_line_embedding])

movie_embedding = pd.DataFrame(
    movie_embedding, columns=['movie_name', 'char', 'clean_line_embedding'])
print(movie_embedding.shape)

X_train = [
    np.array([float(n) for n in x[1:-2].split(', ')])
    for x in essays_embedding['clean_text_embedding']
]

X_test = [np.array(x) for x in movie_embedding['clean_line_embedding']]

for y_col in y_cols:
    print(f'predicting {y_col} at {time.ctime()}')
    y_train = np.array(essays_embedding[y_col])
    movie_embedding[y_col] = predict_personality(y_col,
                                                 kernel='linear',
                                                 gamma='scale')

movie_char_big_five_predict_result = movie_embedding.drop(
    columns='clean_line_embedding')

for rows in movie_char_big_five_predict_result.itertuples():
    if rows.movie_name.split('-')[-1] == 'The':
        movie_char_big_five_predict_result.loc[
            rows.Index, 'movie_name'] = movie_char_big_five_predict_result.loc[
                rows.Index, 'movie_name'].replace('-The', '')

    movie_char_big_five_predict_result.loc[
        rows.Index, 'movie_name'] = movie_char_big_five_predict_result.loc[
            rows.Index, 'movie_name'].replace('-', ' ')

    movie_char_big_five_predict_result.loc[
        rows.Index,
        'char'] = movie_char_big_five_predict_result.loc[rows.Index,
                                                         'char'].capitalize()

movie_char_big_five_predict_result = movie_char_big_five_predict_result.sort_values(
    by=['movie_name', 'char'])

movie_char_big_five_predict_result.to_csv(
    'movie_char_big_five_predict_result.csv', index=False)
