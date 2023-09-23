import pandas as pd
import numpy as np
import time
import yaml
from pathlib import Path
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings

warnings.filterwarnings('ignore')

os.makedirs('training_result')
result_path = Path('training_result')

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


def the_best_svc(y_col, kernel, gamma, s, e, n):
    X = [x for x in essays_embedding['clean_text_embedding']]
    y = essays_embedding[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        stratify=y,
                                                        test_size=0.2,
                                                        random_state=42)

    Cs = np.logspace(s, e, n)

    svc = SVC(kernel=kernel, gamma=gamma, random_state=42)
    param_grid = {'C': Cs}
    search = GridSearchCV(svc, param_grid, return_train_score=True, n_jobs=-1)
    search.fit(X_train, y_train)

    best_params.update({y_col: search.best_params_['C']})

    record = pd.DataFrame()
    record['C'] = Cs
    record['mean_train_score'] = search.cv_results_['mean_train_score']
    record['mean_test_score'] = search.cv_results_['mean_test_score']
    record['y'] = y_col

    record = record.reindex(columns=[
        'y',
        'C',
        'mean_train_score',
        'mean_test_score',
    ])

    record.to_csv(
        result_path /
        f'svc_essays_embedding_{y_col}_{kernel}_{gamma}_{len(Cs)}.csv',
        index=False)


best_params = {}
for y_col in y_cols:
    print(f'running y = {y_col}, file_name = {file_name} at {time.ctime()}')
    the_best_svc(y_col, 'linear', 'scale', 0, 0.33, 50)

with open(result_path / 'the_best_params.yaml', 'w') as f:
    yaml.dump(best_params, f)