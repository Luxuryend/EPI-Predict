import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def one_hot_encode(s):
    maps = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    matrix = np.zeros((200, 4), dtype=np.int8)
    for i, j in enumerate(s):
        try:
            matrix[i, maps[j]] = 1
        except KeyError:
            continue
    return matrix


df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

# enhancer

X_enhancer_train = np.stack(df_train['enhancer_frag_seq'].map(one_hot_encode).values).astype('int8')
np.save('npydata/X_enhancer_train.npy', X_enhancer_train)
del X_enhancer_train
gc.collect()

X_enhancer_test = np.stack(df_test['enhancer_frag_seq'].map(one_hot_encode).values).astype('int8')
np.save('npydata/X_enhancer_test.npy', X_enhancer_test)
del X_enhancer_test
gc.collect()

# promoter

X_promoter_train = np.stack(df_train['promoter_frag_seq'].map(one_hot_encode).values).astype('int8')
np.save('npydata/X_promoter_train.npy', X_promoter_train)
del X_promoter_train
gc.collect()

X_promoter_test = np.stack(df_test['promoter_frag_seq'].map(one_hot_encode).values).astype('int8')
np.save('npydata/X_promoter_test.npy', X_promoter_test)
del X_promoter_test
gc.collect()

# distance

dist_train_log = np.log1p(df_train['enhancer_distance_to_promoter'].values).reshape(-1, 1)
dist_test_log = np.log1p(df_test['enhancer_distance_to_promoter'].values).reshape(-1, 1)

scaler = StandardScaler()
scaler.fit(dist_train_log)

X_distance_train = scaler.transform(dist_train_log).astype('float32')
X_distance_test = scaler.transform(dist_test_log).astype('float32')

np.save('npydata/X_dist_train.npy', X_distance_train)
np.save('npydata/X_dist_test.npy', X_distance_test)

# label

y_train = df_train['label'].values.reshape(-1, 1).astype('int8')
y_test = df_test['label'].values.reshape(-1, 1).astype('int8')

np.save('npydata/y_train.npy', y_train)
np.save('npydata/y_test.npy', y_test)

print('over')
