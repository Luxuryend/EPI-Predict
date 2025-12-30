import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt

df = pd.read_csv('data/test_predictions.csv')


def get_status(row):
    if row['label'] == 1 and row['prediction'] == 1: return 'TP (Success)'
    if row['label'] == 0 and row['prediction'] == 0: return 'TN (Success)'
    if row['label'] == 0 and row['prediction'] == 1: return 'FP (FPR Error)'
    if row['label'] == 1 and row['prediction'] == 0: return 'FN (FNR Error)'


df['status'] = df.apply(get_status, axis=1)


def fast_encode(seqs, max_len=200):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((len(seqs), max_len * 4))
    for i, seq in enumerate(seqs):
        s = str(seq)[:max_len]
        for j, base in enumerate(s):
            if base in mapping:
                encoded[i, j * 4 + mapping[base]] = 1
    return encoded


print("正在处理特征...")
feat_enh = fast_encode(df['enhancer_frag_seq'].values)
feat_pro = fast_encode(df['promoter_frag_seq'].values)
X = np.hstack([feat_enh, feat_pro])

print("UMAP 降维中...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X)
df['u1'], df['u2'] = embedding[:, 0], embedding[:, 1]

import matplotlib

matplotlib.use('Agg')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

colors = {
    'TP (Success)': '#27ae60',  # 绿
    'TN (Success)': '#2980b9',  # 蓝
    'FP (FPR Error)': '#e74c3c',  # 红
    'FN (FNR Error)': '#f1c40f'  # 黄
}

for status in ['TN (Success)', 'TP (Success)', 'FN (FNR Error)', 'FP (FPR Error)']:
    data = df[df['status'] == status]
    ax1.scatter(data['u1'], data['u2'], c=colors[status], label=status,
                s=20, alpha=0.6, edgecolors='none')
ax1.set_title("Overlay: Errors on Top", fontsize=16)
ax1.legend(loc='upper right')

ax2.scatter(df['u1'], df['u2'], c='#dcdde1', s=10, alpha=0.2, label='Correct Samples')  # 背景
for status in ['FN (FNR Error)', 'FP (FPR Error)']:
    data = df[df['status'] == status]
    ax2.scatter(data['u1'], data['u2'], c=colors[status], label=status,
                s=40, alpha=0.9, edgecolors='black', linewidth=0.5)
ax2.set_title("Error Concentration Analysis", fontsize=16)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('image/umap.png', dpi=300, bbox_inches='tight')
print("over")
