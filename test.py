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


df = pd.read_csv('data/train.csv')
df['log_d'] = df['enhancer_distance_to_promoter'].map(lambda x: np.log1p(x))
scaler = StandardScaler()
X_distance = scaler.fit_transform(np.array(df['log_d']).reshape(-1, 1))
print(X_distance.shape)





import numpy as np
import gc

# 假设 df 是你加载好的 180 万行原始数据

# 1. 保存 Enhancer (内存压力最大)
print("正在处理并保存 Enhancer...")
X_enh = np.stack(df['enhancer_frag_seq'].map(one_hot_encode).values).astype('int8')
np.save('X_enhancer.npy', X_enh)
del X_enh # 及时释放内存
gc.collect()

# 2. 保存 Promoter
print("正在处理并保存 Promoter...")
X_pro = np.stack(df['promoter_frag_seq'].map(one_hot_encode).values).astype('int8')
np.save('X_promoter.npy', X_pro)
del X_pro
gc.collect()

# 3. 保存 Distance (记得 reshape)
print("正在保存 Distance...")
X_dist = df['enhancer_distance_to_promoter'].values.reshape(-1, 1).astype('float32')
np.save('X_distance.npy', X_dist)

# 4. 保存 Labels
print("正在保存 Labels...")
y = df['label'].values.reshape(-1, 1).astype('int8')
np.save('y_labels.npy', y)

print("所有文件已分开保存完成！")





import numpy as np
import tensorflow as tf

# 使用 'r' (read-only) 模式进行内存映射，这不会占用你的实际 RAM
X_enh_mmap = np.load('X_enhancer.npy', mmap_mode='r')
X_pro_mmap = np.load('X_promoter.npy', mmap_mode='r')
X_dist_mmap = np.load('X_distance.npy', mmap_mode='r')
y_labels_mmap = np.load('y_labels.npy', mmap_mode='r')

# 划分训练集和验证集的索引 (手动划分)
total_samples = X_enh_mmap.shape[0]
train_split = int(0.8 * total_samples)

# 喂给模型
# TensorFlow 的 fit 函数可以直接处理这种内存映射的 NumPy 数组
history = model.fit(
    x={
        'enh_in': X_enh_mmap[:train_split], 
        'pro_in': X_pro_mmap[:train_split], 
        'dist_in': X_dist_mmap[:train_split]
    },
    y=y_labels_mmap[:train_split],
    validation_data=(
        {
            'enh_in': X_enh_mmap[train_split:], 
            'pro_in': X_pro_mmap[train_split:], 
            'dist_in': X_dist_mmap[train_split:]
        }, 
        y_labels_mmap[train_split:]
    ),
    batch_size=64,
    epochs=20
)
