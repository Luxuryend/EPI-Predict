import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import numpy as np


# ==========================================
# 1. 定义模型架构
# ==========================================
def build_dna_model():
    # 输入层：Enhancer(200bp), Promoter(200bp), Distance(1个数值)
    input_enh = layers.Input(shape=(200, 4), name='enh_in')
    input_pro = layers.Input(shape=(200, 4), name='pro_in')
    input_dist = layers.Input(shape=(1,), name='dist_in')

    # --- Enhancer 特征提取分支 ---
    e = layers.Conv1D(64, kernel_size=7, activation='relu', padding='same')(input_enh)
    e = layers.BatchNormalization()(e)  # 归一化，加速收敛
    e = layers.GlobalMaxPooling1D()(e)  # 提取最强 Motif 信号
    e = layers.Dropout(0.2)(e)

    # --- Promoter 特征提取分支 ---
    p = layers.Conv1D(64, kernel_size=7, activation='relu', padding='same')(input_pro)
    p = layers.BatchNormalization()(p)
    p = layers.GlobalMaxPooling1D()(p)
    p = layers.Dropout(0.2)(p)

    # --- 特征融合 ---
    # 拼接两个序列特征和距离特征 (64 + 64 + 1 = 129 维)
    merged = layers.Concatenate()([e, p, input_dist])

    # --- 分类决策层 ---
    x = layers.Dense(64, activation='relu')(merged)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=[input_enh, input_pro, input_dist], outputs=output)

    # 编译：使用 Adam 优化器和二分类交叉熵损失
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy']
    )
    return model


# ==========================================
# 2. 加载 .npy 数据
# ==========================================
print("正在从磁盘加载数据...")
# 6万条数据建议直接加载到内存以提高速度
X_enh_train = np.load('npydata/X_enhancer_train.npy')
X_pro_train = np.load('npydata/X_promoter_train.npy')
X_dist_train = np.load('npydata/X_dist_train.npy')
y_train = np.load('npydata/y_train.npy')

# ==========================================
# 3. 训练配置
# ==========================================
model = build_dna_model()
model.summary()

# 定义回调函数：防止过拟合
early_stop = callbacks.EarlyStopping(
    monitor='val_auc',
    patience=10,  # 如果10轮内验证集AUC不提升则停止
    restore_best_weights=True,  # 停止后恢复表现最好那一轮的权重
    mode='max'
)

print("\n开始训练...")
history = model.fit(
    x={
        'enh_in': X_enh_train,
        'pro_in': X_pro_train,
        'dist_in': X_dist_train
    },
    y=y_train,
    epochs=100,  # 最大迭代100次
    batch_size=64,  # 6万条数据，64是非常稳健的Batch Size
    validation_split=0.2,  # 自动切分1.2万条数据做验证
    callbacks=[early_stop],
    verbose=1
)

# ==========================================
# 4. 保存模型
# ==========================================
model.save('dna_interaction_model.h5')
print("\n✅ 模型训练完成并已保存！")
