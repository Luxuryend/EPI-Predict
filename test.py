import numpy as np
import matplotlib

# 强制使用 Tkinter 窗口显示（这样会弹出一个独立的窗口显示图片）
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report

model = tf.keras.models.load_model('dna_interaction_model.h5')

X_enh_test = np.load('npydata/X_enhancer_test.npy')
X_pro_test = np.load('npydata/X_promoter_test.npy')
X_dist_test = np.load('npydata/X_dist_test.npy')
y_test = np.load('npydata/y_test.npy')

y_probs = model.predict({'enh_in': X_enh_test, 'pro_in': X_pro_test, 'dist_in': X_dist_test})

# 2. 计算 ROC 曲线数据
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# 3. 计算 PR 曲线数据
precision, recall, _ = precision_recall_curve(y_test, y_probs)
ap_score = average_precision_score(y_test, y_probs)

# 4. 绘图
plt.figure(figsize=(12, 5))

# 子图1: ROC 曲线
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

# 子图2: PR 曲线
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap_score:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()
