import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    matthews_corrcoef

model = tf.keras.models.load_model('dna_interaction_model.h5')

X_enh_test = np.load('npydata/X_enhancer_test.npy')
X_pro_test = np.load('npydata/X_promoter_test.npy')
X_dist_test = np.load('npydata/X_dist_test.npy')
y_test = np.load('npydata/y_test.npy')

y_probs = model.predict({'enh_in': X_enh_test, 'pro_in': X_pro_test, 'dist_in': X_dist_test})
y_pred = (y_probs > 0.6).astype(int)

# 保存预测结果
df = pd.read_csv('data/test.csv')
df['probability'] = y_probs
df['prediction'] = y_pred
df.to_csv('data/test_predictions.csv', index=False)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_probs)

print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"F1 分数 (F1-Score):  {f1:.4f}")
print(f"AUC 值 (ROC-AUC):    {auc:.4f}")

mcc = matthews_corrcoef(y_test, y_pred)
print(f"MCC: {mcc:.4f}")

print("\n混淆矩阵 (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))
