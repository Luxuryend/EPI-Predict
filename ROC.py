import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib

matplotlib.use('TkAgg')  # 解决Pycharm旧版本问题
import matplotlib.pyplot as plt

df = pd.read_csv('data/test_predictions.csv')

y_true = df['label']
y_probs = df['probability']
y_pred = df['prediction']

fpr, tpr, thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, lw=2, label=f'Auc = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], lw=2, linestyle='--')  # 绘制对角基准线
plt.xlim([0.0, 1.1])
plt.ylim([0.0, 1.1])
plt.xlabel('FPR')
plt.ylabel('TPR (Recall)')
plt.title('ROC Curve')
plt.legend(loc="lower right")  # 显示图例
plt.grid(True)

plt.savefig(
    "image/ROC.png",
    dpi=300,
    bbox_inches='tight'
)
plt.show()
