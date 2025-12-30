# EPI-Predict
数据集：   

https://huggingface.co/datasets/Luxuryend/EpiDataset  

EPI-Predict: Enhancer-Promoter Interaction Prediction

基于卷积神经网络 (CNN) 与独热编码 (One-hot Encoding) 的 DNA 序列相互作用预测工具。

## 🔬 方法论 (Methodology)

### 特征表征 (Representation)
采用 **One-hot Encoding** 将基因组碱基（A, T, C, G）转换为高维稀疏矩阵，完整保留了序列的原始排列信息。

### 模型架构 (Model Architecture)
构建了双通路 **Deep CNN** (卷积神经网络)：
* **并行编码器**: 分别提取增强子 (Enhancer) 与启动子 (Promoter) 的局部空间特征。
* **特征融合层**: 整合序列特征与物理距离（Distance）信息，通过全连接层输出相互作用概率。

### 预测逻辑与阈值 (Prediction & Threshold)
为了在生物序列分析中获得更高的灵敏度（Recall），本项目在推理阶段将分类阈值设定为 **0.6**。当模型输出概率大于或等于 0.6 时，判定为存在相互作用。

---

## 📊 实验结果 (Model Evaluation)

本项目在 DNA 相互作用测试集上进行了评估。通过将阈值优化至 **0.6**，模型在保持高准确率的同时，表现出极高的召回率（接近 99.5%）。

### 核心指标 (Latest Results)
| 指标 (Metric) | 数值 (Value) | 说明 (Note) |
| :--- | :--- | :--- |
| **Accuracy** | **0.9567** | 整体预测准确率 |
| **Precision** | **0.9611** | 预测为正样本中的真实比例 |
| **Recall** | **0.9944** | 真实正样本被成功检出的比例 |
| **F1-Score** | **0.9775** | 精确率与召回率的调和平均 |
| **ROC-AUC** | **0.8217** | 衡量模型区分正负样本的能力 |
| **MCC** | **0.4486** | 衡量不平衡数据分类质量的核心指标 |

### 混淆矩阵 (Confusion Matrix)
> 测试集样本总数：13062。实验结果显示模型对正样本（Interaction）具有极强的识别捕捉能力。

| 真实 \ 预测 | Negative (0) | Positive (1) |
| :--- | :---: | :---: |
| **Negative** | 203 | 497 |
| **Positive** | 69 | 12293 |



### 结果分析 (Result Analysis)
1. **灵敏度优先**：在 EPI（增强子-启动子相互作用）预测中，漏诊的代价通常高于误报。**0.9944** 的召回率确保了绝大多数潜在的调控关系被成功捕获。
2. **鲁棒性表现**：尽管数据集存在类别不平衡（正样本远多于负样本），但 **0.4486** 的 MCC 值证明模型依然具有稳健的分类能力。


## 📈 数据可视化分析 (Data Visualization)

为了从多个统计与分布角度全面评估模型性能，本项目对预测结果进行了系统性的数据可视化分析。所有图像均由 matplotlib 生成，并统一保存在 image/ 目录中。

### ROC曲线

<img width="2074" height="1638" alt="ROC" src="https://github.com/user-attachments/assets/f46d3ec5-a727-44c8-b0a3-585264582ff6" />


### 综合指标可视化 (Overall Metrics Visualization)

<img width="4170" height="2966" alt="summary_visualization" src="https://github.com/user-attachments/assets/657540d5-551f-48db-96c3-3c04401107f0" />


下图整合了 六项核心评估指标，通过柱状图与雷达图从不同视角展示模型整体性能：

柱状图：直观比较各评价指标的数值大小

雷达图：从多维角度刻画模型性能轮廓，便于整体对比

箱线图：展示预测概率的集中趋势与离散程度

小提琴图：结合概率密度估计（KDE）

更直观反映预测概率的分布形态

图中以 绿色虚线 标出了模型采用的分类阈值 0.6。

## UMAP 数据降维

为了了解模型分类性能，利用 **UMAP** 算法将高维特征映射到二维空间：



* **全局叠加图 (Overlay)**: 强制将 **FP（红色）** 与 **FN（黄色）** 错误样本置于顶层绘制。若红/黄点聚集成团，说明模型对特定序列模式存在系统性偏见。

* **分离对比图 (Separation)**: 将正确预测样本淡化为背景，专注于分析错误样本在特征空间中的地理分布，从而识别模型“翻车”的盲区。
