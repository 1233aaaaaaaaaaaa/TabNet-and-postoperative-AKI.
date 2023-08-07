import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import roc_curve, auc
import joblib
import os
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# 读取数据
df = pd.read_excel(r"C:\Users\18105\Desktop\医学\9个特征-不带手术时间不带钾.xlsx")
df.replace('-', np.nan, inplace=True)#将表格中-替换为nun
df = df.apply(pd.to_numeric, errors='coerce')#转换为数值型
df.fillna(df.mean(), inplace=True)#替换为均值
# 分离特征和标签
features = df.iloc[2:, 1:].values
labels = df.iloc[2:, 0].values
# 使用SMOTE或者其变种算法进行过采样
smote = BorderlineSMOTE(sampling_strategy=0.6, k_neighbors=2, m_neighbors=5, kind='borderline-1', random_state=42)
features_resampled, labels_resampled = smote.fit_resample(features, labels)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features_resampled, labels_resampled, test_size=0.2, random_state=42)
# 创建并训练TabNet模型
model = TabNetClassifier()#可以调参数
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], patience=80, max_epochs=1000)
# 获取脚本所在的目录路径
script_directory = os.path.dirname(os.path.abspath(__file__))
# 创建用于保存模型的文件夹
model_directory = os.path.join(script_directory, 'saved_models')#生成文件夹，存储每一次训练的模型参数
os.makedirs(model_directory, exist_ok=True)
# 生成唯一的PKL文件名
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
model_filename = f"trained_model_{timestamp}.pkl"
# 保存训练好的模型参数
model_path = os.path.join(model_directory, model_filename)
joblib.dump(model, model_path)
# 在测试集上进行预测
y_pred = model.predict_proba(X_test)[:, 1]
# 计算ROC曲线的假正率和真正率
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')  # 保存为名为 'roc_curve.png' 的图片
plt.show()
# 打印特征重要性
feature_importances = model.feature_importances_
columns = df.columns[1:]
# 排序特征重要性
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_importances = feature_importances[sorted_indices]
sorted_columns = columns[sorted_indices]
# 绘制特征重要性柱状图
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_importances)), sorted_importances)
plt.xticks(range(len(sorted_importances)), sorted_columns, rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('Feature Importance.png')
plt.show()
# 打印特征重要性
for feature_name, importance in zip(sorted_columns, sorted_importances):
    print(f"{feature_name}: {importance}")