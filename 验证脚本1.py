import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import roc_curve, auc
import joblib
import matplotlib.pyplot as plt

# 读取测试数据
df_test = pd.read_excel(r"C:\Users\18105\Desktop\医学\9-验证数据集-不带钾不带手术时间.xlsx")
df_test.replace('-', np.nan, inplace=True)
df_test = df_test.apply(pd.to_numeric, errors='coerce')
df_test.fillna(df_test.mean(), inplace=True)

# 分离特征和标签
features_test = df_test.iloc[1:, 1:].values
labels_test = df_test.iloc[1:, 0].values

# 使用BorderlineSMOTE进行过采样
smote = BorderlineSMOTE(sampling_strategy=0.6, k_neighbors=2, m_neighbors=5, kind='borderline-1', random_state=42)
features_resampled_test, labels_resampled_test = smote.fit_resample(features_test, labels_test)

# 加载保存的模型
model = joblib.load(r'C:\Users\18105\PycharmProjects\untitled3\saved_models\trained_model_20230731171654.pkl')

# 在测试集上进行预测
y_pred_test = model.predict_proba(features_resampled_test)[:, 1]

# 计算ROC曲线的假正率和真正率
fpr_test, tpr_test, _ = roc_curve(labels_resampled_test, y_pred_test)
roc_auc_test = auc(fpr_test, tpr_test)

# 打印测试集的AUC值
print("测试集的AUC值:", roc_auc_test)
# 画出ROC曲线
plt.figure()
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# 保存ROC曲线为PNG格式
plt.savefig("test.png")
plt.show()

