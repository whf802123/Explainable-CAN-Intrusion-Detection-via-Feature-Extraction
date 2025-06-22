
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from lime.lime_tabular import LimeTabularExplainer
from tqdm import tqdm
import matplotlib.pyplot as plt
import shap
from collections import defaultdict
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ========== 1. Load and prepare data ==========
csv_path = r'C:\\Users\\whf80\\Desktop\\Car-Dataset\\CAN-MIRGU-main\\enhancement-CAN-MIRGU.csv'
df = pd.read_csv(csv_path, encoding='utf-8')

# ========== 2. Define extended feature list ==========
ext_cols = [
    'frequency', 'data_mean', 'data_std', 'data_max', 'data_min',
    'entropy', 'hamming_weight',
    'id_mean_period', 'id_std_period', 'rolling_dt_mean', 'rolling_dt_std',
    'rolling_id_entropy',
    'byte_0_mean', 'byte_0_std', 'byte_1_mean', 'byte_1_std',
    'byte_2_mean', 'byte_2_std', 'byte_3_mean', 'byte_3_std',
    'byte_4_mean', 'byte_4_std', 'byte_5_mean', 'byte_5_std',
    'byte_6_mean', 'byte_6_std', 'byte_7_mean', 'byte_7_std'
]

# ========== 3. Build X and y ==========
X = (df[ext_cols]
       .replace([np.inf, -np.inf], np.nan)
       .fillna(0)
       .astype(np.float32)
       .values)

le = LabelEncoder()
y = le.fit_transform(df['Class'].astype(str))
class_names = le.classes_

# ========== 4. Split train/test ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)


# ========== 5. Train a baseline RandomForest ==========
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
# inject feature names so PDP/ICE won't warn
rf.feature_names_in_ = np.array(ext_cols)


# ========== 3. ANOVA F-test 可视化 ==========
F_values, p_values = f_classif(X_train, y_train)

# 按 F 值降序排序
indices = np.argsort(F_values)

plt.figure(figsize=(10, 8))
y_pos = np.arange(len(ext_cols))
plt.barh(y_pos, F_values[indices][::-1], align='center')
plt.yticks(y_pos, [ext_cols[i] for i in indices][::-1], fontsize=10)
plt.gca().invert_yaxis()
plt.xlabel('F-value')
plt.tight_layout()
plt.show()

# ========== 6. Permutation Importance ==========
result = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,
    random_state=42,
    scoring='accuracy'
)

# sort features by importance
importances = result.importances_mean
stds = result.importances_std
indices = np.argsort(importances)[::-1]

# print top 10
print("Top 10 features by permutation importance:")
for i in indices[:10]:
    print(f"{ext_cols[i]:<20}  {importances[i]:.4f} ± {stds[i]:.4f}")

# plot all features
plt.figure(figsize=(10, 8))
plt.barh(
    y=np.arange(len(ext_cols)),
    width=importances[indices],
    xerr=stds[indices],
    align='center'
)
plt.yticks(np.arange(len(ext_cols)), [ext_cols[i] for i in indices])
plt.xlabel("Decrease in accuracy")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


lime_explainer = LimeTabularExplainer(
    training_data        = X_train,
    feature_names        = ext_cols,
    class_names          = class_names,
    mode                 = 'classification',
    discretize_continuous= True
)

# ========== 4. 随机抽取 10% 样本并聚合 LIME 权重 ==========
n_total = len(X_test)
n_sample = max(1, int(0.05 * n_total))           # 至少抽 1 个样本
rng = np.random.RandomState(42)                 # 固定种子，可选
sample_indices = rng.choice(n_total, size=n_sample, replace=False)

agg_weights = defaultdict(list)

# 把 sample_indices 包在 tqdm 中，tqdm 会自动渲染进度条
for idx in tqdm(sample_indices, desc='LIME Progress', unit='样本'):
    exp = lime_explainer.explain_instance(
        data_row    = X_test[idx],
        predict_fn  = rf.predict_proba,
        num_features= len(ext_cols)
    )
    class_map = exp.as_map()[1]
    for feat_idx, weight in class_map:
        feat_name = ext_cols[feat_idx]
        agg_weights[feat_name].append(abs(weight))

# 计算平均绝对权重
mean_abs_weights = {feat: np.mean(ws) for feat, ws in agg_weights.items()}

# 排序并可视化
sorted_feats = sorted(mean_abs_weights, key=lambda f: mean_abs_weights[f], reverse=True)
sorted_vals  = [mean_abs_weights[f] for f in sorted_feats]

plt.figure(figsize=(10, 8))
y_pos = np.arange(len(sorted_feats))
plt.barh(y_pos, sorted_vals, align='center')
plt.yticks(y_pos, sorted_feats, fontsize=10)
plt.gca().invert_yaxis()
plt.xlabel('LIME weight')
plt.tight_layout()
plt.show()



'''
n_total = X_test.shape[0]
n_sample = max(1, int(0.00001 * n_total))
rng = np.random.RandomState(42)
idxs = rng.choice(n_total, size=n_sample, replace=False)
X_sub = X_test[idxs]

# 2. 构造 TreeExplainer，使用 interventional 扰动以支持 probability 空间
explainer = shap.TreeExplainer(
    rf,
    data=X_train,
    feature_perturbation="interventional",
    model_output="probability"
)

# 3. 分批计算 SHAP 值并显示进度
batches = np.array_split(X_sub, 5)
shap_list = []

for batch in tqdm(batches, desc="SHAP 计算进度", unit="批"):
    # 返回 [shap_class0, shap_class1], 每个 shape=(batch_size, n_features)
    sv = explainer.shap_values(batch)
    shap_list.append(sv[1])  # 取正类 class=1 的贡献

# 4. 合并 & 计算平均绝对 SHAP 值
shap_all  = np.vstack(shap_list)                # shape=(n_sample, n_features)
mean_shap = np.mean(np.abs(shap_all), axis=0)   # shape=(n_features,)

# 5. 排序并可视化
order        = np.argsort(mean_shap)[::-1]
sorted_feats = [ext_cols[i] for i in order]
sorted_vals  = mean_shap[order]

y_pos = np.arange(len(sorted_feats))
plt.figure(figsize=(10, 8))
plt.barh(y_pos, sorted_vals[::-1], align='center')
plt.yticks(y_pos, sorted_feats[::-1], fontsize=10)
plt.gca().invert_yaxis()
plt.xlabel('SHAP Weight')
plt.tight_layout()
plt.show()
'''
