
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch

from sklearn.model_selection   import train_test_split
from sklearn.preprocessing    import LabelEncoder
from sklearn.feature_selection import f_classif
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)

from pytorch_tabnet.tab_model import TabNetClassifier

csv_path = r'C:\\Users\\whf80\\Desktop\\enhanced-Car-Hacking.csv' 
# E:\B\1\data\Car-Dataset\Car_Hacking_Challenge_Dataset_rev20Mar2021\enhanced_can_dataset.csv
df = pd.read_csv(csv_path, encoding='utf-8')

df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
df['Data'] = df['Data'].fillna('').astype('category').cat.codes

def parse_id(x):
    if isinstance(x, str):
        try:
            return int(x, 16)
        except ValueError:
            try:
                return float(x)
            except:
                return 0
    return x

df['Arbitration_ID'] = df['Arbitration_ID'].apply(parse_id)

raw_cols = []
ext_cols = [
    'frequency', 'data_mean', 'data_std', 'data_max', 'data_min',
    'entropy', 'is_all_zero', 'hamming_weight',
    'id_mean_period', 'id_std_period', 'rolling_dt_mean', 'rolling_dt_std',
    'rolling_id_entropy',
] + [f'byte_{i}_mean' for i in range(8)] + [f'byte_{i}_std' for i in range(8)]
feature_cols = raw_cols + ext_cols

X = (
    df[feature_cols]
      .replace([np.inf, -np.inf], np.nan)
      .fillna(0)
      .astype(np.float32)
      .values
)
y = LabelEncoder().fit_transform(df['Class'].astype(str))
class_names = LabelEncoder().fit(df['Class'].astype(str)).classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ANOVA F-value
F_values, p_values = f_classif(X_train, y_train)
anova_df = pd.DataFrame({
    'feature': feature_cols,
    'F_value': F_values,
    'p_value': p_values
}).sort_values('F_value', ascending=False)

print("=== ANOVA F-value 排序结果 ===")
print(anova_df.to_string(index=False))

top_k = 40
top_feats = anova_df['feature'].iloc[:top_k].tolist()

Xtr = pd.DataFrame(X_train, columns=feature_cols)[top_feats].values
Xte = pd.DataFrame(X_test,  columns=feature_cols)[top_feats].values

# TabNetClassifier model
tabnet_params = {
    'n_d': 8,
    'n_a': 8,
    'n_steps': 3,
    'gamma': 1.5,
    'optimizer_fn': torch.optim.Adam,
    'optimizer_params': dict(lr=2e-2),
    'mask_type': 'sparsemax',
    'verbose': 10,
    'seed': 42
}

clf = TabNetClassifier(**tabnet_params)

clf.fit(
    X_train=Xtr, y_train=y_train,
    eval_set=[(Xtr, y_train), (Xte, y_test)],
    eval_name=['train', 'test'],
    eval_metric=['accuracy'],
    max_epochs=1,
    patience=10,
    batch_size=1024,
    virtual_batch_size=128
)

# Evaluation
y_pred = clf.predict(Xte)
y_prob = clf.predict_proba(Xte)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\n Using top {top_k} features, TabNet accuracy：{acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}\n")

print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# Confusoin Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=45)
plt.yticks(ticks, class_names)
thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

feature_importances = clf.feature_importances_
importance_df = pd.DataFrame({
    'feature': top_feats,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print("TabNet features importance rank: ")
print(importance_df.to_string(index=False))

plt.figure(figsize=(8,6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.gca().invert_yaxis()
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

n_samples_to_explain = min(100, Xte.shape[0])
subset = Xte[:n_samples_to_explain]
explain_output = clf.explain(subset)

masks_arr = None

if isinstance(explain_output, tuple) and len(explain_output) >= 2:
    candidate = explain_output[1]
    if isinstance(candidate, dict):

        sorted_keys = sorted(candidate.keys())
        try:
            masks_list = [candidate[k] for k in sorted_keys]
            masks_arr = np.stack(masks_list, axis=0)  # shape -> (n_steps, batch_size, n_features)
        except Exception as e:
            raise RuntimeError(f"Failed to stack the arrays in masks_dict into a 3D array: {e}")
    else:
        # If candidate is not a dict but is already an ndarray/list, try converting it directly
        masks_arr = np.array(candidate)

elif isinstance(explain_output, dict):
    # Some versions may directly return a dict. In this case, stack values by integer keys as well.
    candidate = explain_output
    sorted_keys = sorted(candidate.keys())
    try:
        masks_list = [candidate[k] for k in sorted_keys]
        masks_arr = np.stack(masks_list, axis=0)
    except Exception as e:
        raise RuntimeError(f"Failed to stack the explain_output dictionary into a 3D array: {e}")

elif isinstance(explain_output, np.ndarray):
    masks_arr = explain_output

if masks_arr is None:
    print("Attention masks (masks_arr) were not found or have an incompatible format. Skipping visualization.")
else:
    # Check whether the dimensions are (n_steps, batch_size, n_features)
    if masks_arr.ndim != 3:
        print(f"Warning: masks_arr has {masks_arr.ndim} dimensions instead of the expected 3. Skipping visualization.")
    else:
        n_steps, batch_size, n_features = masks_arr.shape
        print(f"Shape of masks_arr: n_steps={n_steps}, batch_size={batch_size}, n_features={n_features}")

        avg_masks = np.mean(masks_arr, axis=(0, 1))

        order_desc = np.argsort(avg_masks)[::-1]

        sorted_feats = [top_feats[i] for i in order_desc]
        sorted_weights = avg_masks[order_desc]

        plt.figure(figsize=(8, 6))
        plt.barh(sorted_feats, sorted_weights)
        plt.gca().invert_yaxis()
        plt.xlabel('Average Mask Weight')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

'''
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection   import train_test_split
from sklearn.preprocessing    import LabelEncoder, label_binarize
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)

from pytorch_tabnet.tab_model import TabNetClassifier

# Load data
csv_path = r"C:\\Users\\whf80\\Desktop\\Car-Dataset\\Car_Hacking_Challenge_Dataset_rev20Mar2021\\Fin_host_session_submit_S.csv"
             # E:\B\1\data\Car-Dataset\Car_Hacking_Challenge_Dataset_rev20Mar2021\Fin_host_session_submit_S.csv
df = pd.read_csv(csv_path)

# Analyze Arbitration_ID
def parse_id(x):
    if isinstance(x, str):
        try:
            return int(x, 16)
        except ValueError:
            try:
                return float(x)
            except:
                return 0
    return x

df['Arbitration_ID'] = df['Arbitration_ID'].apply(parse_id)

def parse_data_byte(s, idx):
    if not isinstance(s, str) or s.strip() == "":
        return 0
    parts = s.split()
    if idx < len(parts):
        try:
            return int(parts[idx], 16)
        except ValueError:
            return 0
    return 0

for i in range(8):
    df[f'Data{i}'] = df['Data'].apply(lambda s: parse_data_byte(s, i)).astype(np.int32)

feature_cols = ['Arbitration_ID']
if 'DLC' in df.columns:
    feature_cols.append('DLC')
feature_cols += [f'Data{i}' for i in range(8)]

X = df[feature_cols].astype(np.float32).values
le = LabelEncoder()
y = le.fit_transform(df['Class'].astype(str))
class_names = le.classes_

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# TabNetClassifier
tabnet_params = {
    'n_d': 8,
    'n_a': 8,
    'n_steps': 3,
    'gamma': 1.5,
    'optimizer_fn': torch.optim.Adam,
    'optimizer_params': dict(lr=2e-2),
    'mask_type': 'sparsemax',
    'verbose': 10,
    'seed': 42
}

clf = TabNetClassifier(**tabnet_params)

clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_name=['train', 'test'],
    eval_metric=['accuracy'],
    max_epochs=1,
    patience=10,
    batch_size=1024,
    virtual_batch_size=128
)

# Evaluation
y_pred   = clf.predict(X_test)
y_prob   = clf.predict_proba(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"TabNet Test Accuracy: {acc:.4f}")
print(f"Precision:           {prec:.4f}")
print(f"Recall:              {rec:.4f}")
print(f"F1-Score:            {f1:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=45)
plt.yticks(ticks, class_names)
thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

# ROC curve
y_true_bin = label_binarize(y_test, classes=range(len(class_names)))
if y_true_bin.shape[1] > 1:
    plt.figure(figsize=(8,6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.title('ROC Curves')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
else:
    print("ROC skipped: only one class present.")
'''
