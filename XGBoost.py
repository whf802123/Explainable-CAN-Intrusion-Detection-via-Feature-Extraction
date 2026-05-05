
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import itertools

# Load enhanced CSV
csv_path = r'E:\B\1\data\Car-Dataset\Car_Hacking_Challenge_Dataset_rev20Mar2021\enhanced_can_dataset.csv'
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

# Build full feature list
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

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ANOVA F-value
F_values, p_values = f_classif(X_train, y_train)
anova_df = pd.DataFrame({
    'feature': feature_cols,
    'F_value': F_values,
    'p_value': p_values
}).sort_values('F_value', ascending=False)

print("=== ANOVA F-value ranking ===")
print(anova_df.to_string(index=False))

top_k = 40
top_feats = anova_df['feature'].iloc[:top_k].tolist()
Xtr = pd.DataFrame(X_train, columns=feature_cols)[top_feats].values
Xte = pd.DataFrame(X_test,  columns=feature_cols)[top_feats].values

# Train XGBoost
xgb = XGBClassifier(
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb.fit(Xtr, y_train)
y_pred = xgb.predict(Xte)
y_prob = xgb.predict_proba(Xte)

print(f"\nXGBoost accuracy using top {top_k} features: {accuracy_score(y_test, y_pred):.4f}\n")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}\n")
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('XGBoost Confusion Matrix')
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

# ROC Curve
if len(class_names) == 2:
    # Binary classification
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('XGBoost ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

else:
    # Multi-class classification: One-vs-Rest ROC
    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))

    plt.figure(figsize=(8, 6))

    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{name} AUC = {roc_auc:.2f}"
        )

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
'''

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)

from xgboost import XGBClassifier

# Load data
csv_path = r"E:\B\1\data\Car-Dataset\Car_Hacking_Challenge_Dataset_rev20Mar2021\Fin_host_session_submit_S.csv"
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
    df[f'Data{i}'] = df['Data'].apply(
        lambda s: parse_data_byte(s, i)
    ).astype(np.int32)
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
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)
# XGBoost Classifier
xgb = XGBClassifier(
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)
rec = recall_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)
f1 = f1_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)

print(f"XGBoost Test Accuracy: {acc:.4f}")
print(f"Precision:             {prec:.4f}")
print(f"Recall:                {rec:.4f}")
print(f"F1-Score:              {f1:.4f}\n")
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()

ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=45)
plt.yticks(ticks, class_names)
thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        cm[i, j],
        ha="center",
        va="center",
        color="white" if cm[i, j] > thresh else "black"
    )
plt.tight_layout()
plt.show()

# ROC curve
if len(class_names) == 2:
    # Binary classification ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

else:
    y_true_bin = label_binarize(
        y_test,
        classes=range(len(class_names))
    )
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.2f})"
        )
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('ROC Curves')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
'''
