'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import itertools

# ========== Step 1: Load enhanced CSV ==========
csv_path = r'C:\\Users\\Administrator\\Desktop\\enhanced-Car-Hacking.csv'
df = pd.read_csv(csv_path, encoding='utf-8')

# ========== Step 2: Convert raw fields to numeric ==========
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

# ========== Step 3: Build full feature list ==========
raw_cols = []
ext_cols = [
    'frequency', 'data_mean', 'data_std', 'data_max', 'data_min',
    'entropy', 'is_all_zero', 'hamming_weight',
    'id_mean_period', 'id_std_period', 'rolling_dt_mean', 'rolling_dt_std',
    'rolling_id_entropy',
] + [f'byte_{i}_mean' for i in range(8)] + [f'byte_{i}_std' for i in range(8)]
feature_cols = raw_cols + ext_cols

# ========== Step 4: Clean & prepare X, y ==========
X = (
    df[feature_cols]
      .replace([np.inf, -np.inf], np.nan)
      .fillna(0)
      .astype(np.float32)
      .values
)
y = LabelEncoder().fit_transform(df['Class'].astype(str))
class_names = LabelEncoder().fit(df['Class'].astype(str)).classes_

# ========== Step 5: Train / test split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ========== Step 6: ANOVA F-value ==========
F_values, p_values = f_classif(X_train, y_train)
anova_df = pd.DataFrame({
    'feature': feature_cols,
    'F_value': F_values,
    'p_value': p_values
}).sort_values('F_value', ascending=False)

print("=== ANOVA F-value ranking ===")
print(anova_df.to_string(index=False))

# ========== Step 7: Build training set using top_k features ==========
top_k = 40
top_feats = anova_df['feature'].iloc[:top_k].tolist()
Xtr = pd.DataFrame(X_train, columns=feature_cols)[top_feats].values
Xte = pd.DataFrame(X_test,  columns=feature_cols)[top_feats].values

# ========== Step 8: Train XGBoost ==========
xgb = XGBClassifier(
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb.fit(Xtr, y_train)
y_pred = xgb.predict(Xte)

# ========== Step 9: Output evaluation metrics ==========
print(f"\nXGBoost accuracy using top {top_k} features: {accuracy_score(y_test, y_pred):.4f}\n")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}\n")
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ========== Step 10: Visualize confusion matrix ==========
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
'''
