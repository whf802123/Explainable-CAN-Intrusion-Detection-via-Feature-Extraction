'''
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import LabelEncoder
from sklearn.feature_selection  import f_classif
from sklearn.metrics            import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from tensorflow.keras.models     import Sequential
from tensorflow.keras.layers    import Input, LSTM, Dense, Dropout
from tensorflow.keras.utils     import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


csv_path = r'C:\\Users\\whf80\\Desktop\\Car-Dataset\\CAN-MIRGU-main\\enhancement-CAN-MIRGU.csv'
df = pd.read_csv(csv_path, encoding='utf-8')

df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')

for col in ['Interface', 'Flag', 'DATA']:
    df[col] = df[col].fillna('').astype('category').cat.codes

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
    # 'rolling_dt_skew', 'rolling_dt_kurt',
    'rolling_id_entropy',
] + [
    f'byte_{i}_mean' for i in range(8)
] + [
    f'byte_{i}_std' for i in range(8)
]
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

# ANOVA F-test
F_values, p_values = f_classif(X_train, y_train)
anova_df = pd.DataFrame({
    'feature': feature_cols,
    'F_value': F_values,
    'p_value': p_values
}).sort_values('F_value', ascending=False)

top_k = 40
top_feats = anova_df['feature'].iloc[:top_k].tolist()
Xtr = pd.DataFrame(X_train, columns=feature_cols)[top_feats].values
Xte = pd.DataFrame(X_test,  columns=feature_cols)[top_feats].values

n_feats = Xtr.shape[1] 

Xtr_lstm = Xtr.reshape((Xtr.shape[0], 1, n_feats))
Xte_lstm = Xte.reshape((Xte.shape[0], 1, n_feats))

le = LabelEncoder().fit(df['Class'].astype(str))
class_names = le.classes_
n_classes = len(class_names)

ytr_cat = to_categorical(y_train, num_classes=n_classes)
yte_cat = to_categorical(y_test,  num_classes=n_classes)

# LSTM 
model = Sequential([
    Input(shape=(1, n_feats)),
    LSTM(32),            # 默认 return_sequences=False
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

es = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = model.fit(
    Xtr_lstm, ytr_cat,
    validation_split=0.2,
    epochs=1,
    batch_size=32,
    callbacks=[es],
    verbose=2
)

# Evaluation 
y_prob = model.predict(Xte_lstm)
y_pred = np.argmax(y_prob, axis=1)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec  = recall_score(y_test, y_pred,    average='weighted')
f1   = f1_score(y_test, y_pred,         average='weighted')

print(f"\nLSTM (top {len(top_feats)} features) accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap='Blues')
plt.title('LSTM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
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

import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ========== Step 1: Load data ==========
csv_path = r"C:\\Users\\whf80\\Desktop\\Car-Dataset\\CAN-MIRGU-main\\CAN-MIRGU.csv"
df = pd.read_csv(csv_path)

# ========== Step 2: Parse Arbitration_ID ==========
def parse_id(x):
    if isinstance(x, str):
        try:   return int(x, 16)
        except:
            try: return float(x)
            except: return 0
    return x
df['Arbitration_ID'] = df['Arbitration_ID'].apply(parse_id)

# ========== Step 3: Split DATA into Data0–Data7 ==========

def parse_data_bytes(data_str, idx):
    if not isinstance(data_str, str) or data_str == "":
        return 0
    parts = data_str.split()
    if idx < len(parts):
        try:
            return int(parts[idx], 16)
        except ValueError:
            return 0
    else:
        return 0

for i in range(8):
    df[f'Data{i}'] = df['DATA'].apply(lambda x: parse_data_bytes(x, i)).astype(np.int32)

# ========== Step 4: Prepare feature matrix ==========
feature_cols = ['Arbitration_ID']
if 'DLC' in df.columns:
    feature_cols.append('DLC')
feature_cols += [f'Data{i}' for i in range(8)]

X = df[feature_cols].astype(np.float32).values

# ========== Step 5: Encode labels ==========
le = LabelEncoder()
y = le.fit_transform(df['Class'].astype(str))
class_names = le.classes_

# ========== Step 6: Train/test split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ========== Step 7: Scale features ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ========== Step 8: Reshape for LSTM ==========
n_feats = X_train_scaled.shape[1]
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, n_feats))
X_test_lstm  = X_test_scaled.reshape((X_test_scaled.shape[0], 1, n_feats))

# ========== Step 9: One-hot encode labels ==========
n_classes    = len(class_names)
y_train_cat  = to_categorical(y_train, num_classes=n_classes)
y_test_cat   = to_categorical(y_test,  num_classes=n_classes)

# ========== Step 10: Build and train LSTM ==========
model = Sequential([
    Input(shape=(1, n_feats)),
    LSTM(32),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

es = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_lstm, y_train_cat,
    validation_split=0.2,
    epochs=1,
    batch_size=32,
    callbacks=[es],
    verbose=2
)

# ========== Step 11: Evaluate on test set ==========
y_prob = model.predict(X_test_lstm)
y_pred = np.argmax(y_prob, axis=1)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec  = recall_score(y_test, y_pred,    average='weighted')
f1   = f1_score(y_test, y_pred,         average='weighted')

print(f"\nLSTM Test Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}\n")

# ========== Step 12: Classification report ==========
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ========== Step 13: Confusion matrix ==========
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('LSTM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
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

# ========== Step 14: ROC curves ==========
y_true_bin  = label_binarize(y_test,  classes=range(n_classes))
y_score_bin = y_prob
if n_classes > 1:
    plt.figure(figsize=(8,6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score_bin[:, i])
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
