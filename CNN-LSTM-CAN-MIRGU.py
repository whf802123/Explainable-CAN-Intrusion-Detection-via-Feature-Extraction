

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load data
csv_path = r'E:\B\1\data\Car-Dataset\CAN-MIRGU-main\\enhancement-CAN-MIRGU.csv'
df = pd.read_csv(csv_path, encoding='utf-8')

df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
for col in ['Interface', 'Flag', 'DATA']:
    df[col] = df[col].fillna('').astype('category').cat.codes

def parse_id(x):
    if isinstance(x, str):
        try:    return int(x, 16)
        except:
            try: return float(x)
            except: return 0
    return x

df['Arbitration_ID'] = df['Arbitration_ID'].apply(parse_id)

# ANOVA F-test
ext_cols = [
    'delta_time','frequency','data_mean','data_std','data_max','data_min',
    'entropy','is_all_zero','hamming_weight',
    'id_mean_period','id_std_period','rolling_dt_mean','rolling_dt_std',
    'rolling_id_entropy'
] + [f'byte_{i}_mean' for i in range(8)] + [f'byte_{i}_std' for i in range(8)]

X_all = (
    df[ext_cols]
      .replace([np.inf, -np.inf], np.nan)
      .fillna(0)
      .astype(np.float32)
      .values
)
y_all = LabelEncoder().fit_transform(df['Class'].astype(str))
class_names = LabelEncoder().fit(df['Class'].astype(str)).classes_

X_train_all, X_unused, y_train_all, _ = train_test_split(
    X_all, y_all, test_size=0.7, stratify=y_all, random_state=42
)

F_values, p_values = f_classif(X_train_all, y_train_all)
anova_df = pd.DataFrame({
    'feature': ext_cols,
    'F_value': F_values,
    'p_value': p_values
}).sort_values('F_value', ascending=False)

# top_k
top_k = 16
top_feats = anova_df['feature'].iloc[:top_k].tolist()

print("Selected top features:", top_feats)

X = df[top_feats].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).values
y = y_all

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

X_train_seq = X_train.reshape((X_train.shape[0], top_k, 1))
X_test_seq  = X_test.reshape((X_test.shape[0],  top_k, 1))

# CNN-LSTM
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu',
           input_shape=(top_k,1), padding='same'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()

# Train
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train_seq, y_train,
    validation_split=0.1,
    epochs=1,
    batch_size=32,
    callbacks=[es],
    verbose=2
)

# Evaluation
loss, acc = model.evaluate(X_test_seq, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.4f}\n")

# Confusion Matrix
y_pred = np.argmax(model.predict(X_test_seq), axis=1)
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
thresh = cm.max()/2
for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j,i,cm[i,j],ha='center',va='center',
             color='white' if cm[i,j]>thresh else 'black')
plt.tight_layout()
plt.show()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ROC Curve
y_true_bin  = tf.keras.utils.to_categorical(y_test, num_classes=len(class_names))
y_score_bin = model.predict(X_test_seq)
if y_true_bin.shape[1] > 1:
    plt.figure(figsize=(8,6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_score_bin[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],linestyle='--',color='gray')
    plt.title('ROC Curves')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
else:
    print("ROC skipped (only one class).")
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load data
csv_path = r"E:\B\1\data\Car-Dataset\CAN-MIRGU-main\\CAN-MIRGU.csv"
df = pd.read_csv(csv_path)
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

# Split DATA field into Data0–Data7
for i in range(8):
    df[f'Data{i}'] = (
        df['DATA'].fillna('').astype(str).str.split().str[i]
        .apply(lambda x: int(x, 16) if isinstance(x, str) and x else 0)
    )

# Feature columns
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

# Feature normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# CNN-LSTM input reshape: samples × timesteps × channels
top_k = X_train.shape[1]
X_train_seq = X_train.reshape(X_train.shape[0], top_k, 1)
X_test_seq = X_test.reshape(X_test.shape[0], top_k, 1)

# CNN-LSTM
model = Sequential([
    Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu',
        input_shape=(top_k, 1),
        padding='same'
    ),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()

# Train
es = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
history = model.fit(
    X_train_seq,
    y_train,
    validation_split=0.1,
    epochs=1,
    batch_size=32,
    callbacks=[es],
    verbose=2
)

# Predict
y_prob = model.predict(X_test_seq)
y_pred = np.argmax(y_prob, axis=1)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('CNN-LSTM Confusion Matrix')
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

print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)

# ROC Curve
if len(class_names) == 2:
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('CNN-LSTM ROC Curve')
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
    plt.title('CNN-LSTM ROC Curves')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
'''
