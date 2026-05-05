
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical

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

le = LabelEncoder()
y = le.fit_transform(df['Class'].astype(str))
class_names = le.classes_
num_classes = len(class_names)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
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

print("=== ANOVA F-value ranking ===")
print(anova_df.to_string(index=False))

top_k = 40
top_k = min(top_k, len(feature_cols))
top_feats = anova_df['feature'].iloc[:top_k].tolist()
Xtr = pd.DataFrame(X_train, columns=feature_cols)[top_feats].values
Xte = pd.DataFrame(X_test, columns=feature_cols)[top_feats].values

# Feature normalization
scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr)
Xte = scaler.transform(Xte)

# CNN-LSTM reshape: samples × timesteps × channels
Xtr_cnn_lstm = Xtr.reshape(Xtr.shape[0], Xtr.shape[1], 1)
Xte_cnn_lstm = Xte.reshape(Xte.shape[0], Xte.shape[1], 1)

# One-hot labels
y_train_cat = to_categorical(y_train, num_classes=num_classes)

# CNN-LSTM model
model = Sequential([
    Conv1D(
        64,
        kernel_size=3,
        activation='relu',
        input_shape=(Xtr_cnn_lstm.shape[1], 1)
    ),
    MaxPooling1D(pool_size=2),
    LSTM(100),
    Dropout(0.5),
    Dense(100, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()

# Train
history = model.fit(
    Xtr_cnn_lstm,
    y_train_cat,
    epochs=1,
    batch_size=64,
    validation_split=0.1
)

# Predict
y_prob = model.predict(Xte_cnn_lstm)
y_pred = np.argmax(y_prob, axis=1)

# Evaluation
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
print(f"\nCNN-LSTM accuracy using top {top_k} features: {acc:.4f}\n")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}\n")
print("Classification report:")
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

# ROC curves
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
    y_true_bin = label_binarize(y_test, classes=range(len(class_names)))
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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import itertools

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical

# Load data
csv_path = r"E:\B\1\data\Car-Dataset\Car_Hacking_Challenge_Dataset_rev20Mar2021\Fin_host_session_submit_S.csv"
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

# Split Data field into Data0–Data7
for i in range(8):
    df[f'Data{i}'] = (
        df['Data'].fillna('').astype(str).str.split().str[i]
        .apply(lambda x: int(x, 16) if isinstance(x, str) and x else 0)
    )

feature_cols = ['Arbitration_ID'] + [f'Data{i}' for i in range(8)]
df.drop(columns=['DLC'], errors='ignore', inplace=True)
X = df[feature_cols].astype(np.float32).values

le = LabelEncoder()
y = le.fit_transform(df['Class'].astype(str))
class_names = le.classes_
num_classes = len(class_names)

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

# CNN-LSTM reshape: samples × timesteps × channels
X_train_cnn_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# One-hot labels
y_train_cat = to_categorical(y_train, num_classes=num_classes)

# CNN-LSTM model
model = Sequential([
    Conv1D(
        64,
        kernel_size=3,
        activation='relu',
        input_shape=(X_train_cnn_lstm.shape[1], 1)
    ),
    MaxPooling1D(pool_size=2),
    LSTM(100),
    Dropout(0.5),
    Dense(100, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()

# Train
history = model.fit(
    X_train_cnn_lstm,
    y_train_cat,
    epochs=1,
    batch_size=64,
    validation_split=0.1
)

# Predict
y_prob = model.predict(X_test_cnn_lstm)
y_pred = np.argmax(y_prob, axis=1)

# Evaluation
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"CNN-LSTM Test Accuracy: {acc:.4f}")
print(f"Precision:             {precision:.4f}")
print(f"Recall:                {recall:.4f}")
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
plt.title('Confusion Matrix (CNN-LSTM)')
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

# ROC curves
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
    y_true_bin = label_binarize(y_test, classes=range(len(class_names)))
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
