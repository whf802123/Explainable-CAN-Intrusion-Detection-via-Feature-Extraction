
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools

df = pd.read_csv(
    r"E:\B\1\data\Car-Dataset\Car_Hacking_Challenge_Dataset_rev20Mar2021\enhanced_can_dataset.csv",
    encoding="utf-8"
)

ext_cols = [
    "frequency", "data_mean", "data_std", "data_max", "data_min",
    "entropy", "is_all_zero", "hamming_weight",
    "id_mean_period", "id_std_period", "rolling_dt_mean", "rolling_dt_std",
    "rolling_id_entropy"
] + [f"byte_{i}_mean" for i in range(8)] + [f"byte_{i}_std" for i in range(8)]

missing_cols = [c for c in ext_cols if c not in df.columns]

X = df[ext_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
le = LabelEncoder()
y = le.fit_transform(df["Class"].astype(str))
class_names = le.classes_

scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y
)

class ContrastiveDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return len(self.X)

    def augment(self, x):
        return x + torch.randn_like(x) * 0.01

    def __getitem__(self, idx):
        x = self.X[idx]
        return self.augment(x), self.augment(x)

ds_train = ContrastiveDataset(X_train)
loader = DataLoader(ds_train, batch_size=512, shuffle=True, drop_last=True)

class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=1)

def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    mask = (~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device))
    exp_sim = torch.exp(sim) * mask

    positives = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    positives = torch.cat([positives, positives], dim=0)

    denom = exp_sim.sum(dim=1)
    loss = -torch.log(positives / denom).mean()
    return loss

device = "cuda" if torch.cuda.is_available() else "cpu"

enc = Encoder(X.shape[1]).to(device)
opt = optim.Adam(enc.parameters(), lr=1e-3)

for epoch in range(20):
    enc.train()
    total_loss = 0

    for a, b in loader:
        a, b = a.to(device), b.to(device)

        z1, z2 = enc(a), enc(b)
        loss = nt_xent_loss(z1, z2)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1:02d}  loss={total_loss / len(loader):.4f}")

enc.eval()

with torch.no_grad():
    emb_train = enc(torch.from_numpy(X_train).float().to(device)).cpu().numpy()
    emb_test = enc(torch.from_numpy(X_test).float().to(device)).cpu().numpy()

clf = LogisticRegression(max_iter=500)
clf.fit(emb_train, y_train)

y_pred = clf.predict(emb_test)
y_score = clf.predict_proba(emb_test)

print("Downstream Classification")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
plt.yticks(np.arange(len(class_names)), class_names)

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

n_classes = len(class_names)

if n_classes == 2:
    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.show()

else:
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

    plt.figure(figsize=(8, 6))

    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc:.2f}")

    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title("ROC Curves")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.show()
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split

csv_path = r"E:\B\1\data\Car-Dataset\CAN-MIRGU-main\\CAN-MIRGU.csv"
df = pd.read_csv(csv_path)

def parse_id(x):
    if isinstance(x, str):
        try: return int(x, 16)
        except ValueError:
            try: return float(x)
            except: return 0
    return x
df['Arbitration_ID'] = df['Arbitration_ID'].apply(parse_id)

for i in range(8):
    df[f'Data{i}'] = (
        df['DATA'].fillna('').astype(str).str.split().str[i]
        .apply(lambda x: int(x, 16) if isinstance(x, str) and x else 0)
    )

feature_cols = ['Arbitration_ID']
if 'DLC' in df.columns: feature_cols.append('DLC')
feature_cols += [f'Data{i}' for i in range(8)]
X = df[feature_cols].astype(np.float32).values
le = LabelEncoder()
y = le.fit_transform(df['Class'].astype(str))
class_names = le.classes_

scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

class ContrastiveDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data)
    def __len__(self):
        return len(self.data)
    def augment(self, x):
        noise = torch.randn_like(x) * 0.02
        return x + noise
    def __getitem__(self, idx):
        x = self.data[idx]
        return self.augment(x), self.augment(x)

train_ds = ContrastiveDataset(X_train)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)

class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64)
        )
    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=1)

def nt_xent(z1, z2, temp=0.5):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temp
    mask = (~torch.eye(2*B, dtype=torch.bool)).to(z.device)
    exp_sim = torch.exp(sim) * mask
    positives = torch.exp((z1 * z2).sum(dim=1) / temp)
    positives = torch.cat([positives, positives], dim=0)
    denom = exp_sim.sum(dim=1)
    return -torch.log(positives / denom).mean()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = Encoder(X_train.shape[1]).to(device)
optimizer = optim.Adam(encoder.parameters(), lr=1e-3)

for epoch in range(20):
    total_loss = 0
    for x1, x2 in train_loader:
        x1, x2 = x1.to(device), x2.to(device)
        z1, z2 = encoder(x1), encoder(x2)
        loss = nt_xent(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/20, Loss: {total_loss/len(train_loader):.4f}")

encoder.eval()
with torch.no_grad():
    emb_train = encoder(torch.from_numpy(X_train).to(device)).cpu().numpy()
    emb_test = encoder(torch.from_numpy(X_test).to(device)).cpu().numpy()

clf = LogisticRegression(max_iter=500).fit(emb_train, y_train)
y_pred = clf.predict(emb_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nContrastive + Logistic Reg Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
plt.yticks(np.arange(len(class_names)), class_names)
thresh = cm.max()/2
for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j,i,cm[i,j], ha='center', va='center',
             color='white' if cm[i,j]>thresh else 'black')
plt.tight_layout(); plt.show()

y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
y_score = clf.predict_proba(emb_test)

if len(class_names) > 2:
    plt.figure(figsize=(8,6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")

    plt.plot([0,1], [0,1], '--', color='gray')
    plt.title('ROC Curves')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.tight_layout()
    plt.show()

else:
    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0,1], [0,1], '--', color='gray')
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.tight_layout()
    plt.show()
'''
