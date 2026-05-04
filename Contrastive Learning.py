'''
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv(r'C:\\Users\\whf80\\Desktop\\Car-Dataset\\CAN-MIRGU-main\\enhancement-CAN-MIRGU.csv', encoding='utf-8')
ext_cols = [
    'frequency','data_mean','data_std','data_max','data_min',
    'entropy','is_all_zero','hamming_weight',
    'id_mean_period','id_std_period','rolling_dt_mean','rolling_dt_std',
    'rolling_id_entropy'
] + [f'byte_{i}_mean' for i in range(8)] + [f'byte_{i}_std' for i in range(8)]
X = df[ext_cols].replace([np.inf,-np.inf], np.nan).fillna(0).values.astype(np.float32)
y = LabelEncoder().fit_transform(df['Class'].astype(str))

scaler = StandardScaler()
X = scaler.fit_transform(X)

class ContrastiveDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X)
    def __len__(self):
        return len(self.X)
    def augment(self, x):
        noise = torch.randn_like(x) * 0.01
        return x + noise
    def __getitem__(self, idx):
        x = self.X[idx]
        return self.augment(x), self.augment(x)

n = len(X)
perm = np.random.permutation(n)
cut = int(n * 0.7)
train_idx, test_idx = perm[:cut], perm[cut:]

ds_train = ContrastiveDataset(X[train_idx])
loader = DataLoader(ds_train, batch_size=512, shuffle=True, drop_last=True)

class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, 64)
        )
    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=1)

def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2*B, D)
    sim = torch.matmul(z, z.T)
    sim = sim / temperature
    mask = (~torch.eye(2*batch_size, dtype=torch.bool)).to(z.device)
    exp_sim = torch.exp(sim) * mask
    positives = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    positives = torch.cat([ positives, positives ], dim=0)
    denom = exp_sim.sum(dim=1)
    loss = -torch.log(positives / denom).mean()
    return loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = Encoder(X.shape[1]).to(device)
opt = optim.Adam(enc.parameters(), lr=1e-3)

for epoch in range(20):
    total_loss = 0
    for a, b in loader:
        a, b = a.to(device), b.to(device)
        z1, z2 = enc(a), enc(b)
        loss = nt_xent_loss(z1, z2)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d}  loss={total_loss/len(loader):.4f}")

enc.eval()
with torch.no_grad():
    emb_train = enc(torch.from_numpy(X[train_idx]).to(device)).cpu().numpy()
    emb_test  = enc(torch.from_numpy(X[test_idx]).to(device)).cpu().numpy()

clf = LogisticRegression(max_iter=500).fit(emb_train, y[train_idx])
y_pred = clf.predict(emb_test)

print("Downstream Classification")
print(f"Accuracy: {accuracy_score(y[test_idx], y_pred):.4f}")
print(classification_report(y[test_idx], y_pred))
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import itertools

# Load data 
csv_path = r"C:\\Users\\whf80\\Desktop\\Car-Dataset\\CAN-MIRGU-main\\CAN-MIRGU.csv"
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
y = LabelEncoder().fit_transform(df['Class'].astype(str))
class_names = LabelEncoder().fit(df['Class'].astype(str)).classes_

scaler = StandardScaler()
X = scaler.fit_transform(X)

n = len(X)
indices = np.random.permutation(n)
split = int(n * 0.7)
idx_train, idx_test = indices[:split], indices[split:]
X_train, X_test = X[idx_train], X[idx_test]
y_train, y_test = y[idx_train], y[idx_test]

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

# Evaluation 
acc = accuracy_score(y_test, y_pred)
print(f"\nContrastive + Logistic Reg Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion Matrix 
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

# ROC Curve  
y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
y_pred_bin = label_binarize(y_pred, classes=range(len(class_names)))
if y_test_bin.shape[1] > 1:
    plt.figure(figsize=(8,6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.title('ROC Curves'); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend(); plt.tight_layout(); plt.show()
