import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import os

# 固定隨機種子
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

FEATURES = [
    "sex","handId","strengthId","spinId",
    "pointId","actionId","positionId","strikeId","scoreSelf","scoreOther","strikeNumber"]
PAD_TOKEN = 0

# ==========================================
# 1. 預處理 (Preprocessing) - 資料集封裝
# ==========================================
class RallyDataset(Dataset):
    def __init__(self, X, yA, yP, yR, L):
        self.X = torch.tensor(X, dtype=torch.long)      
        self.yA = torch.tensor(yA, dtype=torch.long)    
        self.yP = torch.tensor(yP, dtype=torch.long)    
        self.yR = torch.tensor(yR, dtype=torch.float32) 
        self.L  = torch.tensor(L,  dtype=torch.long)    
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.yA[i], self.yP[i], self.yR[i], self.L[i]

# ==========================================
# 注意力機制模組 (Attention)
# ==========================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_out, mask):
        attn_weights = self.attn(lstm_out).squeeze(-1) 
        attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return context

# ==========================================
# 2. 架構 (Architecture) - Attention-LSTM 模型
# ==========================================
class MultiTaskLSTM(nn.Module):
    def __init__(self, num_tokens_per_feature, n_act, n_pt, emb_dim=32, hidden=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n+1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature])
        self.lstm = nn.LSTM(len(num_tokens_per_feature)*emb_dim, hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.attention = Attention(hidden)

        self.act_head = nn.Linear(hidden, n_act)   
        self.pt_head  = nn.Linear(hidden, n_pt)    
        self.rly_head = nn.Linear(hidden, 1)       

    def forward(self, X, lengths):
        es = [emb(X[:,:,i]) for i,emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.lstm(packed)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=X.size(1))
        o = self.drop(o)

        mask = (X[:,:,0] != PAD_TOKEN)
        attn_context = self.attention(o, mask)

        return self.act_head(o), self.pt_head(o), self.rly_head(attn_context).squeeze(1)

def pad2d(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64); out[:len(a)] = a; return out
def pad1d(a, m, ignore_index=-1):
    out = np.full((m,), ignore_index, dtype=np.int64); out[:len(a)] = a; return out

def main(args):
    # ==========================================
    # 1. 預處理
    # ==========================================
    train = pd.read_csv(args.train).sort_values(["rally_uid","strikeNumber"])
    test  = pd.read_csv(args.test).sort_values(["rally_uid","strikeNumber"])

    train["strikeNumber"] = train["strikeNumber"].clip(0, 40)
    cats = {c: pd.Categorical(train[c]).categories for c in FEATURES}

    def encode_frame(df):
        outs = []
        for col in FEATURES:
            codes = pd.Categorical(df[col], categories=cats[col]).codes + 1
            outs.append(np.asarray(codes, dtype=np.int64))
        return np.stack(outs, axis=1)

    X_list, yA_list, yP_list, yR_list, L_list = [], [], [], [], []
    for rid, g in train.groupby("rally_uid"):
        if len(g) < 2: continue
        X = encode_frame(g)[:-1] 
        yA = g["actionId"].values[1:].astype(np.int64) 
        yP = g["pointId"].values[1:].astype(np.int64)  
        X_list.append(X); yA_list.append(yA); yP_list.append(yP)
        yR_list.append(int(g["serverGetPoint"].iloc[0])); L_list.append(len(X))

    MAXLEN = max(L_list)
    X_all  = np.stack([pad2d(s, MAXLEN) for s in X_list])
    yA_all = np.stack([pad1d(s, MAXLEN) for s in yA_list])
    yP_all = np.stack([pad1d(s, MAXLEN) for s in yP_list])
    yR_all = np.array(yR_list, dtype=np.float32)
    L_all  = np.array(L_list, dtype=np.int64)

    act_classes = np.sort(train["actionId"].unique()); n_act = len(act_classes); act_id2idx = {v:i for i,v in enumerate(act_classes)}
    pt_classes  = np.sort(train["pointId"].unique());  n_pt  = len(pt_classes);  pt_id2idx  = {v:i for i,v in enumerate(pt_classes)}
    yA_all = np.vectorize(act_id2idx.get)(yA_all, -1)
    yP_all = np.vectorize(pt_id2idx.get)(yP_all, -1)

    idx = np.arange(len(X_all))
    tr_idx, va_idx = train_test_split(idx, test_size=args.val_size, random_state=42, stratify=(yR_all>0.5))

    act_counts = np.bincount(yA_all[tr_idx][yA_all[tr_idx]!=-1].ravel(), minlength=n_act) + 1
    act_w = torch.tensor(1.0 / np.sqrt(act_counts), dtype=torch.float32)
    act_w = (act_w * (n_act / act_w.sum()))

    pt_counts = np.bincount(yP_all[tr_idx][yP_all[tr_idx]!=-1].ravel(), minlength=n_pt) + 1
    pt_w = torch.tensor(1.0 / np.sqrt(pt_counts), dtype=torch.float32)
    pt_w = (pt_w * (n_pt / pt_w.sum()))

    num_pos = (yR_all[tr_idx] == 1).sum()
    num_neg = (yR_all[tr_idx] == 0).sum()
    rally_pos_weight = torch.tensor(num_neg / (num_pos + 1e-5), dtype=torch.float32)

    sample_weights = np.where(yR_all[tr_idx] == 1, 1.0 / num_pos, 1.0 / num_neg)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        RallyDataset(X_all[tr_idx], yA_all[tr_idx], yP_all[tr_idx], yR_all[tr_idx], L_all[tr_idx]), 
        batch_size=args.batch, 
        sampler=sampler
    )
    val_loader = DataLoader(
        RallyDataset(X_all[va_idx], yA_all[va_idx], yP_all[va_idx], yR_all[va_idx], L_all[va_idx]), 
        batch_size=128, 
        shuffle=False
    )

    # ==========================================
    # 3. 損失函數與訓練流程
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tokens_per_feature = [len(cats[c]) + 1 for c in FEATURES]
    
    model = MultiTaskLSTM(num_tokens_per_feature, n_act, n_pt, 
                          emb_dim=args.emb, hidden=args.hidden, 
                          num_layers=args.layers, dropout=args.drop).to(device)

    ce_action = nn.CrossEntropyLoss(ignore_index=-1, weight=act_w.to(device))  
    ce_point  = nn.CrossEntropyLoss(ignore_index=-1, weight=pt_w.to(device))   
    bce_rally = nn.BCEWithLogitsLoss(pos_weight=rally_pos_weight.to(device))   

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)

    print(f"\n🚀 模型已升級 (Attention + 提早停止) | 訓練參數: Epochs={args.epochs}, Emb={args.emb}, Hidden={args.hidden}")
    
    best_score = 0
    patience_counter = 0 # 用來計算連續幾個 Epoch 沒有進步
    best_model_path = 'best_model.pth'

    for ep in range(1, args.epochs+1):
        model.train(); run_loss=0.0
        for Xb,yAb,yPb,yRb,Lb in train_loader:
            Xb,yAb,yPb,yRb,Lb = Xb.to(device),yAb.to(device),yPb.to(device),yRb.to(device),Lb.to(device)
            opt.zero_grad()
            la, lp, lr = model(Xb,Lb)
            loss = 0.4*ce_action(la.view(-1,la.size(-1)), yAb.view(-1)) + \
                   0.4*ce_point(lp.view(-1,lp.size(-1)), yPb.view(-1)) + \
                   0.2*bce_rally(lr,yRb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            run_loss += loss.item()*Xb.size(0)

        # 驗證集評估
        model.eval(); allA,allAp,allP,allPp,allR,allRp=[],[],[],[],[],[]
        with torch.no_grad():
            for Xb,yAb,yPb,yRb,Lb in val_loader:
                Xb,yAb,yPb,yRb,Lb = Xb.to(device),yAb.to(device),yPb.to(device),yRb.to(device),Lb.to(device)
                la,lp,lr = model(Xb,Lb)
                
                allR+=yRb.cpu().tolist(); allRp+=torch.sigmoid(lr).cpu().tolist()
                yA_flat=yAb.view(-1).cpu().numpy(); a_pred=la.argmax(-1).view(-1).cpu().numpy()
                yP_flat=yPb.view(-1).cpu().numpy(); p_pred=lp.argmax(-1).view(-1).cpu().numpy()
                mA=(yA_flat!=-1); mP=(yP_flat!=-1) 
                allA+=yA_flat[mA].tolist(); allAp+=a_pred[mA].tolist()
                allP+=yP_flat[mP].tolist(); allPp+=p_pred[mP].tolist()

        f1A=f1_score(allA,allAp,average="macro"); f1P=f1_score(allP,allPp,average="macro")
        auc=roc_auc_score(allR,allRp); final=0.4*f1A+0.4*f1P+0.2*auc
        
        # 根據 Final Score 調整學習率
        scheduler.step(final) 

        print(f"[Epoch {ep}/{args.epochs}] Final Score: {final:.4f} (Act F1: {f1A:.4f}, Pt F1: {f1P:.4f}, AUC: {auc:.4f})")
        
        # [新增] Early Stopping 與儲存最佳權重
        if final > best_score:
            best_score = final
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0 # 分數創新高，歸零計數器
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"\n⚠️ 連續 {args.patience} 個 Epoch 分數沒有進步，觸發 Early Stopping 提早停止訓練！")
            break # 跳出訓練迴圈

    # ==========================================
    # 5. 推論 (Inference)
    # ==========================================
    print(f"\n>>> 載入最佳模型 (最高分: {best_score:.4f}) 進行預測...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

    def pad2d_cap(a, m, pad_val=PAD_TOKEN):
        out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
        T = min(len(a), m); out[:T]=a[:T]; return out, T

    model.eval()
    pred_rows = []

    for rid, g in test.groupby("rally_uid", sort=False):
        Xg = encode_frame(g)
        Xp, T = pad2d_cap(Xg, MAXLEN)

        X_t = torch.tensor(Xp[None,...], dtype=torch.long, device=device)
        L_t = torch.tensor([max(1, T)], dtype=torch.long, device=device)

        with torch.no_grad():
            la, lp, lr = model(X_t, L_t)

        last_t = L_t.item() - 1
        a_idx = int(torch.argmax(la[0, last_t]).item())
        p_idx = int(torch.argmax(lp[0, last_t]).item())
        s_prob = float(torch.sigmoid(lr).item())

        pred_rows.append({
            "rally_uid": int(rid),
            "actionId": int(act_classes[a_idx]),
            "pointId": int(pt_classes[p_idx]),
            "serverGetPoint": round(s_prob, 4)
        })

    out_df = pd.DataFrame(pred_rows)
    final_columns = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    out_df = out_df[final_columns]
    
    out_df.to_csv(args.out, index=False)
    print(f"\n🎉 成功！預測檔案已儲存至: {args.out} (產出筆數: {len(out_df)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--out", default="submission.csv") 
    
    # --- 你可以在這裡手動改參數 ---
    ap.add_argument("--epochs", type=int, default=50)    # 改成了 50 次
    ap.add_argument("--patience", type=int, default=5)   # 容忍連續 5 次沒進步
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--emb", type=int, default=32)       
    ap.add_argument("--hidden", type=int, default=256)   
    ap.add_argument("--layers", type=int, default=2)     
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_size", type=float, default=0.10)
    args = ap.parse_args()
    main(args)