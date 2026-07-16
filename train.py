import os
import gc
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TabDataset(Dataset):
    def __init__(self, x_cat, x_num, y=None):
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x_cat)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_cat[idx], self.x_num[idx], self.y[idx]
        return self.x_cat[idx], self.x_num[idx]


class FTTransformer(nn.Module):
    def __init__(self, cat_cards, n_num, d=64, heads=8, layers=4, drop=0.1, n_cls=3):
        super().__init__()
        self.cat_embs = nn.ModuleList([nn.Embedding(c + 2, d) for c in cat_cards])
        self.num_toks = nn.ModuleList([nn.Linear(1, d) for _ in range(n_num)])
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 4,
            dropout=drop, batch_first=True, norm_first=True
        )
        self.tfm = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_cls))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.cls_tok)

    def forward(self, cat, num):
        B = cat.shape[0]
        cat_t = torch.stack([self.cat_embs[i](cat[:, i]) for i in range(cat.shape[1])], dim=1)
        num_t = torch.stack([self.num_toks[i](num[:, i:i+1]) for i in range(num.shape[1])], dim=1)
        x = torch.cat([self.cls_tok.expand(B, -1, -1), cat_t, num_t], dim=1)
        return self.head(self.tfm(x)[:, 0])


def get_probas(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            xc, xn = batch[0].to(DEVICE), batch[1].to(DEVICE)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(xc, xn)
            preds.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.vstack(preds)


def main():
    print(f"Device: {DEVICE}")
    submission_path = '/kaggle/working/submission.csv'
    train_path = "/kaggle/input/competitions/playground-series-s6e7/train.csv"
    test_path = "/kaggle/input/competitions/playground-series-s6e7/test.csv"

    print("Loading datasets...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    def create_features(df):
        df['stress_sleep'] = df['stress_level'].astype(str) + "_" + df['sleep_quality'].astype(str)
        df['diet_activity'] = df['diet_type'].astype(str) + "_" + df['physical_activity_level'].astype(str)
        df['stress_activity'] = df['stress_level'].astype(str) + "_" + df['physical_activity_level'].astype(str)
        return df

    print("Creating new features...")
    train = create_features(train)
    test = create_features(test)

    target_col = 'health_condition'
    le = LabelEncoder()
    train[target_col] = le.fit_transform(train[target_col])
    n_classes = len(le.classes_)

    cat_cols = [
        'diet_type', 'stress_level', 'sleep_quality', 'physical_activity_level',
        'smoking_alcohol', 'gender', 'stress_sleep', 'diet_activity', 'stress_activity'
    ]
    num_cols = [
        'sleep_duration', 'heart_rate', 'bmi',
        'calorie_expenditure', 'step_count', 'exercise_duration', 'water_intake'
    ]

    # NaN indicators BEFORE imputation — the MNAR signal
    for col in num_cols:
        train[f'{col}_nan'] = train[col].isna().astype(np.float32)
        test[f'{col}_nan'] = test[col].isna().astype(np.float32)
    nan_cols = [f'{col}_nan' for col in num_cols]

    # Mean imputation using training set mean only
    for col in num_cols:
        m = train[col].mean()
        train[col] = train[col].fillna(m)
        test[col] = test[col].fillna(m)

    # Z-score normalization (fit on train, apply to both)
    for col in num_cols:
        mu, std = train[col].mean(), train[col].std() + 1e-8
        train[col] = ((train[col] - mu) / std).astype(np.float32)
        test[col] = ((test[col] - mu) / std).astype(np.float32)

    # Integer-encode categoricals (fit on combined train+test for vocabulary)
    cat_cardinalities = []
    for col in cat_cols:
        combined = pd.concat([train[col].astype(str), test[col].astype(str)], ignore_index=True)
        cats = combined.astype('category').cat.categories
        cat_cardinalities.append(len(cats))
        train[col] = pd.Categorical(train[col].astype(str), categories=cats).codes
        test[col] = pd.Categorical(test[col].astype(str), categories=cats).codes

    all_num_cols = num_cols + nan_cols
    n_numeric = len(all_num_cols)

    X_cat = train[cat_cols].values.astype(np.int64)
    X_num = train[all_num_cols].values.astype(np.float32)
    y_all = train[target_col].values.astype(np.int64)

    X_cat_test = test[cat_cols].values.astype(np.int64)
    X_num_test = test[all_num_cols].values.astype(np.float32)

    class_priors = np.bincount(y_all) / len(y_all)
    cw = torch.tensor(1.0 / (class_priors * n_classes), dtype=torch.float32).to(DEVICE)
    print(f"Class priors: {class_priors.round(4)}")
    print(f"Features: {len(cat_cols)} cat + {n_numeric} num = {len(cat_cols) + n_numeric} total")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(train), n_classes))
    test_preds = np.zeros((len(test), n_classes))
    seeds = [42, 123]

    print("\nStarting 5-fold FT-Transformer training (2 seeds per fold)...")

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_cat, y_all)):
        print(f"\n--- Fold {fold + 1}/5 ---")
        fold_val_proba = np.zeros((len(vl_idx), n_classes))
        fold_test_proba = np.zeros((len(X_cat_test), n_classes))

        for seed in seeds:
            set_seed(seed)

            tr_ds = TabDataset(X_cat[tr_idx], X_num[tr_idx], y_all[tr_idx])
            vl_ds = TabDataset(X_cat[vl_idx], X_num[vl_idx], y_all[vl_idx])
            ts_ds = TabDataset(X_cat_test, X_num_test)

            tr_ld = DataLoader(tr_ds, batch_size=4096, shuffle=True, num_workers=2, pin_memory=True)
            vl_ld = DataLoader(vl_ds, batch_size=8192, shuffle=False, num_workers=2, pin_memory=True)
            ts_ld = DataLoader(ts_ds, batch_size=8192, shuffle=False, num_workers=2, pin_memory=True)

            model = FTTransformer(
                cat_cards=cat_cardinalities, n_num=n_numeric,
                d=64, heads=8, layers=4, drop=0.1, n_cls=n_classes
            ).to(DEVICE)

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)
            scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

            best_val_loss = float('inf')
            best_state = None
            patience, no_improve = 8, 0

            for epoch in range(60):
                model.train()
                for xc, xn, y in tr_ld:
                    xc, xn, y = xc.to(DEVICE), xn.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        loss = F.cross_entropy(model(xc, xn), y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()

                model.eval()
                vl_loss = 0.0
                with torch.no_grad():
                    for xc, xn, y in vl_ld:
                        xc, xn, y = xc.to(DEVICE), xn.to(DEVICE), y.to(DEVICE)
                        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                            vl_loss += F.cross_entropy(model(xc, xn), y).item()
                vl_loss /= len(vl_ld)

                if vl_loss < best_val_loss:
                    best_val_loss = vl_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if (epoch + 1) % 10 == 0:
                    print(f"  [seed={seed}] Epoch {epoch+1}: val_loss={vl_loss:.5f} (best={best_val_loss:.5f})")

                if no_improve >= patience:
                    print(f"  [seed={seed}] Early stop @ epoch {epoch+1}")
                    break

            model.load_state_dict(best_state)
            fold_val_proba += get_probas(model, vl_ld) / len(seeds)
            fold_test_proba += get_probas(model, ts_ld) / len(seeds)

            del model, optimizer, scheduler, scaler
            del tr_ds, vl_ds, ts_ds, tr_ld, vl_ld, ts_ld, best_state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        corrected_val = fold_val_proba / class_priors
        corrected_val /= corrected_val.sum(axis=1, keepdims=True)
        oof_preds[vl_idx] = corrected_val

        corrected_test = fold_test_proba / class_priors
        corrected_test /= corrected_test.sum(axis=1, keepdims=True)
        test_preds += corrected_test / skf.n_splits

        fold_score = balanced_accuracy_score(y_all[vl_idx], np.argmax(corrected_val, axis=1))
        print(f"Fold {fold + 1} OOF Balanced Accuracy: {fold_score:.4f}")

    final_oof = balanced_accuracy_score(y_all, np.argmax(oof_preds, axis=1))
    print(f"\nOverall OOF Balanced Accuracy: {final_oof:.4f}")

    print("Generating submission file...")
    sub = pd.DataFrame({'id': test['id']})
    sub[target_col] = le.inverse_transform(np.argmax(test_preds, axis=1))
    sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == '__main__':
    main()
