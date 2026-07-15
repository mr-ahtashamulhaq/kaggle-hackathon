import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def main():
    # Kaggle cloud environment paths
    train_path = '/kaggle/input/playground-series-s6e7/train.csv'
    test_path = '/kaggle/input/playground-series-s6e7/test.csv'
    submission_path = '/kaggle/working/submission.csv'
    
    print("Loading datasets...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    target_col = 'health_condition'
    features = [c for c in train.columns if c not in ['id', target_col]]
    
    # Target encoding
    le = LabelEncoder()
    train[target_col] = le.fit_transform(train[target_col])
    
    categorical_cols = ['diet_type', 'stress_level', 'sleep_quality', 
                        'physical_activity_level', 'smoking_alcohol', 'gender']
    
    # Convert categorical to 'category' dtype for LightGBM
    for col in categorical_cols:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
        
    print("Starting Stratified K-Fold (K=5) training with LightGBM...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros((len(train), len(le.classes_)))
    test_preds = np.zeros((len(test), len(le.classes_)))
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(train[features], train[target_col])):
        print(f"--- Fold {fold+1} ---")
        X_train, y_train = train[features].iloc[train_idx], train[target_col].iloc[train_idx]
        X_valid, y_valid = train[features].iloc[valid_idx], train[target_col].iloc[valid_idx]
        
        clf = lgb.LGBMClassifier(
            objective='multiclass',
            random_state=42,
            class_weight='balanced',
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=-1
        )
        
        # Fit model
        clf.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # Predict on validation set
        val_preds = clf.predict_proba(X_valid)
        oof_preds[valid_idx] = val_preds
        
        # Predict on test set
        test_preds += clf.predict_proba(test[features]) / skf.n_splits
        
        # Evaluate fold
        val_pred_classes = np.argmax(val_preds, axis=1)
        fold_score = balanced_accuracy_score(y_valid, val_pred_classes)
        print(f"Fold {fold+1} Balanced Accuracy: {fold_score:.4f}")
        
    # Overall OOF score
    oof_pred_classes = np.argmax(oof_preds, axis=1)
    oof_score = balanced_accuracy_score(train[target_col], oof_pred_classes)
    print(f"Overall OOF Balanced Accuracy: {oof_score:.4f}")
    
    print("Generating submission file...")
    sub = pd.DataFrame({'id': test['id']})
    test_pred_classes = np.argmax(test_preds, axis=1)
    sub[target_col] = le.inverse_transform(test_pred_classes)
    sub.to_csv(submission_path, index=False)
    print(f"Submission saved successfully to {submission_path}")

if __name__ == '__main__':
    main()
