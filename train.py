import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def main():
    # Kaggle cloud environment paths
    submission_path = '/kaggle/working/submission.csv'
    train_path = "/kaggle/input/competitions/playground-series-s6e7/train.csv"
    test_path = "/kaggle/input/competitions/playground-series-s6e7/test.csv"
    print("Loading datasets...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # --- FEATURE ENGINEERING ---
    def create_features(df):
        # Removed numerical ratios due to known train/test distribution shifts
        # Categorical interactions
        df['stress_sleep'] = df['stress_level'].astype(str) + "_" + df['sleep_quality'].astype(str)
        df['diet_activity'] = df['diet_type'].astype(str) + "_" + df['physical_activity_level'].astype(str)
        df['stress_activity'] = df['stress_level'].astype(str) + "_" + df['physical_activity_level'].astype(str)
        return df

    print("Creating new features...")
    train = create_features(train)
    test = create_features(test)
    
    target_col = 'health_condition'
    features = [c for c in train.columns if c not in ['id', target_col]]
    
    # Target encoding
    le = LabelEncoder()
    train[target_col] = le.fit_transform(train[target_col])
    
    categorical_cols = ['diet_type', 'stress_level', 'sleep_quality', 
                        'physical_activity_level', 'smoking_alcohol', 'gender',
                        'stress_sleep', 'diet_activity', 'stress_activity']
    
    # Convert categorical to 'category' dtype for LightGBM
    for col in categorical_cols:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
        
    print("Starting Stratified K-Fold (K=5) training with LightGBM & CatBoost Blend...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    lgb_oof_preds = np.zeros((len(train), len(le.classes_)))
    cb_oof_preds = np.zeros((len(train), len(le.classes_)))
    lgb_test_preds = np.zeros((len(test), len(le.classes_)))
    cb_test_preds = np.zeros((len(test), len(le.classes_)))
    
    # Convert categories to str for CatBoost compatibility just in case
    for col in categorical_cols:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        # Re-convert to category for LGBM natively
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
        
    # Calculate class priors for Prior-Correction
    class_priors = train[target_col].value_counts(normalize=True).sort_index().values
        
    for fold, (train_idx, valid_idx) in enumerate(skf.split(train[features], train[target_col])):
        print(f"--- Fold {fold+1} ---")
        X_train, y_train = train[features].iloc[train_idx], train[target_col].iloc[train_idx]
        X_valid, y_valid = train[features].iloc[valid_idx], train[target_col].iloc[valid_idx]
        
        # --- LightGBM ---
        print("-> Training LightGBM...")
        lgb_clf = lgb.LGBMClassifier(
            objective='multiclass',
            random_state=42,
            # Removed class_weight='balanced' to use Prior-Correction instead
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=8,
            colsample_bytree=0.8,
            subsample=0.8,
            n_jobs=-1
        )
        lgb_clf.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=200)
            ]
        )
        
        # Prior-Correction for LightGBM
        raw_lgb_val = lgb_clf.predict_proba(X_valid)
        fold_lgb_val = raw_lgb_val / class_priors
        fold_lgb_val = fold_lgb_val / fold_lgb_val.sum(axis=1, keepdims=True)
        lgb_oof_preds[valid_idx] = fold_lgb_val
        
        raw_lgb_test = lgb_clf.predict_proba(test[features])
        test_lgb_pred = raw_lgb_test / class_priors
        test_lgb_pred = test_lgb_pred / test_lgb_pred.sum(axis=1, keepdims=True)
        lgb_test_preds += test_lgb_pred / skf.n_splits
        print("-> LightGBM finished.")
        
        # --- CatBoost ---
        print("-> Training CatBoost...")
        cb_params = {
            'loss_function': 'MultiClass',
            'eval_metric': 'MultiClass',
            'random_seed': 42,
            # Removed auto_class_weights='Balanced' to use Prior-Correction instead
            'iterations': 1500,
            'learning_rate': 0.03,
            'depth': 6,
            'cat_features': categorical_cols,
            'early_stopping_rounds': 50,
            'verbose': 200
        }
        
        try:
            # Try to use GPU if available on Kaggle
            cb_clf = cb.CatBoostClassifier(**cb_params, task_type='GPU')
            cb_clf.fit(X_train, y_train, eval_set=(X_valid, y_valid))
        except cb.CatBoostError:
            print("   GPU not detected for CatBoost. Falling back to CPU...")
            cb_clf = cb.CatBoostClassifier(**cb_params)
            cb_clf.fit(X_train, y_train, eval_set=(X_valid, y_valid))
            
        # Prior-Correction for CatBoost
        raw_cb_val = cb_clf.predict_proba(X_valid)
        fold_cb_val = raw_cb_val / class_priors
        fold_cb_val = fold_cb_val / fold_cb_val.sum(axis=1, keepdims=True)
        cb_oof_preds[valid_idx] = fold_cb_val
        
        raw_cb_test = cb_clf.predict_proba(test[features])
        test_cb_pred = raw_cb_test / class_priors
        test_cb_pred = test_cb_pred / test_cb_pred.sum(axis=1, keepdims=True)
        cb_test_preds += test_cb_pred / skf.n_splits
        print("-> CatBoost finished.")
        
        # Evaluate fold individually
        lgb_score = balanced_accuracy_score(y_valid, np.argmax(fold_lgb_val, axis=1))
        cb_score = balanced_accuracy_score(y_valid, np.argmax(fold_cb_val, axis=1))
        blend_score = balanced_accuracy_score(y_valid, np.argmax(0.5 * fold_lgb_val + 0.5 * fold_cb_val, axis=1))
        
        print(f"Fold {fold+1} Scores -> LGBM: {lgb_score:.4f} | CB: {cb_score:.4f} | 50/50 Blend: {blend_score:.4f}")
        
    # --- Post-Training Optimization ---
    print("\nOptimizing blend weights using Nelder-Mead...")
    
    def loss_func(weights):
        w1, w2 = weights
        if w1 < 0 or w2 < 0 or (w1 + w2) == 0:
            return 9999.0 # heavily penalize invalid weights
        # normalize
        w1, w2 = w1 / (w1+w2), w2 / (w1+w2)
        blended = w1 * lgb_oof_preds + w2 * cb_oof_preds
        # Negate because minimize finds minimum, we want maximum balanced accuracy
        return -balanced_accuracy_score(train[target_col], np.argmax(blended, axis=1))
    
    res = minimize(loss_func, [0.5, 0.5], method='Nelder-Mead', tol=1e-4)
    best_w1, best_w2 = res.x
    best_w1, best_w2 = best_w1 / (best_w1+best_w2), best_w2 / (best_w1+best_w2)
    
    print(f"Optimal Weights -> LGBM: {best_w1:.4f} | CB: {best_w2:.4f}")
    
    # Calculate final OOF score
    final_oof_preds = best_w1 * lgb_oof_preds + best_w2 * cb_oof_preds
    final_oof_score = balanced_accuracy_score(train[target_col], np.argmax(final_oof_preds, axis=1))
    print(f"Overall OOF Balanced Accuracy (Optimized): {final_oof_score:.4f}")
    
    print("Generating submission file...")
    sub = pd.DataFrame({'id': test['id']})
    final_test_preds = best_w1 * lgb_test_preds + best_w2 * cb_test_preds
    sub[target_col] = le.inverse_transform(np.argmax(final_test_preds, axis=1))
    sub.to_csv(submission_path, index=False)
    print(f"Submission saved successfully to {submission_path}")

if __name__ == '__main__':
    main()
