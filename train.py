import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
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
    xgb_oof_preds = np.zeros((len(train), len(le.classes_)))
    lgb_test_preds = np.zeros((len(test), len(le.classes_)))
    cb_test_preds = np.zeros((len(test), len(le.classes_)))
    xgb_test_preds = np.zeros((len(test), len(le.classes_)))
    
    # Convert categories to str for CatBoost compatibility just in case
    for col in categorical_cols:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        # Re-convert to category for LGBM natively
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
        
    # Calculate class priors for Prior-Correction
    class_priors = train[target_col].value_counts(normalize=True).sort_index().values
        
    num_cols = ['sleep_duration', 'heart_rate', 'bmi', 'calorie_expenditure', 'step_count', 'exercise_duration', 'water_intake']
    for col in num_cols:
        train[f'{col}_bin'] = pd.qcut(train[col], q=15, labels=False, duplicates='drop').fillna(-1)
        test[f'{col}_bin'] = pd.qcut(test[col], q=15, labels=False, duplicates='drop').fillna(-1)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(train[features], train[target_col])):
        print(f"--- Fold {fold+1} ---")
        X_train = train[features].iloc[train_idx].copy()
        y_train = train[target_col].iloc[train_idx]
        X_valid = train[features].iloc[valid_idx].copy()
        y_valid = train[target_col].iloc[valid_idx]
        X_test = test[features].copy()
        
        # Dynamic Target Encoding inside the fold to prevent leakage
        train_bins = train[[f'{col}_bin' for col in num_cols]].iloc[train_idx]
        valid_bins = train[[f'{col}_bin' for col in num_cols]].iloc[valid_idx]
        test_bins = test[[f'{col}_bin' for col in num_cols]]
        
        te_features = []
        for col in num_cols:
            bin_col = f'{col}_bin'
            for class_val in range(len(le.classes_)):
                feat_name = f'{col}_TE_class_{class_val}'
                means = (y_train == class_val).groupby(train_bins[bin_col]).mean()
                global_mean = (y_train == class_val).mean()
                
                X_train[feat_name] = train_bins[bin_col].map(means).fillna(global_mean)
                X_valid[feat_name] = valid_bins[bin_col].map(means).fillna(global_mean)
                X_test[feat_name] = test_bins[bin_col].map(means).fillna(global_mean)
                te_features.append(feat_name)
                
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
        
        raw_lgb_test = lgb_clf.predict_proba(X_test)
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
        
        raw_cb_test = cb_clf.predict_proba(X_test)
        test_cb_pred = raw_cb_test / class_priors
        test_cb_pred = test_cb_pred / test_cb_pred.sum(axis=1, keepdims=True)
        cb_test_preds += test_cb_pred / skf.n_splits
        print("-> CatBoost finished.")
        
        # --- XGBoost ---
        print("-> Training XGBoost...")
        xgb_clf = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            device='cuda',
            enable_categorical=True,
            early_stopping_rounds=50
        )
        try:
            xgb_clf.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=200
            )
        except Exception as e:
            print(f"   GPU XGBoost failed, falling back to CPU...")
            xgb_clf.set_params(device='cpu')
            xgb_clf.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=200
            )
            
        # Prior-Correction for XGBoost
        raw_xgb_val = xgb_clf.predict_proba(X_valid)
        fold_xgb_val = raw_xgb_val / class_priors
        fold_xgb_val = fold_xgb_val / fold_xgb_val.sum(axis=1, keepdims=True)
        xgb_oof_preds[valid_idx] = fold_xgb_val
        
        raw_xgb_test = xgb_clf.predict_proba(X_test)
        test_xgb_pred = raw_xgb_test / class_priors
        test_xgb_pred = test_xgb_pred / test_xgb_pred.sum(axis=1, keepdims=True)
        xgb_test_preds += test_xgb_pred / skf.n_splits
        print("-> XGBoost finished.")
        
        # Evaluate fold individually
        lgb_score = balanced_accuracy_score(y_valid, np.argmax(fold_lgb_val, axis=1))
        cb_score = balanced_accuracy_score(y_valid, np.argmax(fold_cb_val, axis=1))
        xgb_score = balanced_accuracy_score(y_valid, np.argmax(fold_xgb_val, axis=1))
        blend_score = balanced_accuracy_score(y_valid, np.argmax(0.33 * fold_lgb_val + 0.33 * fold_cb_val + 0.34 * fold_xgb_val, axis=1))
        
        print(f"Fold {fold+1} Scores -> LGBM: {lgb_score:.4f} | CB: {cb_score:.4f} | XGB: {xgb_score:.4f} | Mean Blend: {blend_score:.4f}")
        
    # --- Post-Training Optimization ---
    print("\nOptimizing 3-way blend weights using Nelder-Mead...")
    
    def loss_func(weights):
        w1, w2, w3 = weights
        if w1 < 0 or w2 < 0 or w3 < 0 or (w1 + w2 + w3) == 0:
            return 9999.0 # heavily penalize invalid weights
        # normalize
        total_w = w1 + w2 + w3
        w1, w2, w3 = w1/total_w, w2/total_w, w3/total_w
        blended = w1 * lgb_oof_preds + w2 * cb_oof_preds + w3 * xgb_oof_preds
        # Negate because minimize finds minimum, we want maximum balanced accuracy
        return -balanced_accuracy_score(train[target_col], np.argmax(blended, axis=1))
    
    res = minimize(loss_func, [0.33, 0.33, 0.34], method='Nelder-Mead', tol=1e-4)
    best_w1, best_w2, best_w3 = res.x
    total_best_w = best_w1 + best_w2 + best_w3
    best_w1, best_w2, best_w3 = best_w1/total_best_w, best_w2/total_best_w, best_w3/total_best_w
    
    print(f"Optimal Weights -> LGBM: {best_w1:.4f} | CB: {best_w2:.4f} | XGB: {best_w3:.4f}")
    
    # Calculate final OOF score
    final_oof_preds = best_w1 * lgb_oof_preds + best_w2 * cb_oof_preds + best_w3 * xgb_oof_preds
    final_oof_score = balanced_accuracy_score(train[target_col], np.argmax(final_oof_preds, axis=1))
    print(f"Overall OOF Balanced Accuracy (Optimized): {final_oof_score:.4f}")
    
    print("Generating submission file...")
    sub = pd.DataFrame({'id': test['id']})
    final_test_preds = best_w1 * lgb_test_preds + best_w2 * cb_test_preds + best_w3 * xgb_test_preds
    sub[target_col] = le.inverse_transform(np.argmax(final_test_preds, axis=1))
    sub.to_csv(submission_path, index=False)
    print(f"Submission saved successfully to {submission_path}")

if __name__ == '__main__':
    main()
