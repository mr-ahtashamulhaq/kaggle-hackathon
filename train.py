import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import LabelEncoder
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
        # Prevent division by zero
        df['calorie_per_step'] = df['calorie_expenditure'] / (df['step_count'] + 1)
        df['steps_per_exercise'] = df['step_count'] / (df['exercise_duration'] + 1)
        df['water_per_calorie'] = df['water_intake'] / (df['calorie_expenditure'] + 1)
        
        # Categorical interactions
        df['stress_sleep'] = df['stress_level'].astype(str) + "_" + df['sleep_quality'].astype(str)
        df['diet_activity'] = df['diet_type'].astype(str) + "_" + df['physical_activity_level'].astype(str)
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
                        'stress_sleep', 'diet_activity']
    
    # Convert categorical to 'category' dtype for LightGBM
    for col in categorical_cols:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
        
    print("Starting Stratified K-Fold (K=5) training with LightGBM & CatBoost Blend...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros((len(train), len(le.classes_)))
    test_preds = np.zeros((len(test), len(le.classes_)))
    
    # Convert categories to str for CatBoost compatibility just in case
    for col in categorical_cols:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        # Re-convert to category for LGBM natively
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
        
    for fold, (train_idx, valid_idx) in enumerate(skf.split(train[features], train[target_col])):
        print(f"--- Fold {fold+1} ---")
        X_train, y_train = train[features].iloc[train_idx], train[target_col].iloc[train_idx]
        X_valid, y_valid = train[features].iloc[valid_idx], train[target_col].iloc[valid_idx]
        
        # --- LightGBM ---
        lgb_clf = lgb.LGBMClassifier(
            objective='multiclass',
            random_state=42,
            class_weight='balanced',
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
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        lgb_val_preds = lgb_clf.predict_proba(X_valid)
        lgb_test_preds = lgb_clf.predict_proba(test[features])
        
        # --- CatBoost ---
        cb_clf = cb.CatBoostClassifier(
            loss_function='MultiClass',
            eval_metric='MultiClass',
            random_seed=42,
            auto_class_weights='Balanced',
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            cat_features=categorical_cols,
            verbose=False,
            early_stopping_rounds=50
        )
        cb_clf.fit(X_train, y_train, eval_set=(X_valid, y_valid))
        cb_val_preds = cb_clf.predict_proba(X_valid)
        cb_test_preds = cb_clf.predict_proba(test[features])
        
        # --- Blend (50/50) ---
        val_preds = 0.5 * lgb_val_preds + 0.5 * cb_val_preds
        oof_preds[valid_idx] = val_preds
        
        test_preds += (0.5 * lgb_test_preds + 0.5 * cb_test_preds) / skf.n_splits
        
        # Evaluate fold
        val_pred_classes = np.argmax(val_preds, axis=1)
        fold_score = balanced_accuracy_score(y_valid, val_pred_classes)
        print(f"Fold {fold+1} Blended Balanced Accuracy: {fold_score:.4f}")
        
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
