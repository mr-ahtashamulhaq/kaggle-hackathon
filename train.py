import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, TargetEncoder
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
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

    categorical_cols = [
        'diet_type', 'stress_level', 'sleep_quality',
        'physical_activity_level', 'smoking_alcohol', 'gender',
        'stress_sleep', 'diet_activity', 'stress_activity'
    ]
    numeric_te_cols = [
        'sleep_duration', 'heart_rate', 'bmi',
        'calorie_expenditure', 'step_count', 'exercise_duration', 'water_intake'
    ]

    for col in categorical_cols:
        train[col] = train[col].astype('category').cat.codes.astype('int16')
        test[col] = test[col].astype('category').cat.codes.astype('int16')

    features = [c for c in train.columns if c not in ['id', target_col]]
    n_classes = len(le.classes_)
    te_col_names = [f'{col}_TE_cls{c}' for col in numeric_te_cols for c in range(n_classes)]
    class_priors = train[target_col].value_counts(normalize=True).sort_index().values

    def apply_te_and_get_splits(X_tr, y_tr, X_vl, X_ts):
        te = TargetEncoder(target_type='multiclass', smooth='auto', cv=5, random_state=42)
        tr_te = pd.DataFrame(te.fit_transform(X_tr[numeric_te_cols], y_tr), columns=te_col_names, index=X_tr.index)
        vl_te = pd.DataFrame(te.transform(X_vl[numeric_te_cols]), columns=te_col_names, index=X_vl.index)
        ts_te = pd.DataFrame(te.transform(X_ts[numeric_te_cols]), columns=te_col_names, index=X_ts.index)
        X_tr_out = X_tr.drop(columns=numeric_te_cols).join(tr_te)
        X_vl_out = X_vl.drop(columns=numeric_te_cols).join(vl_te)
        X_ts_out = X_ts.drop(columns=numeric_te_cols).join(ts_te)
        return X_tr_out, X_vl_out, X_ts_out

    def prior_correct(raw_proba):
        corrected = raw_proba / class_priors
        return corrected / corrected.sum(axis=1, keepdims=True)

    print(f"Total features before TE: {len(features)}")

    optuna_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=99)
    pre_splits = list(optuna_skf.split(train[features], train[target_col]))

    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20, 200),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 80),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 5.0, log=True),
            'max_bins': trial.suggest_int('max_bins', 64, 255),
            'max_iter': 300,
            'early_stopping': False,
            'verbose': 0,
            'categorical_features': categorical_cols,
            'random_state': 42,
        }

        scores = []
        for train_idx, valid_idx in pre_splits:
            X_tr = train[features].iloc[train_idx].copy()
            y_tr = train[target_col].iloc[train_idx]
            X_vl = train[features].iloc[valid_idx].copy()
            y_vl = train[target_col].iloc[valid_idx]
            X_ts = test[features].copy()

            X_tr, X_vl, X_ts = apply_te_and_get_splits(X_tr, y_tr, X_vl, X_ts)

            hgbc = HistGradientBoostingClassifier(**params)
            hgbc.fit(X_tr, y_tr)

            corrected_val = prior_correct(hgbc.predict_proba(X_vl))
            scores.append(balanced_accuracy_score(y_vl, np.argmax(corrected_val, axis=1)))

        return np.mean(scores)

    print("\nPhase 1: Optuna hyperparameter search (20 trials, 3-fold CV)...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    best_params = study.best_params
    print(f"\nBest Optuna CV: {study.best_value:.4f}")
    print(f"Best params found: {best_params}")

    print("\nPhase 2: Final 7-fold training with best params + 3 seeds...")
    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(train), n_classes))
    test_preds = np.zeros((len(test), n_classes))
    seeds = [42, 123, 456]

    final_hgbc_params = {
        **best_params,
        'max_iter': 300,
        'early_stopping': False,
        'verbose': 0,
        'categorical_features': categorical_cols,
    }

    for fold, (train_idx, valid_idx) in enumerate(skf.split(train[features], train[target_col])):
        print(f"--- Fold {fold + 1} ---")
        X_train = train[features].iloc[train_idx].copy()
        y_train = train[target_col].iloc[train_idx]
        X_valid = train[features].iloc[valid_idx].copy()
        y_valid = train[target_col].iloc[valid_idx]
        X_test = test[features].copy()

        X_train, X_valid, X_test = apply_te_and_get_splits(X_train, y_train, X_valid, X_test)

        fold_val_proba = np.zeros((len(valid_idx), n_classes))
        fold_test_proba = np.zeros((len(X_test), n_classes))

        for seed in seeds:
            hgbc = HistGradientBoostingClassifier(**final_hgbc_params, random_state=seed)
            hgbc.fit(X_train, y_train)
            fold_val_proba += hgbc.predict_proba(X_valid) / len(seeds)
            fold_test_proba += hgbc.predict_proba(X_test) / len(seeds)

        corrected_val = prior_correct(fold_val_proba)
        oof_preds[valid_idx] = corrected_val

        corrected_test = prior_correct(fold_test_proba)
        test_preds += corrected_test / skf.n_splits

        fold_score = balanced_accuracy_score(y_valid, np.argmax(corrected_val, axis=1))
        print(f"   Fold {fold + 1} OOF Balanced Accuracy: {fold_score:.4f}")

    final_oof_score = balanced_accuracy_score(train[target_col], np.argmax(oof_preds, axis=1))
    print(f"\nOverall OOF Balanced Accuracy: {final_oof_score:.4f}")

    print("Generating submission file...")
    sub = pd.DataFrame({'id': test['id']})
    sub[target_col] = le.inverse_transform(np.argmax(test_preds, axis=1))
    sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == '__main__':
    main()
