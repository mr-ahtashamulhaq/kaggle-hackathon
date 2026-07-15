import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, TargetEncoder
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
    # Keep only categorical interactions that are proven to carry signal
    def create_features(df):
        df['stress_sleep'] = df['stress_level'].astype(str) + "_" + df['sleep_quality'].astype(str)
        df['diet_activity'] = df['diet_type'].astype(str) + "_" + df['physical_activity_level'].astype(str)
        # Confirmed strongest signal from community (Rugved Bane, 0.95014)
        df['stress_activity'] = df['stress_level'].astype(str) + "_" + df['physical_activity_level'].astype(str)
        return df

    print("Creating new features...")
    train = create_features(train)
    test = create_features(test)

    target_col = 'health_condition'

    # Label encode target
    le = LabelEncoder()
    train[target_col] = le.fit_transform(train[target_col])

    # Separate numeric and categorical columns
    # HGBC handles categoricals natively via integer encoding
    categorical_cols = [
        'diet_type', 'stress_level', 'sleep_quality',
        'physical_activity_level', 'smoking_alcohol', 'gender',
        'stress_sleep', 'diet_activity', 'stress_activity'
    ]
    # Numeric cols to apply exact-value TargetEncoding (proven +0.0009 by Mark Susol)
    numeric_te_cols = [
        'sleep_duration', 'heart_rate', 'bmi',
        'calorie_expenditure', 'step_count', 'exercise_duration', 'water_intake'
    ]

    # Encode categoricals as integer codes for HGBC
    for col in categorical_cols:
        train[col] = train[col].astype('category').cat.codes.astype('int16')
        test[col] = test[col].astype('category').cat.codes.astype('int16')

    features = [c for c in train.columns if c not in ['id', target_col]]

    print(f"Total features: {len(features)}")
    print("Starting Stratified K-Fold (K=7) with HGBC + exact-value TargetEncoding...")

    # Use 7 folds — more folds = more stable OOF estimate on 690k rows
    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

    oof_preds = np.zeros((len(train), len(le.classes_)))
    test_preds = np.zeros((len(test), len(le.classes_)))

    # Calculate class priors for Prior-Correction (proven better than class_weight)
    class_priors = train[target_col].value_counts(normalize=True).sort_index().values
    print(f"Class priors (sorted by label index): {class_priors.round(4)}")

    for fold, (train_idx, valid_idx) in enumerate(skf.split(train[features], train[target_col])):
        print(f"--- Fold {fold + 1} ---")

        X_train = train[features].iloc[train_idx].copy()
        y_train = train[target_col].iloc[train_idx]
        X_valid = train[features].iloc[valid_idx].copy()
        y_valid = train[target_col].iloc[valid_idx]
        X_test = test[features].copy()

        # --- Exact-Value Target Encoding (sklearn TargetEncoder) ---
        # target_type='multiclass' returns n_features * n_classes columns
        # For 7 numeric features and 3 classes -> 21 TE columns
        te = TargetEncoder(
            target_type='multiclass',
            smooth='auto',
            cv=5,
            random_state=42
        )
        te_col_names = [f'{col}_TE_cls{c}' for col in numeric_te_cols for c in range(len(le.classes_))]

        te_train = pd.DataFrame(
            te.fit_transform(X_train[numeric_te_cols], y_train),
            columns=te_col_names,
            index=X_train.index
        )
        te_valid = pd.DataFrame(
            te.transform(X_valid[numeric_te_cols]),
            columns=te_col_names,
            index=X_valid.index
        )
        te_test_arr = pd.DataFrame(
            te.transform(X_test[numeric_te_cols]),
            columns=te_col_names,
            index=X_test.index
        )

        # Drop raw numerics and attach TE columns
        X_train = X_train.drop(columns=numeric_te_cols).join(te_train)
        X_valid = X_valid.drop(columns=numeric_te_cols).join(te_valid)
        X_test = X_test.drop(columns=numeric_te_cols).join(te_test_arr)

        # --- HistGradientBoostingClassifier with Seed Averaging ---
        # Proven breakthrough model (Mark Susol v0.7, 0.95036 LB)
        # Fix 1: No internal early-stopping validation split — train on full fold data.
        #         Model was stopping at 127-167 iters, so max_iter=220 is sufficient.
        # Fix 2: Seed averaging (3 seeds) — reduces the 0.005 fold variance we observed
        #         without changing model architecture or creating ensemble diversity trap.
        seeds = [42, 123, 456]
        fold_val_proba = np.zeros((len(valid_idx), len(le.classes_)))
        fold_test_proba = np.zeros((len(X_test), len(le.classes_)))

        for seed in seeds:
            hgbc = HistGradientBoostingClassifier(
                max_iter=220,
                learning_rate=0.05,
                max_leaf_nodes=63,
                max_depth=None,
                min_samples_leaf=20,
                l2_regularization=0.1,
                early_stopping=False,   # use all fold training data
                random_state=seed,
                verbose=0,
                categorical_features=categorical_cols
            )
            hgbc.fit(X_train, y_train)
            fold_val_proba += hgbc.predict_proba(X_valid) / len(seeds)
            fold_test_proba += hgbc.predict_proba(X_test) / len(seeds)

        # Prior-Correction: divide raw probabilities by class priors, then renormalize
        corrected_val = fold_val_proba / class_priors
        corrected_val = corrected_val / corrected_val.sum(axis=1, keepdims=True)
        oof_preds[valid_idx] = corrected_val

        corrected_test = fold_test_proba / class_priors
        corrected_test = corrected_test / corrected_test.sum(axis=1, keepdims=True)
        test_preds += corrected_test / skf.n_splits

        fold_score = balanced_accuracy_score(y_valid, np.argmax(corrected_val, axis=1))
        print(f"   Fold {fold + 1} OOF Balanced Accuracy: {fold_score:.4f}")

    # --- Final OOF Score ---
    final_oof_score = balanced_accuracy_score(train[target_col], np.argmax(oof_preds, axis=1))
    print(f"\nOverall OOF Balanced Accuracy: {final_oof_score:.4f}")

    # --- Generate Submission ---
    print("Generating submission file...")
    sub = pd.DataFrame({'id': test['id']})
    sub[target_col] = le.inverse_transform(np.argmax(test_preds, axis=1))
    sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == '__main__':
    main()
