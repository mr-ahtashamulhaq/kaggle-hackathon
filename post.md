# Breaking the 0.950 Ceiling: The 2 Levers That Actually Work

After hitting a wall at 0.949 with various XGBoost and LightGBM ensembles, I finally broke through 0.950 on the Public LB (0.95017). Through deep experimentation and studying the behavior of models under train/test shift, I realized that getting past 0.950 requires doing less, not more. 

Here are the specific, actionable technical changes that moved the needle.

### 1. Stop Ensembling and Blending
We often rely on stackers and optimized blends (like Nelder Mead) to squeeze out the last bit of performance. However, in this dataset, features like `water_intake`, `calorie_expenditure`, and `bmi` have a distribution shift between train and test. 

When you blend multiple models, your meta weights optimize on the correlation structure of the *train* distribution. Because of the shift, this structure changes in the test set, meaning your optimized ensemble actually overfits the Public LB! Using a single, highly robust model transfers cleanly without this "shift fragile" layer.

### 2. Exact Value Target Encoding with HGBC
Target encoding is powerful, but how you apply it matters. Instead of binning numeric columns, using `sklearn.preprocessing.TargetEncoder` on the exact numeric values yields a much stronger signal. 

Crucially, you must use inner cross validation (`cv=5`) when applying it to prevent target leakage. Pairing this exact value target encoding with `HistGradientBoostingClassifier` (which handles missing values natively with a different splitting strategy than LightGBM or XGBoost) provided the exact diversity and signal needed to break the ceiling.

### 3. Prior Correction > Class Weights
Setting `class_weight='balanced'` distorts the tree building process. Instead, train the model normally and apply **Prior Correction** post training. Simply divide your raw predicted probabilities by the class priors of the training set, and then renormalize them to sum to 1. This consistently provides better Balanced Accuracy.

### The Code
I have put together a clean, single model notebook demonstrating these principles in action without the fluff. 

[Check out the 0.950+ Notebook Here](YOUR_NOTEBOOK_LINK_HERE)

Hope this helps those of you stuck in the 0.949 range! Let me know your thoughts or if you have found other levers that hold up out of fold.
