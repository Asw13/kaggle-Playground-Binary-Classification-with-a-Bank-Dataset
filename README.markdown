# Kaggle Tabular Playground Series - Season 5, Episode 8 (0.96308 ROC AUC)

Hey, this is my work on the Kaggle Tabular Playground Series - Season 5, Episode 8 competition ([link](https://www.kaggle.com/competitions/playground-series-s5e8/leaderboard)) to predict if a customer will subscribe (`y=1`) or not (`y=0`) to a marketing campaign, scored by ROC AUC. I got a 0.96308 score on the public leaderboard! The dataset has features like `age`, `job`, `balance`, and `duration`, but only ~12% have `y=1`, so it’s imbalanced. I used visualizations to find patterns, built a pipeline with preprocessing, picked LightGBM, tuned it, and submitted predictions. I’m not great at explaining, so I’ve broken it down into simple points, mixing what I did and why, to make it clear. This README explains my approach, and the code is in `kaggle_submission_pipeline_with_eda.py`.

## My Approach

- **Checked Data with Stats and Charts to Find Imbalance, Outliers, and Subscription Patterns**:
  - I used `describe()` on training data to see stats for `age`, `balance`, `duration`, etc., and found ~12% `y=1` (imbalanced) and outliers (like high `balance`). I chose `RobustScaler` because it handles outliers better than standard scaling, which is sensitive to them.
  - I ran `info()` and saw 9 integer and 9 object columns with no missing values, so I used `LabelEncoder` for categoricals to keep it simple for LightGBM and improve accuracy.
  - I made histograms for `job`, `marital`, `education`, `default`, and `loan`. `Management` had fewer subscriptions, `single` people had a unique subscription pattern, `tertiary` and `secondary` education had more subscriptions, and no `default` or `loan` meant more subscriptions. I made features like `job_manager` to capture these.
  - I used box plots for `age`, `balance`, `duration`, etc., and saw outliers but too many to remove, so I used `RobustScaler` and binning.
  - I checked test data with `describe()` and saw similar outliers, so I kept preprocessing consistent.
  - I made a heatmap and saw `duration` has a 0.52 correlation with `y`, so I focused on it with features like `duration_log`.

- **Preprocessed Data to Boost Accuracy with Subscription Patterns**:
  - I built a `Preprocessing` class to make `duration_log`, `duration_sqrt`, and `duration_per_campaign` because `duration` is key (0.52 correlation), and these capture non-linear patterns for better accuracy.
  - I turned `month` and `day` into cyclic features (`month_sin`, `month_cos`) because they repeat, helping the model see patterns and boost ROC AUC.
  - I binned `age` (17–37, 37–56), `duration` (319–645 for subscriptions), etc., based on histograms where `y=1` was high, to simplify data and handle outliers.
  - I made `contact_status` from `pdays` (50–200 days for subscriptions) because histograms showed these customers subscribe more.
  - I used `RobustScaler` on `duration_per_campaign`, etc., because box plots showed outliers, and it’s less sensitive than standard scaling.
  - I used `LabelEncoder` for `job`, `education`, etc., for LightGBM because it’s efficient and improves accuracy. For other models, I used `OneHotEncoder` for nominal columns (`job`, `marital`) but `LabelEncoder` for ordinal (`education`).
  - I added `duration_favorable` (duration 319–645) and `pdays_favorable` (pdays 50–200) because histograms showed more subscriptions there, and models struggled with `y=1` without them.
  - I combined `poutcome_success`, `duration_favorable`, and `pdays_favorable` into `poutcome_duration_pdays` because these had high impact in visualizations, boosting accuracy.

- **Tested Models to Pick LightGBM for Best ROC AUC**:
  - I made an `ImbalanceModelTrainer` to test Logistic Regression, Random Forest, LightGBM, and XGBoost with stratified splits for 12% `y=1`. I used `class_weight='balanced'` because of imbalance, and LightGBM was best at catching subscriptions with my features.

- **Tuned LightGBM for Max ROC AUC**:
  - I used Bayesian optimization (`BayesSearchCV`) with 5-fold cross-validation to tune LightGBM parameters like `learning_rate=0.01078`, `num_leaves=379`, and `max_depth=15` because defaults weren’t good enough, and tuning maximizes ROC AUC.

- **Trained LightGBM for Robust Predictions**:
  - I wrote a `train_lightgbm` function with 5-fold cross-validation, tuned parameters, and `class_weight='balanced'` for 12% `y=1`. I used early stopping (500 rounds) to avoid overfitting and averaged predictions for a stable 0.96308 ROC AUC.

- **Submitted Predictions for 0.96308 ROC AUC**:
  - I saved averaged test predictions with test `id` as `submission.csv` (columns `id`, `y`) because Kaggle needs this format, and averaging gave me 0.96308 on the public leaderboard.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Install Dependencies**:
   Install Python packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn lightgbm
   ```

3. **Download Data**:
   - Get `train.csv` and `test.csv` from the [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s5e8/data).
   - Place them in the project folder.

4. **Run the Pipeline**:
   - Run `kaggle_submission_pipeline_with_eda.py`:
     ```bash
     python kaggle_submission_pipeline_with_eda.py
     ```
   - Outputs: Visualizations, `submission.csv`, and ROC AUC scores.

5. **Check Results**:
   - Look at `submission.csv` for predictions.
   - Check console for train/validation ROC AUC scores.

## Results
- **Kaggle Score**: 0.96308 ROC AUC on the public leaderboard.
- **Cross-Validation Scores**: Run the pipeline to see `train_scores` and `val_scores`. Example chart (replace with your scores):
  ```chartjs
  {
    "type": "bar",
    "data": {
      "labels": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
      "datasets": [
        {
          "label": "Train ROC AUC",
          "data": [0, 0, 0, 0, 0],  // Replace with train_scores
          "backgroundColor": "#1f77b4",
          "borderColor": "#1f77b4",
          "borderWidth": 1
        },
        {
          "label": "Validation ROC AUC",
          "data": [0, 0, 0, 0, 0],  // Replace with val_scores
          "backgroundColor": "#ff7f0e",
          "borderColor": "#ff7f0e",
          "borderWidth": 1
        }
      ]
    },
    "options": {
      "scales": {
        "y": {
          "beginAtZero": true,
          "title": {
            "display": true,
            "text": "ROC AUC Score"
          }
        },
        "x": {
          "title": {
            "display": true,
            "text": "Fold"
          }
        }
      },
      "plugins": {
        "legend": {
          "display": true
        },
        "title": {
          "display": true,
          "text": "LightGBM Cross-Validation ROC AUC"
        }
      }
    }
  }
  ```

## Future Improvements
- Check `train_scores` vs. `val_scores` for overfitting. If train ROC AUC is >0.05 higher, add regularization:
  ```python
  best_parms['reg_alpha'] = 12.0
  best_parms['reg_lambda'] = 5.0
  best_parms['min_split_gain'] = 0.01
  ```
- Try an ensemble with XGBoost to push past 0.96308:
  ```python
  from xgboost import XGBClassifier
  xgb_model = XGBClassifier(n_estimators=300, scale_pos_weight=7.29, random_state=42)
  xgb_model.fit(X, y)
  xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
  ensemble_probs = 0.8 * y_probs + 0.2 * xgb_probs
  ensemble_submission = pd.DataFrame({'id': test['id'], 'y': ensemble_probs})
  ensemble_submission.to_csv('ensemble_submission.csv', index=False)
  ```
- Check feature importance to add more features:
  ```python
  import matplotlib.pyplot as plt
  lgb.plot_importance(models[0], max_num_features=20)
  plt.title('LightGBM Feature Importance (Fold 1)')
  plt.show()
  ```

## Acknowledgements
Thanks to the Kaggle community for sharing ideas in notebooks and discussions, which helped me learn and improve my score![](https://towardsdatascience.com/how-i-ranked-in-the-top-25-on-my-first-kaggle-competition-9ea53499d58d/)