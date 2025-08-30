# 🏆 Kaggle Tabular Playground Series – Season 5, Episode 8  

**Competition:** [Playground Series — Season 5, Episode 8: Binary Classification with a Bank Dataset](https://www.kaggle.com/competitions/playground-series-s5e8)  

🎯 **Goal:** Predict whether a customer will subscribe to a marketing campaign (`y=1`) using a bank dataset.  
📈 **Metric:** ROC AUC (achieved **0.96308** on leaderboard).  
⚠️ **Challenge:** Imbalanced dataset (~12% positive class).  

---

## 📂 Repository Structure  

- 📒 `notebook.ipynb` → Main pipeline (EDA, feature engineering, modeling, tuning, submission).  
- 📄 `submission.csv` → Final predictions (`id`, `y`) for Kaggle.  
- 📝 `README.md` → Overview of project (this file).  

---

## 🔍 Workflow Summary  

### 1️⃣ Exploratory Data Analysis (EDA)  
- 📊 Used `describe()` and `info()` → found **imbalance (~12% subscribed)** + outliers (e.g., balance, duration).  
- 📉 Histograms + boxplots → showed education, job, loan, and marital status patterns.  
- 🔥 Correlation heatmap → `duration` had **0.52 correlation with y** → key feature.  

### 2️⃣ Feature Engineering & Preprocessing  
- 🛠️ Built a **Preprocessing class** with:  
  - Duration transforms → `duration_log`, `duration_sqrt`, `duration_per_campaign`.  
  - Cyclical encoding → `month_sin`, `month_cos`, `day_sin`, `day_cos`.  
  - Binning → age, balance, duration, and pdays into meaningful ranges.  
  - Binary flags → `duration_favorable`, `pdays_favorable`.  
  - Combined features → `poutcome_duration_pdays`.  
- ⚖️ Scaled continuous data with **RobustScaler** (handles outliers better).  
- 🔑 Encoding:  
  - LightGBM → LabelEncoder  
  - Others → mix of Label + OneHot depending on feature type.  

### 3️⃣ Model Selection  
- 🧪 Tested Logistic Regression, Random Forest, XGBoost, LightGBM.  
- ✅ **LightGBM** won → handled imbalance + captured complex feature interactions.  
- ⚖️ Used stratified CV + `class_weight="balanced"` / `scale_pos_weight`.  

### 4️⃣ Hyperparameter Tuning  
- 🤖 Bayesian Optimization (BayesSearchCV) with 5-fold CV.  
- Best params:  
  - `learning_rate = 0.01078`  
  - `num_leaves = 379`  
  - `max_depth = 15`  

### 5️⃣ Final Training & Submission  
- 🔁 Trained with 5-fold cross-validation.  
- 🛑 Early stopping (500 rounds).  
- 📊 Averaged fold predictions → stable + robust.  
- 🏅 Public LB ROC AUC → **0.96308**.  

---

## 🌟 Key Takeaways  

- 🚀 Feature engineering (esp. duration & pdays ranges) boosted performance.  
- ⚖️ Handling imbalance correctly was crucial.  
- 🧠 Bayesian tuning saved time vs grid search.  
- 🔄 Averaging folds improved generalization.  

---

## 🔮 Future Improvements  

- 🤝 Try model stacking/blending (LightGBM + XGBoost + CatBoost).  
- 📌 Add SHAP for explainability.  
- 🧑‍💻 Explore deep tabular models (SAINT, FastAI).  

---

## 📚 References  

- 🔗 [Competition Page](https://www.kaggle.com/competitions/playground-series-s5e8)  
- 📑 My Kaggle Notebooks (https://www.kaggle.com/code/astik13aw/notebook16349fe563) 

---

✨ *Thanks for reading! If you find this repo helpful, don’t forget to ⭐ star it!* ✨ 

