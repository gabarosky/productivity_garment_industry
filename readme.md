# Predictive Analytics & Smart Insights for Garment Production
### Optimizing daily manufacturing performance through machine learning and interactive simulation

v 2.0



## 🧠 1. Business Problem

The garment industry is a labor-intensive sector where productivity directly impacts operational costs and delivery timelines. The objective of this analysis is to predict and analyze the productivity performance of manufacturing teams. The study identifies key operational drivers and provides a tool for data-driven decision-making.

### 🎯 **Live Demo**
🔗 **Try the app here:** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://performancetuner.streamlit.app/)

----------

## 📊 2. Dataset

-   [Productivity Prediction of Garment Employees](https://doi.org/10.24432/C51S6D) from [UCI Machine Learning Repository](https://archive.ics.uci.edu/).
    
-   Contains 1197 instances and 14 features. Each instance correspond to a combination of team and departament (sewing or finishing) for a day between 01-Jan-2015 and 11-Mar-2015.
    
-   Target variable: actual_productivity

In this project, we replaced the 'actual_productivity' target with a categorical target based on 'actual_productivity' intervals.

---

## 🛠 3. Methodology

- Data overview and cleaning  
- EDA    
- Feature engineering    
- Split train/test set
- Model comparison
- Preprocessing with pipelines
- Hyperparameter Tuning with Random Search
- Cross-validation
- Model Interpretability: Global and local feature analysis
- Prescriptive simulation
- Benchmark comparison
- Industrial implications


----------
## 📁 4. Project Structure

```
productivity_garmet_industry/
│
├── 📂 data/
│   ├── 📂 raw/
│   │   └── garments_worker_productivity.csv  # original dataset
│   └── 📂 processed/
│       └── dataset.csv
│
├── 📂 logs/
│   ├── .gitkeep             
│   └── app.log              # Local execution logs 
│
├── 📂 notebooks/                 
│   └── initial.ipynb        # EDA, Model Selection, Training & Fitting, Evaluation, Lift Analysis & Business Analysis
│
├── 📂 model/
│   ├── eplainer.pkl         # SHAPE explainer trained
│   ├── label_encodeer.pkl   # label encoder
│   └── modelo_pipeline.pkl  # model trained
│
├── 📂 scripts/
│   ├── data_preprocessing.py     # load, clean and preprocess data
│   └── train_model.py            # train the model and store the model artifacts
││
├── app.py         # Streamlit application
├── docker-compose.yml
├── Dockerfile     # Containerization setup
├── .dockerignore   # Docker exclusion list
├── requirements.txt
└── README.md
```
----------
## ⚙️ 5. Stack

### Modeling & Analysis
- **Language & Environment:** Python 3.11+ · Jupyter Notebook
-   **Data manipulation:** pandas, NumPy, SciPy
-   **Visualization:** Matplotlib, Seaborn, SHAP
-  **Machine Learning:** scikit-learn, XGBoost, LightGBM
  * *Algorithms Evaluated:* Ridge, Random Forest, XGBoost, LightGBM, Extra Trees, K-Neighbors
  * *Pipelines & Validation:* `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `StratifiedKFold`, `RandomizedSearchCV`
  * *Evaluation Metrics:* Quadratic Weighted Kappa (QWK), MAE, Macro F1-Score, Weighted F1-Score, Precision/Recall

### 🖥️ Web Application 
* **Framework:** Streamlit
* **Visualization:** SHAP + Matplotlib
* **Logging:** Python `logging` library for prediction traceability.
* **Containerization & Deployment:** Docker · Docker Compose · Streamlit Cloud

----------
## 📈 6. Results
| Metric | Traditional Target (`targeted_productivity`) | Selected ML Model (Random Forest) | Improvement / Lift |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 46.25% | **62.50%** | **+35.14% Lift** |
| **MAE (Mean Absolute Error)** | 0.7208 | **0.4792** | **33.53% Error Reduction** |
| **Macro F1-Score** | 0.3630 | **0.6113** | **+68.39% Lift** |
| **Weighted F1-Score** | 0.3944 | **0.6233** | **+58.05% Lift** |

### Test Set Performance Metrics
* **QWK (Test):** `0.6485` (Train QWK: `0.7145`)
* **MAE (Test):** `0.4833` (Train MAE: `0.4305`)
* **Class-Level Precision:**
  * **Classes 1, 2, & 3:** Solid, balanced performance across baseline and intermediate tiers (F1-scores of `0.66`, `0.52`, and `0.69`).
  * **Class 4 (High Productivity):** Traditional human targets completely fail to identify top-performing shifts (**0% Precision / Recall / F1**). The ML model effectively captures high-productivity capacity with **71% Precision** (`0.58` F1-score).
  
By using MAE, we ensure that the model is penalized based on the distance between the predicted and actual category, helping us avoid significant gaps in productivity forecasting. Therefore the selected model is Random Forest

After parameter tunning, in test set:
- MAE: 0.4917
- F1-score (macro average): 0.6935
- Balanced accuracy: 0.6985

----------
## 🔎 7. Key Operational Insights (SHAP & Feature Importance)

* **Primary Operational Driver:** Financial **`incentive`** is the single most important feature dictating shift output.
* **Key Secondary Drivers:** **`smv`** (Standard Minute Value), **`no_of_workers`**, and **`over_time`** follow in order of predictive weight.
* **Non-Linear Interactions:** Feature impact varies significantly depending on shift context, demonstrating complex interactions between workforce size and overtime limits.
* **Rare Extremes in Error:** The distribution of errors shows that large misclassifications are extremely rare. Most prediction errors occur between adjacent classes (average deviation < 0.5 classes), with **zero absolute error in nearly 150 instances**.

---

## 💰 8. Industrial Implications & Strategic Recommendations

* **Correction of Systematic Target Overestimation:** Traditional goal-setting over-promises intermediate output while ignoring Class 4 high-productivity capacity. The model realigns plant expectations with true capacity, preventing downstream supply chain bottlenecks.
* **Reduction of Planning Errors:** Minimizing 1- and 2-class prediction errors prevents costly over-staffing or under-staffing on the plant floor.
* **Deploy "What-If" Simulations:** Plant managers can use the interactive web pipeline as a pre-shift decision-support tool. By tweaking actionable inputs (`incentive`, `no_of_workers`, `over_time`), management can simulate productivity outcomes **before committing budget**.

----------
## 🚀 9. Interactive App - How to Use

### 🎮 **Live Demo**
🔗 **Try the app here:** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]
(https://performancetuner.streamlit.app/)

### **Features**
- **Interactive sliders** for numerical variables
- **Dropdown selectors** for categorical variables.
- **Real-time prediction** (with button)
- **SHAP Waterfall plot** provides a magnitude and sense of the impact of each variable
- Visual explanation of how each feature contributes to the prediction

## 🛠️ 10. How to Run Locally

### Option A: direct Python
```bash
# Clone the repository
git clone https://github.com/gabarosky/productivity_garmet_industry.git
cd productivity_garmet_industry

# Install dependencies
pip install -r requirements.txt
```

 **Optional**: Update data & retrain model

If you need to re-process the raw data and train the model before launching:

``` bash
python scripts/data_preprocessing.py && python scripts/train_model.py
```
Run the app
```
streamlit run app.py
```

##  Option B: Run with Docker (Local Build) 🐳

```bash
# Clone the repository
git clone https://github.com/gabarosky/productivity_garmet_industry.git
cd productivity_garmet_industry

# Build the Docker image 
docker build -t productivity-shap-app .

# Run the container
docker-compose up --buid

# Open your browser and go to:
# http://localhost:8501

```
The application will be available at http://localhost:8501.
The execution logs will be automatically saved in the ./logs folder on your host machine.

**Recommended:** Python 3.11+


----------

Made by [Gabriel Carrizo](https://www.linkedin.com/in/carrizogabriel/) · MIT License
