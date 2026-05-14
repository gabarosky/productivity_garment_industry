# Driving garment production efficiency through predictive analytics and smart insights.




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
- Model Interpretability: Analyzing Individual Predictions
- Industrial implications
- Prescriptive Analysis: Using SHAP for Decision Making
    

----------
## 📁 4. Project Structure

```
productivity_garmet_industry/
│
├── 📂 data/
│   └── garments_worker_productivity.csv  # original dataset
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
├──  app.py         # Streamlit application
│
├──  Dockerfile     # Containerization setup
├── .dockerignore   # Docker exclusion list
├── requirements.txt
└── README.md
```
----------
## ⚙️ 5. Stack

### Modeling & Analysis
- **Environment:** Jupyter Notebook · Python 3
-   **Data manipulation:** pandas, NumPy, SciPy
-   **Visualization:** Matplotlib, Seaborn
-   **Modeling:** scikit-learn, XGBClassifier, LGBMClassifier
    -   Algorithms: Logistic Regression, SVM, Random Forest, XGBClassifier, LGBMClassifier, Extratrees, KNN
    -   Pipelines: Pipeline, ColumnTransformer, StandardScaler, OneHotEncoder
    -   Tuning & validation: StratifiedKFold, RandomizedSearchCV
    -   Metrics: MAE, F1-Score macro average, Balanced Accuracy

### 🖥️ Web Application 
**Framework:** Streamlit
**Visualization:** SHAP + Matplotlib
**Logging:** Python `logging` library for prediction traceability.
**Deployment:** Streamlit Cloud / Docker

----------
## 📈 6. Results

Through a cross validation the metrics obtained were
| Model        | F1 Macro Average | Balanced Accuracy | MAE  |
|--------------|------------------|-------------------|-----------------------|
| LogReg       | 0.518154         | 0.516586          | 0.667851              |
| SVC          | 0.597866         | 0.591658          | 0.596195              |
| RandomForest | 0.710819         | 0.712064          | 0.396645              |
| XGBoost      | 0.702878         | 0.704577          | 0.410948              |
| LightGBM     | 0.700600         | 0.701626          | 0.413334              |
| ExtraTrees   | 0.708730         | 0.710022          | 0.396611              |
| KNN          | 0.568091         | 0.565888          | 0.611740              |

By using MAE, we ensure that the model is penalized based on the distance between the predicted and actual category, helping us avoid significant gaps in productivity forecasting. Therefore the selected model is Random Forest

After parameter tunning, in test set:
- MAE: 0.4917
- F1-score (macro average): 0.6935
- Balanced accuracy: 0.6985

----------

## 💰 7. Business Impact


The proposed productivity prediction has a lift of 1.42x, which means it is 42% more accurate than the baseline. This allows data-driven decision-makers to leverage insights and improve overall performance.
   

----------

## 🔎 8. Key Insights


1. **Targeted productivity is the primary global driver of the model.**  
   Despite the shortcomings of targeted productivity for predicting productivity, it is a good input for our model.

2. **Incentives show significant predictive weight, followed by team size and SMV.**  
   This information is very important because incentives and the number of workers are levers for decision-making.

3. **Simulation of the predicted production.**  
   Using the model, it is possible to improve the predicted production using incentives, number of workers in the team, overtime; under the control of the decision-maker.
   
### Recommendation

- Retain 'targeted productivity' as a core model input rather than a standalone metric.

- Utilize productivity simulations as a strategic decision-making tool to optimize production lines.

----------
## 🚀 9. Interactive App - How to Use

### 🎮 **Live Demo**
🔗 **Try the app here:** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]((https://performancetuner.streamlit.app/))

### **Features**
- **Interactive sliders** for numerical variables
- **Dropdown selectors** for categorical variables (department)
- **Real-time prediction** (with button)
- **SHAP Waterfall plots** for all 4 productivity classes
- Visual explanation of how each feature contributes to the prediction

## 🛠️ 10. How to Run Locally

### Option A: direct Python
```bash
# Clone the repository
git clone https://github.com/gabarosky/productivity_garmet_industry.git
cd productivity_garmet_industry

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
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
docker run -p 8501:8501 productivity-shap-app

# Open your browser and go to:
# http://localhost:8501

```

**Recommended:** Python 3.11+` 


----------

Made by [Gabriel Carrizo](https://www.linkedin.com/in/carrizogabriel/) · MIT License
