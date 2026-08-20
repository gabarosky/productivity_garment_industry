# scripts/train_model.py
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize
import shap
import sys
import os

# Add scripts to path
sys.path.append(os.path.dirname(__file__))
from data_preprocessing import load_and_clean_data, create_target, get_feature_lists, prepare_X_y


RANDOM_STATE = 43

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data','raw', 'garments_worker_productivity.csv')    
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')    
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    # Load and prepare data
    print("Loading dataset...")
    df = load_and_clean_data(DATA_PATH)
    df = create_target(df)
    numeric_cols, categorical_cols = get_feature_lists()
    X, y = prepare_X_y(df, numeric_cols, categorical_cols)

    # Set preprocessor and pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('scaler', StandardScaler())]), numeric_cols),
        ('cat', OneHotEncoder( handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', RandomForestRegressor(n_estimators=300,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        max_features=1.0,
                                        max_depth=20,
                                        bootstrap=True,
                                        random_state=RANDOM_STATE
                                        ))
        ])

    
    pipeline.fit(X, y)
    
    #  Save the pipeline
    print("Saving pipeline...")
    joblib.dump(pipeline, os.path.join(MODEL_DIR, 'modelo_pipeline.pkl'))
    
    print("Optimizing thresholds...")
    # optimize the thresholds for the optimized model
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
    y_pred_cont = cross_val_predict(pipeline, X, y, cv=skf, n_jobs=-1)
    best_thresholds = optimize_thresholds(y_pred_cont, y)

    
    #  Save thresholds for inference in Streamlit
    joblib.dump(best_thresholds, os.path.join(MODEL_DIR, 'thresholds.pkl'))
    
    # Create and save SHAP explainer 
    print("Creating SHAP explainer...")
    
    # extract model and preprocessor
    final_model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps["preprocess"]


    # create explainer with train data
    explainer = shap.TreeExplainer(final_model)
    
    print("Saving SHAP explainer...")
    joblib.dump(explainer, os.path.join(MODEL_DIR, 'explainer.pkl'))

    # Get and save feature metadata
    print("Generating metadata...")
 
    # Ranges for numeric features (using original X, before transformation)
    numeric_ranges = {}
    for col in numeric_cols:
        numeric_ranges[col] = {
            'min': float(X[col].min()),
            'max': float(X[col].max())
        }

    # Categories for categorical variables
    categorical_categories = {}
    for col in categorical_cols:
        categorical_categories[col] = X[col].unique().tolist()

    
    print("Process completed! Artifacts saved in", MODEL_DIR)

def optimize_thresholds(y_val_cont: np.ndarray, y_val_true: np.ndarray) -> np.ndarray:
    """ Fit the optimal thresholds for F1-Score Macro."""
    def loss_function(thresholds):
        preds = get_discrete_classes(y_val_cont, thresholds)
        return -cohen_kappa_score(y_val_true, preds, weights='quadratic')
    
    init_thresholds = [1.5, 2.5, 3.5]
    res = minimize(loss_function, init_thresholds, method='Nelder-Mead')
    return np.sort(res.x)

def get_discrete_classes(y_cont: np.ndarray, thresholds: list) -> np.ndarray:
    """ Convert continuous predictions to classes (1, 2, 3, 4) according thresholds."""
    sorted_thresholds = np.sort(thresholds)
    return np.digitize(y_cont, sorted_thresholds) + 1

if __name__ == "__main__":
    main()