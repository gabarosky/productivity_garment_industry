# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:34:36 2026

@author: gabri
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import logging
import os
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO



# ------------------------------------------------------------
# 1. Set up logging
# ------------------------------------------------------------
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log")
    ]
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# 2. Load artifacts
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load('model/modelo_pipeline.pkl')
    explainer = joblib.load('model/explainer.pkl')
    thresholds = joblib.load('model/thresholds.pkl')
    # with open('model/feature_metadata.json', 'r') as f:
    #     metadata = json.load(f)
    # return pipeline, explainer, metadata
    return pipeline, explainer,thresholds


pipeline, explainer,thresholds = load_artifacts()

# ------------------------------------------------------------
# 3. discrtizing function
# ------------------------------------------------------------
def get_discrete_classes(y_cont: np.ndarray, thresholds: list) -> np.ndarray:
    """ Convert continuous predictions to classes (1, 2, 3, 4) according thresholds."""
    sorted_thresholds = np.sort(thresholds)
    return np.digitize(y_cont, sorted_thresholds) + 1 
# ------------------------------------------------------------
# 4. function for waterfalls
# ------------------------------------------------------------
def plot_waterfall(sample, preprocessor, model, explainer, thresholds):
    ''' This function take a sample of data, calculate Shapley values and plot a waterfall for 
    category in order to show the impact of each feature in the final prediction. Furthermore,
    threshold boundaries are depicted
    Variables:
    sample: pd.DataFrame
        A single-row DataFrame with the sample of variables to predict and analize.
    model: Estimator
        The model used for predictions
    preprocessor: ColumnTransformer / Pipeline
        The preprocessor used for transform the data.
    explainer: shap.Explainer
        The Shapley explainer trained with data and model.
    thresholds: list or array-like
        A list of thresholds among categories.
    sample_name: str, optional
        A string for the title.'''
    
    # transform the sample
    sample_transformed = preprocessor.transform(sample)
    feature_names = preprocessor.get_feature_names_out()
    sample_transformed_df = pd.DataFrame(sample_transformed, columns=feature_names)

    # make the continuos prediction and discretize
    pred_cont = model.predict(sample_transformed_df.values)[0]
    pred_class = get_discrete_classes(np.array([pred_cont]), thresholds)[0]
    
    # obtain SHAP Values
    shap_values = explainer(sample_transformed_df)
    
    # prepare for plot
    display_data = []
    display_names = []

    for feat_name in feature_names:
        base_name = feat_name.split("__")[-1]
        
        if base_name in sample.columns:
            value = sample[base_name].iloc[0]
            display_data.append(value)
            display_names.append(base_name.replace("_", " "))
        else:
            parts = base_name.split('_', 1)
            display_data.append(parts[1] if len(parts) > 1 else parts[0])
            display_names.append(parts[0].replace("_", " "))

    exp = shap.Explanation(
        values=shap_values.values[0], 
        base_values=shap_values.base_values[0],
        data=np.array(display_data), 
        feature_names=display_names
    )

    # plot axes
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # waterfall plot
    shap.plots.waterfall(exp, max_display=12, show=False)
    
    # set axes limit
    ax.set_xlim(0, 4)
    
    # Set thresholds
    for t in thresholds:
        ax.axvline(x=t, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, zorder=1)

    class_bounds = [0] + list(thresholds) + [4]
    
    y_top = ax.get_ylim()[1]
    
    # class labels
    for i in range(4):
        mid_point = (class_bounds[i] + class_bounds[i+1]) / 2
        
        ax.text(
            mid_point, y_top * 0.08, f"C{i+1}", 
            color='white', fontweight='bold', fontsize=10, 
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333', alpha=0.85, edgecolor='none')
        )

    plt.title(
        f"SHAP Waterfall for user inputs\nContinuous prediction: {pred_cont:.3f} ➔ Predicted Class: C{pred_class}", 
        fontsize=13, fontweight='bold', pad=25
    )
    
    plt.tight_layout()
    plt.show()
    return fig
    
    
# ------------------------------------------------------------
# 5.  Layout setup: sidebar and two columns
# ------------------------------------------------------------
st.set_page_config(page_title="Prediction with SHAP", layout="wide")
st.title("🧵 Garment productivity prediction and interactive setting")
# Expander with instructions
with st.expander("📘 Instructions", expanded=False):
    st.markdown("""
    1. Set the variable values in the left sidebar.
    2. Click the **Predict / update** button to generate a productivity prediction and a Waterfall chart explanation.
    3. The plot displays the contribution of each variable to each class.
    4. You can adjust the values and click **Predict / update** again to obtain an updated prediction.
    """)


# Create the input form at sidebar
st.sidebar.header("📊 Input settings")
st.markdown("---")

# Dict for the values
input_data = {}
   
# Slider for numericals
input_data['no_of_workers'] = st.sidebar.slider(
    "**No of Workers**",
    min_value=2,
    max_value=90,
    value=34,
    step=1,
    format="%.2f"
)
input_data['over_time'] = st.sidebar.slider(
    "**Over-time**",
    min_value=0,
    max_value=25920,
    value=3960,
    step=1,
    format="%.2f"
)
input_data['incentive'] = st.sidebar.slider(
    "**Incentive**",
    min_value=0,
    max_value=3600,
    value=0,
    step=1,
    format="%.2f"
)
input_data['smv'] = st.sidebar.slider(
    "**SMV**",
    min_value=2.9,
    max_value=55.0,
    value=15.26,
    step=0.01,
    format="%.2f"
)
input_data['wip'] = st.sidebar.slider(
    "**Work in Progress**",
    min_value=0,
    max_value=24000,
    value=1039,
    step=1,
    format="%.2f"
)
input_data['idle_time'] = st.sidebar.slider(
    "**Idle Time**",
    min_value=0,
    max_value=300,
    value=0,
    step=1,
    format="%.2f"
)
input_data['idle_men'] = st.sidebar.slider(
    "**Idle Men**",
    min_value=0,
    max_value=45,
    value=0,
    step=1,
    format="%.2f"
)
# Select box for categorical 
input_data['no_of_style_change'] = st.sidebar.selectbox(
    "**No. of style changes**",
    options=[0,1,2],
    index=0
)
input_data['day'] = st.sidebar.selectbox(
    "**Day**",
    options=['Monday','Tuesday','Wednesday','Thursday','Saturday','Sunday'],
    index=0
)
input_data['quarter'] = st.sidebar.selectbox(
    "**Quarter**",
    options=[1,2,3,4,5],
    index=0
)
input_data['department'] = st.sidebar.selectbox(
    "Department",
    options=['finishing', 'sewing'],
    index=0
)
input_data['team'] = st.sidebar.selectbox(
    "Team",
    options=[1,2,3,4,5,6,7,8,9,10,11,12],
    index=0
)

left_col, right_col = st.columns([1, 3])

with left_col:
    st.header("Predicted class")
    # Placeholder for prediction
    prediction_placeholder = st.empty()
    # Provisory text
    prediction_placeholder.markdown("## _ _ _")
    
    # Button for activate prediction
    predict = st.button("Predict / update", type="primary")
        
with right_col:
    st.header("Waterfall explanatory for classes")
    # Placeholder para la imagen
    waterfall_placeholder = st.empty()
    # Mensaje inicial
    waterfall_placeholder.info("Click 'Predict / update' for generate plots")

# ------------------------------------------------------------
# 6. Prediction and plot generation
# ------------------------------------------------------------
if predict:
    with st.spinner("Calculating predictions and SHAP..."):
        # Create DataFrame with the input data
        df_input = pd.DataFrame([input_data])
     
        # extract the preprocessor from pipeline
        preprocessor = pipeline.named_steps['preprocess']
        model = pipeline.named_steps['model']
        
        categorical_cols = ['department', 'day', 'team', 'quarter']
        df_input[categorical_cols] = df_input[categorical_cols].astype(str)
        # make prediction
        prediction_continuous = pipeline.predict(df_input)[0]
        # discrtize
        prediction = get_discrete_classes(prediction_continuous, thresholds)
        
        # log the prediction
        logger.info("="*50) 
        logger.info("New prediction started")
        logger.info("Data input:")
        logger.info(f"\n{df_input.to_string()}")
        logger.info(f"Prediction obtained: {prediction}")

        # change the matplotlib backend for avoiding emengent windows        
        matplotlib.use('Agg')
        
        # call the function for waterfalls
        plot_waterfall(
            sample=df_input,
            preprocessor=preprocessor,
            model=model,
            explainer=explainer,
            thresholds=thresholds
        )
        
        # capture the figure
        fig = plt.gcf()
        
        # convert the figure to bytes
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        imagen_bytes = buf.getvalue()
        plt.close(fig)  # free memory
        
        # Update the placeholders
        with left_col:
            prediction_placeholder.markdown(f"## {prediction}")
        
        with right_col:
            waterfall_placeholder.image(imagen_bytes, width='stretch')
        
        # Save to session_state for persistence (in case the page reloads)
        st.session_state.prediction = prediction
        st.session_state.waterfall_bytes = imagen_bytes

# ------------------------------------------------------------
# 7. If a prediction was previously made (via session_state), display the results upon page load.
# ------------------------------------------------------------
else:
    if 'prediction' in st.session_state and 'waterfall_bytes' in st.session_state:
        with left_col:
            prediction_placeholder.markdown(f"## {st.session_state.prediction}")
        with right_col:
            waterfall_placeholder.image(st.session_state.waterfall_bytes, width='stretch')
            
            
           