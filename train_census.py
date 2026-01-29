# ============================================================
# PROJET S8 - CLASSIFICATION DU REVENU (CENSUS)
# Interface Streamlit avec design technologique
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# CONFIGURATION PAGE
# ============================================================

st.set_page_config(
    page_title="Census Income Predictor",
    layout="wide"
)

# ============================================================
# CSS – DESIGN TECHNOLOGIQUE
# ============================================================

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}

.navbar {
    background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 30px;
}

.navbar h1 {
    color: #00eaff;
    text-align: center;
    font-weight: 700;
}

.card {
    background-color: #161b22;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 0 15px rgba(0, 234, 255, 0.15);
}

.stButton>button {
    background: linear-gradient(90deg, #00eaff, #0072ff);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    height: 50px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# NAVBAR
# ============================================================

st.markdown("""
<div class="navbar">
    <h1>🔮 Census Income Prediction System</h1>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CHARGEMENT OU ENTRAINEMENT DU MODELE
# ============================================================

MODEL_PATH = "recensement.pkl"

@st.cache_resource
def train_or_load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    df = pd.read_csv("census.csv")
    df["Income"] = df["Income"].map({">50K": 0, "<=50K": 1})

    X = df.drop("Income", axis=1)
    y = df["Income"]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model

model = train_or_load_model()

# ============================================================
# INTERFACE DE PREDICTION
# ============================================================

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧠 Entrer les informations du citoyen")

col1, col2, col3 = st.columns(3)

age = col1.number_input("Age", 18, 100, 30)
education = col2.selectbox("Education", [
    "Bachelors", "HS-grad", "Masters", "Some-college", "Doctorate"
])
hours = col3.number_input("Heures par semaine", 1, 100, 40)

workclass = col1.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Public"
])

occupation = col2.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Sales", "Exec-managerial"
])

marital = col3.selectbox("Statut marital", [
    "Never-married", "Married", "Divorced"
])

# ============================================================
# PREDICTION
# ============================================================

if st.button("📊 Lancer la prédiction"):
    input_df = pd.DataFrame([{
        "age": age,
        "education": education,
        "hours-per-week": hours,
        "workclass": workclass,
        "occupation": occupation,
        "marital-status": marital
    }])

    prediction = model.predict(input_df)[0]

    if prediction == 0:
        st.error("💰 Revenu estimé : > 50K$")
    else:
        st.success("💼 Revenu estimé : ≤ 50K$")

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<p style='text-align:center; color:gray; margin-top:30px'>
Projet S8 – Machine Learning | Bagging & Boosting
</p>
""", unsafe_allow_html=True)
