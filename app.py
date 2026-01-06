import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", layout="wide")

# ----------------------------
# Title
# ----------------------------
st.title("🧠 Customer Segmentation using K-Means")
st.write("This app clusters customers based on demographic and spending behavior.")

# ----------------------------
# Upload Dataset
# ----------------------------
uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ----------------------------
    # Data Cleaning
    # ----------------------------
    st.subheader("🧹 Data Cleaning")

    num_cols = ['Age', 'Work_Experience', 'Family_Size']
    cat_cols = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Var_1']

    for col in num_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    if 'ID' in df.columns:
        df.drop('ID', axis=1, inplace=True)

    st.success("Missing values handled and ID column removed")

    # ----------------------------
    # Encoding
    # ----------------------------
    df_encoded = pd.get_dummies(df, drop_first=True)

    # ----------------------------
    # Scaling
    # ----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    # ----------------------------
    # Choose K
    # ----------------------------
    st.subheader("🔢 Choose Number of Clusters")
    k = st.slider("Select K", min_value=2, max_value=10, value=5)

    # ----------------------------
    # Train Model
    # ----------------------------
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df['Cluster'] = clusters

    # ----------------------------
    # Evaluation
    # ----------------------------
    score = silhouette_score(X_scaled, clusters)
    st.metric("Silhouette Score", round(score, 3))

    # ----------------------------
    # Visualization
    # ----------------------------
    st.subheader("📈 Cluster Visualization")

    fig, ax = plt.subplots()
    sns.scatterplot(
        x='Age',
        y='Spending_Score',
        hue='Cluster',
        palette='Set2',
        data=df,
        ax=ax
    )
    ax.set_title("Customer Clusters")
    st.pyplot(fig)

    # ----------------------------
    # Cluster Summary
    # ----------------------------
    st.subheader("📌 Cluster Summary")
    summary = df.groupby('Cluster')[[
        'Age',
        'Work_Experience',
        'Family_Size',
        'Spending_Score'
    ]].mean()

    st.dataframe(summary)

else:
    st.info("👆 Upload a CSV file to get started")
