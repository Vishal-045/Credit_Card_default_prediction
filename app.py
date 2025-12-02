import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import io

st.set_page_config(page_title="Credit Card Default Dashboard", layout="wide")
st.title("💳 Credit Card Default Prediction Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("default_credit_card.csv", skiprows=1)
    df.drop(columns=["ID"], inplace=True)
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    return df

df = load_data()
st.markdown("### Dataset Preview")
st.dataframe(df.head())

# Feature/Target Split
X = df.drop(columns=["default"])
y = df["default"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Setup
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'SVM (Linear)': SVC(kernel='linear')
}

# Results Storage
results = []

st.markdown("### 🧠 Model Evaluation")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bias = model.score(X_train, y_train)
    variance = model.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Bias": bias,
        "Variance": variance,
        "Confusion Matrix": cm,
        "Classification Report": cr
    })

# Display bar plots for Accuracy, Bias, Variance
st.markdown("### 📊 Model Comparison")
metrics_df = pd.DataFrame([
    {
        "Model": r["Model"],
        "Accuracy": r["Accuracy"],
        "Bias": r["Bias"],
        "Variance": r["Variance"]
    } for r in results
])

cols = st.columns(3)
for i, metric in enumerate(["Accuracy", "Bias", "Variance"]):
    with cols[i]:
        st.markdown(f"#### {metric}")
        fig, ax = plt.subplots()
        sns.barplot(x="Model", y=metric, data=metrics_df, ax=ax, palette="viridis")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

# Confusion Matrices
st.markdown("### 🧮 Confusion Matrices")
for res in results:
    st.markdown(f"**{res['Model']}**")
    fig, ax = plt.subplots()
    sns.heatmap(res["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# Classification Reports (optional)
st.markdown("### 📑 Classification Reports")
for res in results:
    st.markdown(f"**{res['Model']}**")
    cr_df = pd.DataFrame(res["Classification Report"]).transpose()
    st.dataframe(cr_df.style.format("{:.2f}"))

# Downloadable CSV
st.markdown("### 📥 Download Evaluation Summary")
csv = metrics_df.to_csv(index=False).encode()
st.download_button("Download CSV Report", csv, "model_evaluation_summary.csv", "text/csv")
