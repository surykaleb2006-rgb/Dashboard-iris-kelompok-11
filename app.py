import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Dashboard Iris - Kelompok 11", page_icon="🌸", layout="wide")
st.title("🌸 Dashboard Dataset Bunga Iris")
st.caption("Aplikasi web Streamlit | Kelompok 11")

df = pd.read_csv("Iris_Dataset.csv")
st.write("Dashboard ini menampilkan eksplorasi dataset bunga Iris dan contoh prediksi jenis bunga menggunakan SVM.")

st.sidebar.header("Navigasi")
page = st.sidebar.radio("Pilih halaman", ["Beranda & Data", "Visualisasi", "Prediksi Iris"])

if page == "Beranda & Data":
    st.subheader("Ringkasan Dataset")
    c1, c2 = st.columns(2)
    c1.metric("Jumlah baris", len(df))
    c2.metric("Jumlah kelas", df["class"].nunique())
    st.subheader("Data bunga Iris")
    st.dataframe(df, use_container_width=True)
    st.subheader("Ringkasan statistik")
    st.dataframe(df.describe(), use_container_width=True)

elif page == "Visualisasi":
    st.subheader("Distribusi kelas")
    counts = df["class"].value_counts()
    st.bar_chart(counts)
    st.subheader("Hubungan ukuran sepal dan petal")
    x_col = st.selectbox("Variabel sumbu X", df.select_dtypes("number").columns, index=0)
    y_col = st.selectbox("Variabel sumbu Y", df.select_dtypes("number").columns, index=2)
    fig, ax = plt.subplots()
    for cls in df["class"].unique():
        part = df[df["class"] == cls]
        ax.scatter(part[x_col], part[y_col], label=cls)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    st.pyplot(fig)

elif page == "Prediksi Iris":
    st.subheader("Prediksi jenis bunga dengan SVM")
    features = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
    X = df[features]
    y = df["class"]
    model = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    st.caption(f"Contoh akurasi pada data uji: {accuracy_score(y_test, model.predict(X_test)):.2%}")
    st.write("Masukkan ukuran bunga dalam sentimeter:")
    col1, col2 = st.columns(2)
    with col1:
        sl = st.number_input("Sepal length (cm)", min_value=0.0, value=5.1, step=0.1)
        sw = st.number_input("Sepal width (cm)", min_value=0.0, value=3.5, step=0.1)
    with col2:
        pl = st.number_input("Petal length (cm)", min_value=0.0, value=1.4, step=0.1)
        pw = st.number_input("Petal width (cm)", min_value=0.0, value=0.2, step=0.1)
    if st.button("Prediksi jenis Iris"):
        result = model.predict(pd.DataFrame([[sl, sw, pl, pw]], columns=features))[0]
        st.success(f"Prediksi jenis bunga: {result.title()}")
