
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# =====================
# Konfigurasi Halaman
# =====================
st.set_page_config(
    page_title="Steam Dashboard",
    page_icon="🎮",
    layout="wide"
)

# =====================
# Load Data
# =====================
df = pd.read_csv("clean_games.csv")
model = joblib.load("steam_regression_model.pkl")

# =====================
# Header
# =====================
st.title("🎮 Steam Game Analytics Dashboard")
st.markdown("### Final Project Big Data & Predictive Analytics")

# =====================
# Sidebar
# =====================
st.sidebar.header("Filter Data")

developer = st.sidebar.multiselect(
    "Developer",
    options=sorted(df["developer"].unique()),
    default=sorted(df["developer"].unique())
)

df = df[df["developer"].isin(developer)]

# =====================
# Metrics
# =====================
c1,c2,c3,c4 = st.columns(4)

c1.metric("Jumlah Game", len(df))
c2.metric("Rata-rata CCU", int(df["ccu"].mean()))
c3.metric("Rata-rata Owners", int(df["owners"].mean()))
c4.metric("Harga Rata-rata", int(df["price"].mean()))

st.divider()

# =====================
# Histogram
# =====================
left,right = st.columns(2)

with left:
    fig = px.histogram(
        df,
        x="ccu",
        title="Distribusi Peak CCU"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================
# Scatter
# =====================
with right:
    fig = px.scatter(
        df,
        x="owners",
        y="ccu",
        color="price",
        hover_name="name",
        title="Owners vs CCU"
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# =====================
# Heatmap
# =====================
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

st.divider()

# =====================
# Top 10 Game
# =====================
top = df.sort_values(
    "ccu",
    ascending=False
).head(10)

fig = px.bar(
    top,
    x="name",
    y="ccu",
    title="Top 10 Game Berdasarkan CCU"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# =====================
# Data Table
# =====================
st.subheader("Dataset")

st.dataframe(df)

st.divider()

# =====================
# Prediksi
# =====================
st.header("Prediksi Peak Concurrent Users")

owners = st.number_input("Owners", value=500000)
positive = st.number_input("Positive Review", value=10000)
negative = st.number_input("Negative Review", value=500)
average_forever = st.number_input("Average Forever", value=300)
average_2weeks = st.number_input("Average 2 Weeks", value=20)
price = st.number_input("Price", value=1999)
discount = st.number_input("Discount", value=10)

if st.button("Prediksi"):

    pred = model.predict([[
        owners,
        positive,
        negative,
        average_forever,
        average_2weeks,
        price,
        discount
    ]])

    st.success(f"Prediksi CCU : {pred[0]:,.0f}")
