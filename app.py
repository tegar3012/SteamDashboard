import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

# =====================
# Konfigurasi Halaman
# =====================
st.set_page_config(
    page_title="Steam Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "clean_games.csv"
MODEL_PATH = Path(__file__).parent / "steam_regression_model.pkl"

MODEL_FEATURES = [
    "owners",
    "positive",
    "negative",
    "average_forever",
    "average_2weeks",
    "price",
    "discount",
]


# =====================
# Load Data & Model (cached)
# =====================
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"developer", "ccu", "owners", "price", "name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Kolom berikut tidak ditemukan di CSV: {missing}")
    return df


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


try:
    df_raw = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"File data tidak ditemukan: `{DATA_PATH.name}`. Pastikan file ini ada di folder yang sama dengan app.py.")
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

try:
    model = load_model(MODEL_PATH)
    model_ok = True
except FileNotFoundError:
    model = None
    model_ok = False
except Exception as e:
    model = None
    model_ok = False
    st.sidebar.warning(f"Model gagal dimuat: {e}")

# =====================
# Header
# =====================
st.title("🎮 Steam Game Analytics Dashboard")
st.markdown("##### Final Project — Big Data & Predictive Analytics")
st.divider()

# =====================
# Sidebar — Filter
# =====================
st.sidebar.header("🔎 Filter Data")

all_devs = sorted(df_raw["developer"].unique())

col_a, col_b = st.sidebar.columns(2)
select_all = col_a.button("Pilih Semua", use_container_width=True)
clear_all = col_b.button("Bersihkan", use_container_width=True)

if "selected_devs" not in st.session_state:
    st.session_state.selected_devs = all_devs
if select_all:
    st.session_state.selected_devs = all_devs
if clear_all:
    st.session_state.selected_devs = []

developer = st.sidebar.multiselect(
    "Developer",
    options=all_devs,
    default=st.session_state.selected_devs,
    key="selected_devs",
)

price_min, price_max = int(df_raw["price"].min()), int(df_raw["price"].max())
price_range = st.sidebar.slider(
    "Rentang Harga (¢, dalam sen USD)",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, price_max),
    help="Data harga tersimpan dalam sen (mis. 1999 = $19.99).",
)

owners_min, owners_max = float(df_raw["owners"].min()), float(df_raw["owners"].max())
owners_range = st.sidebar.slider(
    "Rentang Owners",
    min_value=owners_min,
    max_value=owners_max,
    value=(owners_min, owners_max),
    format="%.0f",
)

if len(developer) == 0:
    st.warning("Pilih minimal satu developer di sidebar untuk menampilkan data.")
    st.stop()

df = df_raw[
    df_raw["developer"].isin(developer)
    & df_raw["price"].between(*price_range)
    & df_raw["owners"].between(*owners_range)
].copy()

if df.empty:
    st.warning("Tidak ada data yang cocok dengan filter yang dipilih. Coba longgarkan filternya.")
    st.stop()

# =====================
# Tabs
# =====================
tab_overview, tab_viz, tab_data, tab_predict = st.tabs(
    ["📊 Overview", "📈 Visualisasi", "🗂️ Data", "🔮 Prediksi"]
)

# ---------------------
# TAB: Overview
# ---------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Game", f"{len(df):,}")
    c2.metric("Rata-rata CCU", f"{df['ccu'].mean():,.0f}")
    c3.metric("Rata-rata Owners", f"{df['owners'].mean():,.0f}")
    c4.metric("Harga Rata-rata", f"${df['price'].mean() / 100:,.2f}")

    st.divider()

    top = df.sort_values("ccu", ascending=False).head(10)
    fig_top = px.bar(
        top,
        x="ccu",
        y="name",
        orientation="h",
        title="Top 10 Game Berdasarkan Peak CCU",
        labels={"ccu": "Peak CCU", "name": "Game"},
        text_auto=",.0f",
    )
    fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_top, use_container_width=True)

    if not top.empty:
        st.caption(
            f"📌 **{top.iloc[0]['name']}** memimpin dengan peak CCU sebesar "
            f"**{top.iloc[0]['ccu']:,.0f}** pemain dalam data yang difilter."
        )

# ---------------------
# TAB: Visualisasi
# ---------------------
with tab_viz:
    left, right = st.columns(2)

    with left:
        fig_hist = px.histogram(
            df, x="ccu", title="Distribusi Peak CCU", labels={"ccu": "Peak CCU"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with right:
        fig_scatter = px.scatter(
            df,
            x="owners",
            y="ccu",
            color="price",
            hover_name="name",
            title="Owners vs CCU",
            labels={"owners": "Owners", "ccu": "Peak CCU", "price": "Harga (¢)"},
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    numeric_df = df.select_dtypes(include="number").drop(columns=["appid"], errors="ignore")
    corr = numeric_df.corr()
    fig_heat = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Korelasi Antar Variabel Numerik",
    )
    fig_heat.update_layout(height=600)
    st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------
# TAB: Data
# ---------------------
with tab_data:
    st.subheader("Dataset (Setelah Filter)")
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download data terfilter (CSV)",
        data=csv_bytes,
        file_name="filtered_games.csv",
        mime="text/csv",
    )

# ---------------------
# TAB: Prediksi
# ---------------------
with tab_predict:
    st.header("Prediksi Peak Concurrent Users (CCU)")

    if not model_ok:
        st.error(
            f"Model tidak tersedia (`{MODEL_PATH.name}` tidak ditemukan atau gagal dimuat). "
            "Fitur prediksi dinonaktifkan."
        )
    else:
        st.caption(
            "Model regresi linier sederhana — hasil prediksi bersifat estimasi kasar "
            "berdasarkan pola historis, bukan jaminan akurat."
        )

        with st.form("prediction_form"):
            fc1, fc2 = st.columns(2)
            with fc1:
                owners = st.number_input("Owners", min_value=0, value=500_000, step=10_000)
                positive = st.number_input("Positive Review", min_value=0, value=10_000, step=100)
                negative = st.number_input("Negative Review", min_value=0, value=500, step=50)
                average_forever = st.number_input(
                    "Average Playtime Forever (menit)", min_value=0, value=300, step=10
                )
            with fc2:
                average_2weeks = st.number_input(
                    "Average Playtime 2 Weeks (menit)", min_value=0, value=20, step=5
                )
                price = st.number_input("Harga (¢, sen USD)", min_value=0, value=1999, step=100)
                discount = st.number_input("Discount (%)", min_value=0, max_value=100, value=10, step=5)

            submitted = st.form_submit_button("Prediksi", use_container_width=True)

        if submitted:
            input_df = pd.DataFrame(
                [[owners, positive, negative, average_forever, average_2weeks, price, discount]],
                columns=MODEL_FEATURES,
            )
            try:
                pred = model.predict(input_df)[0]
                pred_display = max(pred, 0)  # CCU tidak mungkin negatif
                st.success(f"Prediksi Peak CCU: **{pred_display:,.0f}** pemain")
                if pred < 0:
                    st.caption(
                        "⚠️ Model memprediksi nilai negatif secara mentah (dibulatkan ke 0) — "
                        "ini wajar terjadi pada regresi linier untuk kombinasi input yang jauh dari data training."
                    )
            except Exception as e:
                st.error(f"Gagal melakukan prediksi: {e}")
