import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# Konfigurasi Halaman
# =====================================================
st.set_page_config(
    page_title="Steam Game Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# Custom CSS — tampilan kartu, warna, dan tipografi
# =====================================================
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }

    /* Judul & subjudul */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #66c0f4, #1b2838);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle {
        color: #8f98a0;
        font-size: 1rem;
        margin-top: -0.3rem;
        margin-bottom: 1.2rem;
    }

    /* Kartu metrik custom */
    .metric-card {
        background: linear-gradient(145deg, #1b2838, #16202d);
        border: 1px solid #2a3f5f;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        text-align: left;
    }
    .metric-label {
        color: #8f98a0;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.7rem;
        font-weight: 700;
        margin-top: 0.15rem;
    }
    .metric-icon { font-size: 1.4rem; }

    section[data-testid="stSidebar"] {
        border-right: 1px solid #2a3f5f;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #2a3f5f;
        border-radius: 10px;
    }

    hr { border-color: #2a3f5f; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# Helper — format angka besar
# =====================================================
def format_big_number(n):
    n = float(n)
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:,.0f}"


def format_price(cents):
    """Kolom price/initialprice pada dataset Steam disimpan dalam sen (mis. 1999 = $19.99)."""
    return f"${cents/100:,.2f}"


def metric_card(label, value, icon="🎮"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================
# Load Data & Model (dengan cache dan error handling)
# =====================================================
REQUIRED_COLUMNS = [
    "appid", "name", "developer", "publisher", "positive", "negative",
    "userscore", "owners", "average_forever", "average_2weeks",
    "median_forever", "median_2weeks", "price", "initialprice",
    "discount", "ccu",
]

MODEL_FEATURES = [
    "owners", "positive", "negative", "average_forever",
    "average_2weeks", "price", "discount",
]


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Kolom berikut tidak ditemukan di dataset: {missing}")
    return data


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


try:
    df_raw = load_data("clean_games.csv")
except FileNotFoundError:
    st.error("❌ File **clean_games.csv** tidak ditemukan. Pastikan file berada satu folder dengan app.py.")
    st.stop()
except ValueError as e:
    st.error(f"❌ Dataset tidak valid: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Gagal memuat dataset: {e}")
    st.stop()

try:
    model = load_model("steam_regression_model.pkl")
except FileNotFoundError:
    st.error("❌ File **steam_regression_model.pkl** tidak ditemukan.")
    st.stop()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

if df_raw.empty:
    st.warning("⚠️ Dataset kosong — tidak ada data untuk ditampilkan.")
    st.stop()

# =====================================================
# Header
# =====================================================
st.markdown('<p class="hero-title">🎮 Steam Game Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Final Project — Big Data & Predictive Analytics</p>',
    unsafe_allow_html=True,
)

# =====================================================
# Sidebar — Filter
# =====================================================
st.sidebar.markdown("## 🔍 Filter Data")

all_devs = sorted(df_raw["developer"].dropna().unique())
select_all = st.sidebar.checkbox("Pilih semua developer", value=True)

developer = st.sidebar.multiselect(
    "Developer",
    options=all_devs,
    default=all_devs if select_all else [],
)

price_min, price_max = int(df_raw["price"].min()), int(df_raw["price"].max())
price_range = st.sidebar.slider(
    "Rentang Harga (USD)",
    min_value=price_min / 100,
    max_value=price_max / 100,
    value=(price_min / 100, price_max / 100),
    step=0.5,
    format="$%.2f",
)

search_name = st.sidebar.text_input("Cari nama game", "")

st.sidebar.markdown("---")
st.sidebar.caption(f"Total baris di dataset asli: **{len(df_raw):,}**")

# Terapkan filter dengan validasi
df = df_raw.copy()

if developer:
    df = df[df["developer"].isin(developer)]
else:
    st.sidebar.warning("⚠️ Belum ada developer dipilih — semua data disembunyikan.")

df = df[(df["price"] / 100 >= price_range[0]) & (df["price"] / 100 <= price_range[1])]

if search_name.strip():
    df = df[df["name"].str.contains(search_name.strip(), case=False, na=False)]

if df.empty:
    st.warning("⚠️ Tidak ada game yang cocok dengan filter yang dipilih. Coba ubah filter di sidebar.")
    st.stop()

# =====================================================
# Metrics (kartu custom)
# =====================================================
m1, m2, m3, m4 = st.columns(4)
with m1:
    metric_card("Jumlah Game", f"{len(df):,}", "🎯")
with m2:
    metric_card("Rata-rata CCU", format_big_number(df["ccu"].mean()), "👥")
with m3:
    metric_card("Rata-rata Owners", format_big_number(df["owners"].mean()), "📦")
with m4:
    metric_card("Harga Rata-rata", format_price(df["price"].mean()), "💵")

st.divider()

# =====================================================
# Tabs — struktur konten lebih rapi
# =====================================================
tab_overview, tab_relasi, tab_top, tab_data, tab_prediksi = st.tabs(
    ["📊 Ringkasan", "🔗 Hubungan Antar Variabel", "🏆 Top Game", "📋 Dataset", "🔮 Prediksi"]
)

# ---------------- Tab: Ringkasan ----------------
with tab_overview:
    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            df, x="ccu", nbins=40,
            title="Distribusi Peak CCU",
            color_discrete_sequence=["#66c0f4"],
        )
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            df, x="owners", y="ccu", color="price",
            hover_name="name",
            title="Owners vs CCU",
            color_continuous_scale="Blues",
        )
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    numeric_df = df.select_dtypes(include="number").drop(columns=["appid"], errors="ignore")
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            title="Correlation Heatmap",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Kolom numerik tidak cukup untuk membuat heatmap korelasi.")

# ---------------- Tab: Hubungan Antar Variabel ----------------
with tab_relasi:
    st.subheader("Eksplorasi Hubungan Variabel")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "appid"]

    c1, c2 = st.columns(2)
    x_axis = c1.selectbox("Sumbu X", numeric_cols, index=numeric_cols.index("owners") if "owners" in numeric_cols else 0)
    y_axis = c2.selectbox("Sumbu Y", numeric_cols, index=numeric_cols.index("ccu") if "ccu" in numeric_cols else 1)

    fig = px.scatter(
        df, x=x_axis, y=y_axis, color="developer", hover_name="name",
        title=f"{x_axis} vs {y_axis}",
    )
    fig.update_layout(template="plotly_dark", showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Developer Berdasarkan Jumlah Game")
    dev_count = df["developer"].value_counts().head(10).reset_index()
    dev_count.columns = ["developer", "jumlah_game"]
    fig2 = px.bar(
        dev_count, x="jumlah_game", y="developer", orientation="h",
        title="10 Developer dengan Game Terbanyak",
        color="jumlah_game", color_continuous_scale="Blues",
    )
    fig2.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- Tab: Top Game ----------------
with tab_top:
    st.subheader("Peringkat Game")
    sort_by = st.selectbox("Urutkan berdasarkan", ["ccu", "owners", "positive", "price"], index=0)
    n_top = st.slider("Jumlah game ditampilkan", 5, 25, 10)

    top = df.sort_values(sort_by, ascending=False).head(n_top)
    fig = px.bar(
        top, x="name", y=sort_by,
        title=f"Top {n_top} Game Berdasarkan {sort_by.upper()}",
        color=sort_by, color_continuous_scale="Blues",
    )
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        top[["name", "developer", "owners", "ccu", "positive", "negative", "price"]],
        use_container_width=True,
        hide_index=True,
    )

# ---------------- Tab: Dataset ----------------
with tab_data:
    st.subheader("Dataset Lengkap (Setelah Filter)")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Unduh data terfilter (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_steam_games.csv",
        mime="text/csv",
    )

# ---------------- Tab: Prediksi ----------------
with tab_prediksi:
    st.subheader("🔮 Prediksi Peak Concurrent Users (CCU)")
    st.caption("Model: Linear Regression — fitur: " + ", ".join(MODEL_FEATURES))

    with st.form("prediksi_form"):
        c1, c2 = st.columns(2)
        with c1:
            owners_in = st.number_input("Owners", min_value=0, value=500_000, step=1000)
            positive_in = st.number_input("Positive Review", min_value=0, value=10_000, step=100)
            negative_in = st.number_input("Negative Review", min_value=0, value=500, step=50)
            average_forever_in = st.number_input("Average Forever (menit)", min_value=0, value=300, step=10)
        with c2:
            average_2weeks_in = st.number_input("Average 2 Weeks (menit)", min_value=0, value=20, step=5)
            price_in_usd = st.number_input("Harga (USD)", min_value=0.0, value=19.99, step=0.5, format="%.2f")
            discount_in = st.number_input("Diskon (%)", min_value=0, max_value=100, value=10, step=5)

        submitted = st.form_submit_button("🚀 Prediksi", use_container_width=True)

    if submitted:
        errors = []
        if negative_in > 0 and positive_in == 0 and negative_in > owners_in:
            errors.append("Jumlah review negatif tidak wajar dibanding owners.")
        if positive_in + negative_in > owners_in and owners_in > 0:
            st.warning("⚠️ Total review (positif + negatif) melebihi jumlah owners — hasil prediksi mungkin kurang realistis.")

        if errors:
            for e in errors:
                st.error(f"❌ {e}")
        else:
            try:
                price_cents = int(round(price_in_usd * 100))
                X = pd.DataFrame(
                    [[owners_in, positive_in, negative_in, average_forever_in,
                      average_2weeks_in, price_cents, discount_in]],
                    columns=MODEL_FEATURES,
                )
                pred = model.predict(X)[0]
                pred = max(pred, 0)  # CCU tidak mungkin negatif

                st.success(f"✅ Prediksi Peak CCU: **{pred:,.0f} pemain**")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred,
                    title={"text": "Estimasi Peak CCU"},
                    gauge={
                        "axis": {"range": [0, max(pred * 1.5, df["ccu"].max())]},
                        "bar": {"color": "#66c0f4"},
                    },
                ))
                fig.update_layout(template="plotly_dark", height=300, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Gagal melakukan prediksi: {e}")

st.divider()
st.caption("Dibuat dengan Streamlit • Data: Steam Game Analytics")
