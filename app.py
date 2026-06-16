import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="World Cup 2026 Prediction",
    page_icon="🏆",
    layout="wide"
)


# =========================
# Helper Function
# =========================
def find_file(possible_paths):
    for path in possible_paths:
        if path.exists():
            return path
    return None


@st.cache_resource
def load_model():
    model_path = Path("model/worldcup_match_model.pkl")

    if not model_path.exists():
        return None

    model = joblib.load(model_path)
    return model


@st.cache_data
def load_match_data():
    possible_paths = [
        Path("data/processed/match_processed.csv"),
        Path("data/processed/matches_processed.csv")
    ]

    data_path = find_file(possible_paths)

    if data_path is None:
        return None, None

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df, data_path


@st.cache_data
def load_simulation_result():
    simulation_path = Path("data/processed/tournament_simulation_results.csv")

    if not simulation_path.exists():
        return None

    simulation_df = pd.read_csv(simulation_path)
    return simulation_df


def prepare_team_matches(df):
    home_data = df[[
        "date", "home_team", "away_team",
        "home_score", "away_score", "tournament", "neutral"
    ]].copy()

    home_data.columns = [
        "date", "team", "opponent",
        "goals_for", "goals_against", "tournament", "neutral"
    ]

    home_data["is_home"] = 1

    away_data = df[[
        "date", "away_team", "home_team",
        "away_score", "home_score", "tournament", "neutral"
    ]].copy()

    away_data.columns = [
        "date", "team", "opponent",
        "goals_for", "goals_against", "tournament", "neutral"
    ]

    away_data["is_home"] = 0

    team_matches = pd.concat([home_data, away_data], ignore_index=True)
    team_matches = team_matches.sort_values(["team", "date"]).reset_index(drop=True)

    team_matches["team_result"] = team_matches.apply(get_team_result, axis=1)

    team_matches["points"] = team_matches["team_result"].map({
        "win": 3,
        "draw": 1,
        "loss": 0
    })

    team_matches["win"] = (team_matches["team_result"] == "win").astype(int)

    return team_matches


def get_team_result(row):
    if row["goals_for"] > row["goals_against"]:
        return "win"
    elif row["goals_for"] < row["goals_against"]:
        return "loss"
    else:
        return "draw"


def get_latest_team_stats(team_name, team_matches, window=5):
    team_data = team_matches[team_matches["team"] == team_name].copy()

    if team_data.empty:
        raise ValueError(f"Tim '{team_name}' tidak ditemukan di dataset.")

    latest_matches = team_data.sort_values("date").tail(window)

    stats = {
        "goals_for_last5": latest_matches["goals_for"].mean(),
        "goals_against_last5": latest_matches["goals_against"].mean(),
        "points_last5": latest_matches["points"].mean(),
        "win_rate_last5": latest_matches["win"].mean(),
        "matches_before": len(team_data)
    }

    return stats


def predict_match(model, team_matches, home_team, away_team, tournament="FIFA World Cup", neutral=True):
    home_stats = get_latest_team_stats(home_team, team_matches)
    away_stats = get_latest_team_stats(away_team, team_matches)

    input_data = pd.DataFrame([{
        "home_team": home_team,
        "away_team": away_team,
        "tournament": tournament,
        "neutral": neutral,

        "home_goals_for_last5": home_stats["goals_for_last5"],
        "home_goals_against_last5": home_stats["goals_against_last5"],
        "home_points_last5": home_stats["points_last5"],
        "home_win_rate_last5": home_stats["win_rate_last5"],

        "away_goals_for_last5": away_stats["goals_for_last5"],
        "away_goals_against_last5": away_stats["goals_against_last5"],
        "away_points_last5": away_stats["points_last5"],
        "away_win_rate_last5": away_stats["win_rate_last5"],

        "goals_for_diff_last5": home_stats["goals_for_last5"] - away_stats["goals_for_last5"],
        "goals_against_diff_last5": home_stats["goals_against_last5"] - away_stats["goals_against_last5"],
        "points_diff_last5": home_stats["points_last5"] - away_stats["points_last5"],
        "win_rate_diff_last5": home_stats["win_rate_last5"] - away_stats["win_rate_last5"],
        "experience_diff": home_stats["matches_before"] - away_stats["matches_before"],
        "home_advantage": 0 if neutral else 1
    }])

    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_

    result_df = pd.DataFrame({
        "Result": classes,
        "Probability": probabilities
    })

    label_mapping = {
        "home_win": f"{home_team} menang",
        "draw": "Seri",
        "away_win": f"{away_team} menang"
    }

    result_df["Prediction"] = result_df["Result"].map(label_mapping)
    result_df["Probability (%)"] = (result_df["Probability"] * 100).round(2)

    result_df = result_df[["Prediction", "Probability (%)"]]
    result_df = result_df.sort_values(by="Probability (%)", ascending=False).reset_index(drop=True)

    return result_df


# =========================
# Load Assets
# =========================
model = load_model()
df, data_path = load_match_data()
simulation_df = load_simulation_result()

# =========================
# Sidebar
# =========================
st.sidebar.title("🏆 WC 2026 Prediction")
page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Home",
        "Match Prediction",
        "Tournament Simulation",
        "About Project"
    ]
)

# =========================
# Home Page
# =========================
if page == "Home":
    st.title("🏆 World Cup 2026 Winner Prediction")
    st.write(
        """
        Project ini bertujuan untuk memprediksi hasil pertandingan sepak bola internasional
        dan memperkirakan peluang juara World Cup 2026 menggunakan Machine Learning.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Project Stage", "Dashboard")

    with col2:
        if df is not None:
            st.metric("Jumlah Match Data", df.shape[0])
        else:
            st.metric("Jumlah Match Data", "Tidak ditemukan")

    with col3:
        if model is not None:
            st.metric("Model Status", "Loaded")
        else:
            st.metric("Model Status", "Belum ada")

    st.subheader("Alur Project")
    st.write(
        """
        1. Data Exploration  
        2. Feature Engineering  
        3. Model Training  
        4. Match Prediction  
        5. Tournament Simulation  
        6. Streamlit Dashboard  
        """
    )

    if data_path is not None:
        st.info(f"Dataset digunakan dari: {data_path}")


# =========================
# Match Prediction Page
# =========================
elif page == "Match Prediction":
    st.title("⚽ Match Prediction")

    if model is None:
        st.error("Model belum ditemukan. Pastikan file model/worldcup_match_model.pkl sudah tersedia.")
    elif df is None:
        st.error("Dataset belum ditemukan. Pastikan file data/processed/match_processed.csv tersedia.")
    else:
        team_matches = prepare_team_matches(df)
        teams = sorted(team_matches["team"].unique())

        col1, col2 = st.columns(2)

        with col1:
            home_team = st.selectbox("Pilih Tim A", teams,
                                     index=teams.index("Argentina") if "Argentina" in teams else 0)

        with col2:
            away_team = st.selectbox("Pilih Tim B", teams, index=teams.index("France") if "France" in teams else 1)

        tournament = st.selectbox(
            "Jenis Turnamen",
            ["FIFA World Cup", "Friendly", "FIFA World Cup qualification", "UEFA Euro", "Copa América"],
            index=0
        )

        neutral = st.checkbox("Neutral Venue", value=True)

        if st.button("Prediksi Pertandingan"):
            if home_team == away_team:
                st.warning("Tim A dan Tim B tidak boleh sama.")
            else:
                result_df = predict_match(
                    model=model,
                    team_matches=team_matches,
                    home_team=home_team,
                    away_team=away_team,
                    tournament=tournament,
                    neutral=neutral
                )

                st.subheader(f"Hasil Prediksi: {home_team} vs {away_team}")
                st.dataframe(result_df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(result_df["Prediction"], result_df["Probability (%)"])
                ax.set_title("Probabilitas Hasil Pertandingan")
                ax.set_xlabel("Hasil")
                ax.set_ylabel("Probabilitas (%)")
                ax.set_ylim(0, 100)

                for i, value in enumerate(result_df["Probability (%)"]):
                    ax.text(i, value + 1, f"{value}%", ha="center")

                st.pyplot(fig)

                top_prediction = result_df.iloc[0]
                st.success(
                    f"Prediksi terbesar: {top_prediction['Prediction']} "
                    f"({top_prediction['Probability (%)']}%)"
                )


# =========================
# Tournament Simulation Page
# =========================
elif page == "Tournament Simulation":
    st.title("📊 Tournament Simulation Result")

    if simulation_df is None:
        st.warning(
            """
            File hasil simulasi belum ditemukan.

            Jalankan notebook tournament simulation terlebih dahulu sampai menghasilkan file:
            data/processed/tournament_simulation_results.csv
            """
        )
    else:
        st.subheader("Top 10 Peluang Juara")

        top_10 = simulation_df.sort_values(by="Champion (%)", ascending=False).head(10)

        st.dataframe(
            top_10[["Team", "Champion (%)", "Final (%)", "Semi Final (%)", "Quarter Final (%)"]],
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_10["Team"], top_10["Champion (%)"])
        ax.set_xlabel("Peluang Juara (%)")
        ax.set_ylabel("Tim")
        ax.set_title("Top 10 Prediksi Peluang Juara World Cup 2026")
        ax.invert_yaxis()

        st.pyplot(fig)

        st.subheader("Full Simulation Result")
        st.dataframe(simulation_df, use_container_width=True)


# =========================
# About Project Page
# =========================
elif page == "About Project":
    st.title("ℹ️ About Project")

    st.write(
        """
        Project ini dibuat untuk memprediksi peluang juara World Cup 2026 menggunakan pendekatan Machine Learning.

        Model mempelajari data historis pertandingan internasional, kemudian menggunakan fitur performa tim
        seperti rata-rata gol, rata-rata kebobolan, poin, dan win rate dari pertandingan sebelumnya.
        """
    )

    st.subheader("Fitur yang Digunakan")
    st.write(
        """
        - Rata-rata gol 5 pertandingan terakhir  
        - Rata-rata kebobolan 5 pertandingan terakhir  
        - Rata-rata poin 5 pertandingan terakhir  
        - Win rate 5 pertandingan terakhir  
        - Selisih performa antar tim  
        - Home advantage  
        - Jenis turnamen  
        """
    )

    st.subheader("Catatan")
    st.warning(
        """
        Hasil prediksi bukan kepastian. Model hanya memberikan estimasi probabilitas berdasarkan data historis
        dan fitur yang tersedia.
        """
    )