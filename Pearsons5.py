import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# =====================================================
# PAGE SETUP
# =====================================================

st.set_page_config(
    page_title="Psychoacoustic Correlation Tool",
    layout="wide"
)

st.title("Psychoacoustic Metrics vs Mean Jury Ratings")

st.write("""
This tool correlates the mean jury rating of each machine against:

• Loudness (sone)
• Sharpness (acum)
• Roughness (asper)
• Fluctuation Strength (vacil)
• Psychoacoustic Annoyance (PA)
""")

# =====================================================
# INPUTS
# =====================================================

col1, col2 = st.columns(2)

with col1:
    num_machines = st.number_input(
        "Number of Machines",
        min_value=2,
        value=5,
        step=1
    )

with col2:
    num_participants = st.number_input(
        "Number of Participants",
        min_value=1,
        value=10,
        step=1
    )

num_machines = int(num_machines)
num_participants = int(num_participants)

# =====================================================
# MACHINE METRICS TABLE
# =====================================================

st.subheader("Machine Psychoacoustic Metrics")

metrics_df = pd.DataFrame({
    "Machine": [f"M{i+1}" for i in range(num_machines)],
    "Loudness": [0.0]*num_machines,
    "Sharpness": [0.0]*num_machines,
    "Roughness": [0.0]*num_machines,
    "Fluctuation": [0.0]*num_machines,
    "PA": [0.0]*num_machines
})

metrics_df = st.data_editor(
    metrics_df,
    use_container_width=True,
    num_rows="fixed"
)

# =====================================================
# RATINGS TABLE
# =====================================================

st.subheader("Participant Ratings (0–10)")

rating_columns = [f"M{i+1}" for i in range(num_machines)]

ratings_df = pd.DataFrame(
    np.zeros((num_participants, num_machines)),
    columns=rating_columns
)

ratings_df.index = [
    f"P{i+1}" for i in range(num_participants)
]

ratings_df = st.data_editor(
    ratings_df,
    use_container_width=True,
    num_rows="fixed"
)

# =====================================================
# CALCULATE
# =====================================================

if st.button("Calculate Correlations"):

    try:

        # -----------------------------------------
        # Mean ratings per machine
        # -----------------------------------------

        mean_ratings = ratings_df.mean(axis=0).values
        std_ratings = ratings_df.std(axis=0).values

        summary_df = pd.DataFrame({
            "Machine": metrics_df["Machine"],
            "Loudness": metrics_df["Loudness"],
            "Sharpness": metrics_df["Sharpness"],
            "Roughness": metrics_df["Roughness"],
            "Fluctuation": metrics_df["Fluctuation"],
            "PA": metrics_df["PA"],
            "Mean Rating": np.round(mean_ratings, 3),
            "Std Dev": np.round(std_ratings, 3)
        })

        st.subheader("Machine Summary")

        st.dataframe(
            summary_df,
            use_container_width=True
        )

        # -----------------------------------------
        # Correlation calculations
        # -----------------------------------------

        results = []

        metric_names = [
            "Loudness",
            "Sharpness",
            "Roughness",
            "Fluctuation",
            "PA"
        ]

        for metric in metric_names:

            x = summary_df[metric].astype(float)
            y = summary_df["Mean Rating"].astype(float)

            if len(np.unique(x)) < 2:

                results.append({
                    "Metric": metric,
                    "r": np.nan,
                    "R²": np.nan,
                    "p-value": np.nan
                })

                continue

            r, p = pearsonr(x, y)

            results.append({
                "Metric": metric,
                "r": round(r, 4),
                "R²": round(r**2, 4),
                "p-value": round(p, 6)
            })

        results_df = pd.DataFrame(results)

        st.subheader("Correlation Results")

        st.dataframe(
            results_df.sort_values(
                by="R²",
                ascending=False
            ),
            use_container_width=True
        )

        # -----------------------------------------
        # Individual Metric Plots
        # -----------------------------------------

        st.subheader("Metric vs Mean Jury Rating")

        for metric in metric_names:

            x = summary_df[metric].astype(float)
            y = summary_df["Mean Rating"].astype(float)

            if len(np.unique(x)) < 2:
                continue

            r_plot, p_plot = pearsonr(x, y)

            slope, intercept = np.polyfit(
                x,
                y,
                1
            )

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            ax.scatter(
                x,
                y,
                s=80
            )

            # Label machines
            for i in range(len(summary_df)):
                ax.annotate(
                    summary_df["Machine"][i],
                    (x.iloc[i], y.iloc[i])
                )

            x_line = np.linspace(
                min(x),
                max(x),
                100
            )

            y_line = (
                slope * x_line +
                intercept
            )

            ax.plot(
                x_line,
                y_line,
                linewidth=2
            )

            ax.set_xlabel(metric)
            ax.set_ylabel("Mean Jury Rating")

            ax.set_title(
                f"{metric} vs Mean Jury Rating\n"
                f"r = {r_plot:.3f}, "
                f"R² = {r_plot**2:.3f}, "
                f"p = {p_plot:.4f}"
            )

            ax.grid(True)

            st.pyplot(fig)

        # -----------------------------------------
        # R² Comparison Chart
        # -----------------------------------------

        st.subheader("R² Comparison")

        valid_results = results_df.dropna()

        fig2, ax2 = plt.subplots(
            figsize=(8, 4)
        )

        ax2.bar(
            valid_results["Metric"],
            valid_results["R²"]
        )

        ax2.set_ylabel("R²")
        ax2.set_title(
            "Predictive Power of Each Metric"
        )

        ax2.grid(True)

        st.pyplot(fig2)

        # -----------------------------------------
        # Best Predictor
        # -----------------------------------------

        best_metric = valid_results.loc[
            valid_results["R²"].idxmax()
        ]

        st.subheader("Best Predictor")

        st.success(
            f"{best_metric['Metric']} "
            f"(R² = {best_metric['R²']:.4f})"
        )

    except Exception as e:
        st.error(str(e))
