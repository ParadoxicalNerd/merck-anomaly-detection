"""
Library to analyze the collected healthkit data. 
Recommended workflow: clean_data -> merge_users -> 
    calcuate_pca -> calculate_dist_metric -> caclulate_threshold

@author: Pankaj Meghani
@date: 2021/10/27
"""

import pickle
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from pandas.core.frame import DataFrame
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import webbrowser

from src.SQL_Interface import SQL
from src.XML_to_SQL import XML_to_SQL


pio.renderers.default = "browser"

debug = False


class Healthkit:
    def clean_data(self, healthcare_records: DataFrame) -> list:
        # healthcare_records = self.healthcare_records
        cleaned_data = []

        for i in healthcare_records["user_id"].unique():
            data = healthcare_records[healthcare_records.user_id == i]
            # pivot
            pivot_df = data.pivot_table(index="endDate", columns="type", values="value")
            pivot_df.index = pd.to_datetime(pivot_df.index)

            # splice based on when we start collecting smart-watch data
            index = pivot_df["WalkingStepLength"].first_valid_index()
            pivot_df: pd.DataFrame = pivot_df[index:]

            # resample based on values we want
            aggs = {
                "HeartRate": np.nanmean,
                "StepCount": np.sum,
                "BasalEnergyBurned": np.sum,
                "ActiveEnergyBurned": np.sum,
                "FlightsClimbed": np.sum,
                "AppleExerciseTime": np.sum,
                "AppleStandTime": np.sum,
                "WalkingSpeed": np.nanmean,
                "WalkingStepLength": np.nanmean,
                "WalkingAsymmetryPercentage": np.nanmean,
            }

            df = pivot_df.resample("H").agg(
                {k: v for k, v in aggs.items() if k in pivot_df}
            )

            # fill nan values
            df = df.fillna(df.mean())

            # drop residual columns
            df = df.drop(
                [
                    "DistanceWalkingRunning",
                    "RestingHeartRate",
                    "VO2Max",
                    "WalkingHeartRateAverage",
                    "WalkingDoubleSupportPercentage",
                    "SixMinuteWalkTestDistance",
                    "HighHeartRateEvent",
                    "HeartRateVariabilitySDNN",
                    "StairAscentSpeed",
                    "StairDescentSpeed",
                ],
                axis=1,
                errors="ignore",
            )

            # scale for improved analysis
            scaler = MinMaxScaler()
            df = pd.DataFrame(
                scaler.fit_transform(df), columns=df.columns, index=df.index
            )

            cleaned_data.append(df)

        # self.clean_data = cleaned_data
        return cleaned_data

    def merge_users(self, data: list) -> DataFrame:
        df = pd.concat(data, ignore_index=True)
        df = df.fillna(df.mean())
        return df

    def calcuate_pca(self, df: DataFrame, model_path=None) -> DataFrame:
        if model_path == None:
            pca = PCA(n_components=3, svd_solver="full")
            X_PCA = pca.fit_transform(df)  # can just be obj.df
        else:
            pca: PCA = pickle.load(open(model_path, "rb"))
            X_PCA = pca.transform(df)

        X_PCA: DataFrame = pd.DataFrame(X_PCA)
        X_PCA.index = df.index

        self.pca = pca
        return X_PCA

    def calculate_dist_metric(
        self, X_PCA, model: Union[None, str] = None
    ) -> np.ndarray:
        # MinCovDet is more robust alternative to Mahalanobis Distance
        # https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/covariance/plot_mahalanobis_distances.html
        if model == None:
            robust_cov = MinCovDet().fit(X_PCA)
        else:
            robust_cov = pickle.load(open(model, "rb"))
            # TODO: Check if path exists first

        dist = robust_cov.mahalanobis(X_PCA)
        dist = np.sqrt(dist)  # Normalizing distance

        self.robust_cov = robust_cov

        return dist

    def caclulate_threshold(self, dist: np.ndarray, k: int = 4) -> float:
        return np.mean(dist) + np.std(dist) * k

    def export_pca(self, path: str) -> None:
        pickle.dump(self.pca, open(path, "wb"))

    def export_dist_metric(self, path: str) -> None:
        pickle.dump(self.robust_cov, open(path, "wb"))

    def plot_cleaned_histograms(self, cleaned_data: list) -> None:
        for i in range(len(cleaned_data)):
            plt.figure(figsize=(12, 4))
            sns.lineplot(data=cleaned_data[i]).legend(
                loc="center left", bbox_to_anchor=(1, 0.5)
            )
            plt.show()

    def plot_3d_points(
        self, data: DataFrame, dist: np.ndarray, export_path: Union[str, None] = None
    ) -> None:

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=data[0],
                    y=data[1],
                    z=data[2],
                    mode="markers",
                    marker=dict(
                        size=2,
                        color=dist,  # set color to an array/list of desired values
                    ),
                )
            ]
        )

        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

        if export_path:
            fig.write_html(export_path)  # Modifiy the html file

        try:
            fig.show()
        except webbrowser.Error as e:
            print("Web browser not found")


if __name__ == "__main__":
    # Example usage of module

    # read healthcare records
    conn, cur = SQL("./db.sqlite").create_connection()
    healthcare_records = pd.read_sql_query("SELECT * FROM healthkit_records", conn)

    hk = Healthkit()
    cleaned_data = hk.clean_data(healthcare_records)

    if debug:
        hk.plot_cleaned_histograms(cleaned_data)

    df = hk.merge_users(cleaned_data)
    X_PCA = hk.calcuate_pca(df, "pca_model.pkl")
    dist = hk.calculate_dist_metric(X_PCA, "dist_metric_model.pkl")

    # plt.figure()
    # sns.histplot(dist[dist < 200], bins=25, kde=True, color="green")
    # plt.xlabel("Mahalanobis dist")
    # plt.show()

    hk.plot_3d_points(X_PCA, dist)

    threshold = hk.caclulate_threshold(dist)

    # %%
    # pd.DataFrame({"threshold": threshold, "dist": dist}).plot()

    # region
    """
    # plt.matshow(pca.components_, cmap="viridis")
    # plt.yticks([0, 1], ["First component", "Second component"])
    # plt.colorbar()
    # plt.xticks(range(len(df.columns)), df.columns, rotation=90)
    # plt.xlabel("Feature")
    # plt.ylabel("Principal components")
    # plt.show()

    # plt.scatter(X_PCA.iloc[:, 0], X_PCA.iloc[:, 1])
    # plt.gca().set_aspect("equal")
    # plt.xlabel("First principal component")
    # plt.ylabel("Second principal component")
    # plt.scatter([X_PCA[0].mean()], [X_PCA[1].mean()])
    """
    # endregion
