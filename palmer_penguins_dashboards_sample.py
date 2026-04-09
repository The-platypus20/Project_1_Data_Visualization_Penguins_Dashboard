"""Palmer Penguins dashboard built with Dash.

Run:
    python palmer_penguins_dashboards.py
"""

from __future__ import annotations

import textwrap

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from palmerpenguins import load_penguins
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SPECIES_COLORS = {
    "Adelie": "#d1495b",
    "Chinstrap": "#2b59c3",
    "Gentoo": "#2a9d8f",
}

CLUSTER_COLORS = {
    "Cluster 1": "#264653",
    "Cluster 2": "#f4a261",
    "Cluster 3": "#e76f51",
}

NUMERIC_COLUMNS = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

AXIS_LABELS = {
    "bill_length_mm": "Bill length (mm)",
    "bill_depth_mm": "Bill depth (mm)",
    "flipper_length_mm": "Flipper length (mm)",
    "body_mass_g": "Body mass (g)",
    "bill_ratio": "Bill length/depth ratio",
}


def preprocess_penguins() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load data, clean it, and create derived fields for dashboard use."""
    raw_df = load_penguins()
    df = raw_df.copy()

    df["sex"] = df["sex"].fillna("Unknown").str.title()
    df["species"] = df["species"].str.title()
    df["island"] = df["island"].str.title()

    # Keep rows with complete core measurements for the analytical views.
    df = df.dropna(subset=NUMERIC_COLUMNS).reset_index(drop=True)

    df["bill_ratio"] = (df["bill_length_mm"] / df["bill_depth_mm"]).round(2)
    df["body_mass_kg"] = (df["body_mass_g"] / 1000).round(2)
    df["penguin_id"] = np.arange(1, len(df) + 1)

    ml_df = df.copy()
    scaled = StandardScaler().fit_transform(ml_df[NUMERIC_COLUMNS])

    pca_model = PCA(n_components=2, random_state=42)
    pca_projection = pca_model.fit_transform(scaled)
    ml_df["pc1"] = pca_projection[:, 0]
    ml_df["pc2"] = pca_projection[:, 1]

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    ml_df["cluster"] = [f"Cluster {value + 1}" for value in kmeans.fit_predict(scaled)]

    return df, ml_df


BASE_DF, ML_DF = preprocess_penguins()
PCA_VARIANCE = PCA(n_components=2, random_state=42).fit(
    StandardScaler().fit_transform(BASE_DF[NUMERIC_COLUMNS])
).explained_variance_ratio_


def filter_dataframe(
    df: pd.DataFrame,
    species: list[str],
    islands: list[str],
    sexes: list[str],
    body_mass_range: list[int],
) -> pd.DataFrame:
    filtered = df[
        df["species"].isin(species)
        & df["island"].isin(islands)
        & df["sex"].isin(sexes)
        & df["body_mass_g"].between(body_mass_range[0], body_mass_range[1])
    ]
    return filtered.copy()


CARD_STYLE = {
    "background": "#ffffff",
    "borderRadius": "18px",
    "padding": "20px",
    "boxShadow": "0 10px 30px rgba(55, 73, 87, 0.08)",
}

PAGE_STYLE = {
    "backgroundColor": "#f2efe8",
    "minHeight": "100vh",
    "fontFamily": "Segoe UI, Arial, sans-serif",
    "padding": "24px",
}


def filter_block(label: str, component: dcc.Dropdown | dcc.RangeSlider) -> html.Div:
    return html.Div(
        [html.Label(label, style={"fontWeight": 600, "display": "block", "marginBottom": "8px"}), component],
        style={"minWidth": "220px", "flex": "1 1 220px"},
    )


def build_controls() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H1(
                        "Palmer Penguins Atlas",
                        style={"marginBottom": "8px", "color": "#1f2933"},
                    ),
                    html.P(
                        "A Dash dashboard for comparing penguin morphology across islands "
                        "and inspecting a PCA + K-Means view of the dataset.",
                        style={"marginBottom": "0", "color": "#52606d", "fontSize": "17px"},
                    ),
                ]
            ),
            html.Hr(style={"margin": "20px 0", "borderColor": "#d9d3c7"}),
            html.Div(
                [
                    filter_block(
                        "Species",
                        dcc.Dropdown(
                            id="species-filter",
                            options=[
                                {"label": value, "value": value}
                                for value in sorted(BASE_DF["species"].unique())
                            ],
                            value=sorted(BASE_DF["species"].unique()),
                            multi=True,
                            clearable=False,
                        ),
                    ),
                    filter_block(
                        "Island",
                        dcc.Dropdown(
                            id="island-filter",
                            options=[
                                {"label": value, "value": value}
                                for value in sorted(BASE_DF["island"].unique())
                            ],
                            value=sorted(BASE_DF["island"].unique()),
                            multi=True,
                            clearable=False,
                        ),
                    ),
                    filter_block(
                        "Sex",
                        dcc.Dropdown(
                            id="sex-filter",
                            options=[
                                {"label": value, "value": value}
                                for value in sorted(BASE_DF["sex"].unique())
                            ],
                            value=sorted(BASE_DF["sex"].unique()),
                            multi=True,
                            clearable=False,
                        ),
                    ),
                    filter_block(
                        "Scatter X-axis",
                        dcc.Dropdown(
                            id="x-axis",
                            options=[
                                {"label": label, "value": key}
                                for key, label in AXIS_LABELS.items()
                            ],
                            value="bill_length_mm",
                            clearable=False,
                        ),
                    ),
                    filter_block(
                        "Scatter Y-axis",
                        dcc.Dropdown(
                            id="y-axis",
                            options=[
                                {"label": label, "value": key}
                                for key, label in AXIS_LABELS.items()
                            ],
                            value="flipper_length_mm",
                            clearable=False,
                        ),
                    ),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    html.Label(
                        "Body mass filter (g)",
                        style={"fontWeight": 600, "display": "block", "marginBottom": "8px"},
                    ),
                    dcc.RangeSlider(
                        id="mass-filter",
                        min=int(BASE_DF["body_mass_g"].min()),
                        max=int(BASE_DF["body_mass_g"].max()),
                        step=50,
                        value=[
                            int(BASE_DF["body_mass_g"].min()),
                            int(BASE_DF["body_mass_g"].max()),
                        ],
                        marks={value: f"{value}" for value in range(3000, 6501, 1000)},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ],
                style={"marginTop": "22px"},
            ),
        ],
        style={**CARD_STYLE, "marginBottom": "18px", "background": "#f7f7ef"},
    )


def metric_card(title: str, card_id: str) -> html.Div:
    return html.Div(
        [
            html.Div(
                title,
                style={
                    "textTransform": "uppercase",
                    "fontSize": "12px",
                    "letterSpacing": "0.08em",
                    "color": "#7b8794",
                },
            ),
            html.Div(id=card_id, style={"fontSize": "34px", "fontWeight": 700, "color": "#102a43"}),
        ],
        style={**CARD_STYLE, "flex": "1 1 220px"},
    )


def graph_card(graph_id: str) -> html.Div:
    return html.Div(dcc.Graph(id=graph_id), style=CARD_STYLE)


app = dash.Dash(__name__)
server = app.server
app.title = "Palmer Penguins Dashboard"

app.layout = html.Div(
    [
        build_controls(),
        html.Div(
            [
                metric_card("Penguins in view", "metric-count"),
                metric_card("Average body mass", "metric-mass"),
                metric_card("Average bill ratio", "metric-ratio"),
            ],
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "18px"},
        ),
        html.Div(
            [
                html.Div(graph_card("scatter-chart"), style={"flex": "2 1 620px"}),
                html.Div(
                    html.Div(
                        [
                            html.H4("Dynamic story guide", style={"marginBottom": "12px"}),
                            dcc.Markdown(id="story-summary", className="mb-0"),
                        ],
                        style=CARD_STYLE,
                    ),
                    style={"flex": "1 1 320px"},
                ),
            ],
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "18px"},
        ),
        html.Div(
            [
                html.Div(graph_card("island-bar-chart"), style={"flex": "1 1 480px"}),
                html.Div(graph_card("distribution-chart"), style={"flex": "1 1 480px"}),
            ],
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "18px"},
        ),
        html.Div(
            [
                html.Div(graph_card("pca-chart"), style={"flex": "2 1 620px"}),
                html.Div(
                    html.Div(
                        [
                            html.H4("Machine learning note", style={"marginBottom": "12px"}),
                            dcc.Markdown(
                                textwrap.dedent(
                                    f"""
                                    PCA compresses the four body measurements into two axes.
                                    In the full cleaned dataset, **PC1 explains {PCA_VARIANCE[0] * 100:.1f}%**
                                    of the variance and **PC2 explains {PCA_VARIANCE[1] * 100:.1f}%**.

                                    K-Means is then applied in the standardized 4D feature space and
                                    plotted on top of the PCA projection so you can compare:

                                    - the natural species labels
                                    - the unsupervised cluster assignments
                                    - where overlap or separation happens
                                    """
                                ).strip()
                            ),
                        ],
                        style={**CARD_STYLE, "background": "#fcf4ea"},
                    ),
                    style={"flex": "1 1 320px"},
                ),
            ],
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
        ),
    ],
    style=PAGE_STYLE,
)


@app.callback(
    Output("metric-count", "children"),
    Output("metric-mass", "children"),
    Output("metric-ratio", "children"),
    Output("scatter-chart", "figure"),
    Output("island-bar-chart", "figure"),
    Output("distribution-chart", "figure"),
    Output("pca-chart", "figure"),
    Output("story-summary", "children"),
    Input("species-filter", "value"),
    Input("island-filter", "value"),
    Input("sex-filter", "value"),
    Input("mass-filter", "value"),
    Input("x-axis", "value"),
    Input("y-axis", "value"),
)
def update_dashboard(
    species: list[str],
    islands: list[str],
    sexes: list[str],
    body_mass_range: list[int],
    x_axis: str,
    y_axis: str,
):
    filtered_df = filter_dataframe(BASE_DF, species, islands, sexes, body_mass_range)
    filtered_ml_df = filter_dataframe(ML_DF, species, islands, sexes, body_mass_range)

    if filtered_df.empty:
        empty_figure = go.Figure().update_layout(
            template="plotly_white",
            annotations=[
                dict(
                    text="No penguins match the current filters.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=18),
                )
            ],
        )
        return (
            "0",
            "N/A",
            "N/A",
            empty_figure,
            empty_figure,
            empty_figure,
            empty_figure,
            "Adjust the filters to bring penguins back into view.",
        )

    count_text = f"{len(filtered_df)}"
    mass_text = f"{filtered_df['body_mass_g'].mean():.0f} g"
    ratio_text = f"{filtered_df['bill_ratio'].mean():.2f}"

    scatter_fig = px.scatter(
        filtered_df,
        x=x_axis,
        y=y_axis,
        color="species",
        symbol="island",
        hover_data=["sex", "body_mass_g", "bill_ratio"],
        color_discrete_map=SPECIES_COLORS,
        title=f"{AXIS_LABELS[x_axis]} vs {AXIS_LABELS[y_axis]}",
    )
    scatter_fig.update_traces(marker=dict(size=11, line=dict(width=0.7, color="white")))
    scatter_fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text="Use color for species and symbol for island to spot separation patterns.",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
    )
    scatter_fig.update_layout(template="plotly_white", legend_title_text="", height=460)

    island_counts = (
        filtered_df.groupby(["island", "species"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    island_bar_fig = px.bar(
        island_counts,
        x="island",
        y="count",
        color="species",
        barmode="group",
        color_discrete_map=SPECIES_COLORS,
        title="Species counts across islands",
    )
    island_bar_fig.update_layout(template="plotly_white", legend_title_text="", height=400)

    distribution_fig = px.histogram(
        filtered_df,
        x="body_mass_g",
        color="species",
        marginal="box",
        nbins=20,
        opacity=0.75,
        color_discrete_map=SPECIES_COLORS,
        title="Body mass distribution",
    )
    distribution_fig.add_vline(
        x=filtered_df["body_mass_g"].mean(),
        line_dash="dash",
        line_color="#222222",
        annotation_text="Filtered mean",
    )
    distribution_fig.update_layout(template="plotly_white", legend_title_text="", height=400)

    pca_fig = px.scatter(
        filtered_ml_df,
        x="pc1",
        y="pc2",
        color="cluster",
        symbol="species",
        hover_data=["island", "sex"],
        color_discrete_map=CLUSTER_COLORS,
        title="PCA projection with K-Means clusters",
    )
    pca_fig.update_traces(marker=dict(size=11, line=dict(width=0.5, color="white")))
    pca_fig.add_annotation(
        x=0.01,
        y=0.01,
        xref="paper",
        yref="paper",
        text="Clusters are computed from standardized measurements, not the species labels.",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
    )
    pca_fig.update_layout(template="plotly_white", legend_title_text="", height=460)
    pca_fig.update_xaxes(title=f"PC1 ({PCA_VARIANCE[0] * 100:.1f}% variance)")
    pca_fig.update_yaxes(title=f"PC2 ({PCA_VARIANCE[1] * 100:.1f}% variance)")

    species_mass = (
        filtered_df.groupby("species")["body_mass_g"].mean().sort_values(ascending=False)
    )
    dominant_species = species_mass.index[0]
    lightest_species = species_mass.index[-1]
    top_island = filtered_df["island"].value_counts().idxmax()

    story_summary = textwrap.dedent(
        f"""
        **Filtered snapshot**

        - **{dominant_species}** has the highest average body mass in the current view at
          **{species_mass.iloc[0]:.0f} g**.
        - **{lightest_species}** is the lightest species in the current subset at
          **{species_mass.iloc[-1]:.0f} g** on average.
        - **{top_island}** contributes the largest number of penguins after filtering.

        **Reading guide**

        - Use the scatter plot to compare physical traits directly.
        - Use the histogram to see whether species overlap in body mass.
        - Use the PCA chart to compare human labels with unsupervised clusters.
        """
    ).strip()

    return (
        count_text,
        mass_text,
        ratio_text,
        scatter_fig,
        island_bar_fig,
        distribution_fig,
        pca_fig,
        story_summary,
    )


if __name__ == "__main__":
    app.run(debug=True)
