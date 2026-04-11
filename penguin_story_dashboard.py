from __future__ import annotations

import base64
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "penguins_preprocessed.csv"
LOGO_PATH = ROOT / "palmerpenguins.png"

SPECIES_LABELS = {
    "Adelie Penguin (Pygoscelis adeliae)": "Adelie",
    "Chinstrap penguin (Pygoscelis antarctica)": "Chinstrap",
    "Gentoo penguin (Pygoscelis papua)": "Gentoo",
}
SPECIES_COLORS = {
    "Adelie": "#F97316",
    "Chinstrap": "#8B5CF6",
    "Gentoo": "#14B8A6",
}
SPECIES_ORDER = ["Adelie", "Gentoo", "Chinstrap"]
CLUSTER_COLORS = {
    "Cluster 1": "#0F766E",
    "Cluster 2": "#FDA3DC",
    "Cluster 3": "#58A9FF",
}

CLUSTER_COLORS = {
    "Cluster 1": "#334155",  # slate dark
    "Cluster 2": "#64748B",  # slate medium
    "Cluster 3": "#A3AFD8",  # slate light (or #CBD5E1)
}

ACCENT = "#0F766E"
ACCENT_SOFT = "#DDF4F1"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#F8FCFB"
INK = "#112531"
MUTED = "#5B6E79"
BORDER = "#D9E5E2"
TITLE_FONT = "Aptos Display, Georgia, serif"
BODY_FONT = "Aptos, Segoe UI, sans-serif"


def load_logo(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def prepare_dataframe() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data["species_short"] = data["species"].map(SPECIES_LABELS)
    data["species_short"] = pd.Categorical(
        data["species_short"],
        categories=SPECIES_ORDER,
        ordered=True,
    )
    data["sex"] = data["sex"].fillna("UNKNOWN").str.upper()
    return data


df = prepare_dataframe()
logo_src = load_logo(LOGO_PATH)
px.defaults.template = "plotly_white"


def filtered_df(species_values: list[str] | None, island_values: list[str] | None) -> pd.DataFrame:
    data = df.copy()
    if species_values:
        data = data[data["species_short"].isin(species_values)]
    if island_values:
        data = data[data["island"].isin(island_values)]
    return data

def annotation_container(text: str) -> html.Div:
    return annotation_block(text)


def summarize_species_count(data: pd.DataFrame) -> str:
    if data.empty:
        return "No penguins match the current filters."
    counts = data["species_short"].astype("object").value_counts()
    if len(counts) == 1:
        return f"Only **{counts.index[0]}** penguins remain in the current view."
    lead = counts.index[0]
    runner = counts.index[1]
    return f"Within the selected subset, **{lead}** remains the most common species, ahead of **{runner}**."


def summarize_island_mix(data: pd.DataFrame) -> str:
    if data.empty:
        return "No island composition is available for the current filters."
    counts = data.groupby(["island", "species_short"], observed=False).size().reset_index(name="count")
    top_row = counts.sort_values("count", ascending=False).iloc[0]
    return f"Within the current selection, **{top_row['species_short']}** has its strongest presence on **{top_row['island']}**."


def summarize_scatter(data: pd.DataFrame) -> str:
    if data.empty:
        return "The current filters leave no points in the morphology scatter."
    summary = data.groupby("species_short", observed=False)[["flipper_length_mm", "bill_length_mm"]].mean().dropna()
    if len(summary) < 2:
        species = summary.index[0]
        return f"The selected view focuses on **{species}**, so the scatter shows within-species variation rather than cross-species separation."
    longest_flipper = summary["flipper_length_mm"].idxmax()
    longest_bill = summary["bill_length_mm"].idxmax()
    return f"In the selected subset, **{longest_flipper}** has the longest average flippers, while **{longest_bill}** shows the longest bills."


def summarize_sex_difference(data: pd.DataFrame) -> str:
    subset = data[data["sex"].isin(["MALE", "FEMALE"])].copy()
    if subset.empty:
        return "Sex-based comparison is unavailable for the current filters."
    grouped = subset.groupby(["species_short", "sex"], observed=False)[["bill_length_mm", "bill_depth_mm"]].mean().reset_index()
    species_stats = grouped.groupby("species_short", observed=False)
    stronger = []
    for species, frame in species_stats:
        male = frame[frame["sex"] == "MALE"]
        female = frame[frame["sex"] == "FEMALE"]
        if male.empty or female.empty:
            continue
        if float(male["bill_length_mm"].iloc[0]) >= float(female["bill_length_mm"].iloc[0]):
            stronger.append(species)
    if not stronger:
        return "Sex differences are muted in the current selection."
    if len(stronger) == 1:
        return f"Within the selected view, **{stronger[0]}** still shows the clearest male-above-female separation."
    return f"Across the selected species, **male penguins remain larger on average** for {', '.join(stronger[:-1]) + (' and ' + stronger[-1] if len(stronger) > 1 else stronger[0])}."


def summarize_body_mass(data: pd.DataFrame) -> str:
    if data.empty:
        return "No body-mass distribution is available for the current filters."
    masses = data.groupby("species_short", observed=False)["body_mass_g"].mean().dropna()
    if len(masses) == 1:
        species = masses.index[0]
        return f"Only **{species}** is visible, so the violin emphasizes spread and outliers within that species."
    heaviest = masses.idxmax()
    return f"Within the selected subset, **{heaviest}** has the highest average body mass."


def summarize_bill_profile(data: pd.DataFrame) -> str:
    if data.empty:
        return "No bill profile can be computed for the current filters."
    bills = data.groupby("species_short", observed=False)[["bill_length_mm", "bill_depth_mm"]].mean().dropna()
    if len(bills) == 1:
        species = bills.index[0]
        return f"The current filters isolate **{species}**, so this panel shows its average bill proportions only."
    longest = bills["bill_length_mm"].idxmax()
    deepest = bills["bill_depth_mm"].idxmax()
    if longest == deepest:
        return f"Within the selected species, **{longest}** has both the longest and deepest average bill."
    return f"Within the selected species, **{longest}** has the longest bill, while **{deepest}** has the deepest bill."


def summarize_ecology(data: pd.DataFrame) -> str:
    eco = data.dropna(subset=["delta_13_c", "delta_15_n"])
    if eco.empty:
        return "No isotope measurements are available for the current filters."
    eco_means = eco.groupby("species_short", observed=False)[["delta_13_c", "delta_15_n"]].mean().dropna()
    if len(eco_means) == 1:
        species = eco_means.index[0]
        return f"The isotope view currently focuses on **{species}** only."
    carbon = eco_means["delta_13_c"].idxmax()
    nitrogen = eco_means["delta_15_n"].idxmax()
    return f"In the current selection, **{carbon}** reaches the highest average Delta 13 C, while **{nitrogen}** reaches the highest Delta 15 N."


def summarize_pca_subset(data: pd.DataFrame) -> str:
    if data.empty:
        return "**Note:** Model is trained on the full dataset. The current PCA view has no points after filtering."
    visible_species = list(data["species_short"].dropna().unique())
    if len(visible_species) == 1:
        species_text = visible_species[0]
    else:
        species_text = ", ".join(visible_species[:-1]) + f" and {visible_species[-1]}" if len(visible_species) > 1 else "the selected subset"
    return f"**Note:** Model is trained on the full dataset. Current PCA view shows **{species_text}** only."


def style_figure(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=BODY_FONT, color=INK, size=13),
        title=dict(font=dict(family=TITLE_FONT, size=22, color=INK), x=0),
        margin=dict(l=24, r=20, t=82, b=38),
        hovermode="closest",
        hoverlabel=dict(font=dict(family=BODY_FONT, size=13, color="white")),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.2,
            xanchor="right",
            x=1,
            title=None,
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="rgba(217,229,226,0.9)",
            borderwidth=1,
        ),
        legend_itemclick=False,
        legend_itemdoubleclick=False,
    )
    fig.update_xaxes(gridcolor="#E9F0EE", zeroline=False, linecolor=BORDER, tickfont=dict(color=MUTED), title_font=dict(color=INK))
    fig.update_yaxes(gridcolor="#E9F0EE", zeroline=False, linecolor=BORDER, tickfont=dict(color=MUTED), title_font=dict(color=INK))
    return fig


def color_hoverlabels(fig: go.Figure) -> go.Figure:
    for trace in fig.data:
        color = None
        marker = getattr(trace, "marker", None)
        line = getattr(trace, "line", None)
        if marker and isinstance(getattr(marker, "color", None), str):
            color = marker.color
        elif line and isinstance(getattr(line, "color", None), str):
            color = line.color
        if color:
            trace.hoverlabel = dict(bgcolor=color, bordercolor=color, font=dict(color="white", family=BODY_FONT, size=13))
    return fig


def card(title: str, body, subtle: bool = False, extra_style: dict | None = None) -> html.Div:
    style = {
        "background": SURFACE_ALT if subtle else SURFACE,
        "border": f"1px solid {BORDER}",
        "borderRadius": "24px",
        "padding": "22px",
        "boxShadow": "0 16px 40px rgba(17, 37, 49, 0.06)",
        "height": "100%",
    }
    if extra_style:
        style.update(extra_style)
    return html.Div(
        [
            html.Div(
                title,
                style={
                    "fontSize": "0.76rem",
                    "letterSpacing": "0.12em",
                    "textTransform": "uppercase",
                    "fontWeight": "700",
                    "color": ACCENT,
                    "marginBottom": "10px",
                },
            ),
            body,
        ],
        style=style,
    )


def annotation_block(text: str) -> html.Div:
    return html.Div(
        dcc.Markdown(text, style={"margin": 0}),
        style={
            "marginTop": "14px",
            "padding": "12px 14px",
            "borderRadius": "16px",
            "background": "#F0F7F5",
            "color": INK,
            "fontSize": "0.96rem",
            "lineHeight": 1.55,
        },
    )


def graph_card(title: str, subtitle: str, figure_id: str, note: str, height: int, note_id: str | None = None, title_id: str | None = None) -> html.Div:
    title_props = {
        "style": {
            "fontFamily": TITLE_FONT,
            "fontWeight": "700",
            "fontSize": "1.55rem",
            "margin": "0 0 6px 0",
            "color": INK,
        }
    }
    if title_id is not None:
        title_props["id"] = title_id
    return card(
        "",
        html.Div(
            [
                html.H3(title, **title_props),
                html.P(subtitle, style={"margin": "0 0 8px 0", "color": MUTED, "fontSize": "0.96rem"}),
                dcc.Graph(id=figure_id, config={"displayModeBar": False}, style={"height": f"{height}px"}),
                annotation_block(note) if note_id is None else html.Div(id=note_id, children=annotation_block(note)),
            ]
        ),
    )


def kpi_card(title: str, value_id: str, subtitle: str) -> dbc.Col:
    return dbc.Col(
        card(
            title,
            html.Div(
                [
                    html.Div(id=value_id, style={"fontSize": "2.15rem", "fontWeight": "800", "color": INK, "lineHeight": 1.1}),
                    html.Div(subtitle, style={"marginTop": "10px", "color": MUTED, "fontSize": "0.93rem"}),
                ]
            ),
            subtle=True,
        ),
        md=3,
        sm=6,
        xs=12,
        className="mb-4",
    )


def summarize_species_title(data: pd.DataFrame) -> str:
    if data.empty:
        return "No species remain after the current filters"
    counts = data["species_short"].astype("object").value_counts()
    if len(counts) == 1:
        return f"{counts.index[0]} is the only visible species"
    return f"{counts.index[0]} leads the visible population"


def build_species_count_figure(data: pd.DataFrame) -> go.Figure:
    counts = (
        data["species_short"]
        .value_counts()
        .rename_axis("species_short")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    counts = counts[counts["count"] > 0].copy()
    counts["species_short"] = counts["species_short"].astype(str)
    fig = px.bar(
        counts,
        x="species_short",
        y="count",
        color="species_short",
        color_discrete_map=SPECIES_COLORS,
        text="count",
        title=summarize_species_title(data),
        labels={"species_short": "Species", "count": "Penguins"},
    )
    fig.update_traces(
        marker_line_color=SURFACE,
        marker_line_width=2,
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Penguins: %{y:,}<extra></extra>",
    )
    fig.update_layout(showlegend=False, xaxis_title=None)
    return color_hoverlabels(style_figure(fig, 350))


def build_island_species_figure(data: pd.DataFrame) -> go.Figure:
    counts = (
        data.groupby(["island", "species_short"], observed=False)
        .size()
        .reset_index(name="count")
    )
    island_totals = counts.groupby("island")["count"].transform("sum")
    counts["share"] = np.where(island_totals > 0, counts["count"] / island_totals, 0)
    fig = px.bar(
        counts,
        x="island",
        y="count",
        color="species_short",
        color_discrete_map=SPECIES_COLORS,
        category_orders={"species_short": SPECIES_ORDER},
        custom_data=["share"],
        text="count",
        title="Island residency sharply separates species",
        labels={"count": "Penguins", "island": "Island", "species_short": "Species"},
    )
    fig.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate="<b>%{fullData.name}</b><br>Island: %{x}<br>Count: %{y}<br>Share: %{customdata[0]:.1%}<extra></extra>",
    )
    fig.update_layout(barmode="stack")
    return color_hoverlabels(style_figure(fig, 350))


def build_measurement_scatter(data: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        data,
        x="flipper_length_mm",
        y="bill_length_mm",
        color="species_short",
        color_discrete_map=SPECIES_COLORS,
        category_orders={"species_short": SPECIES_ORDER},
        trendline="ols",
        title="Body shape clusters stay visually distinct",
        labels={
            "flipper_length_mm": "Flipper Length (mm)",
            "bill_length_mm": "Bill Length (mm)",
            "species_short": "Species",
        },
        custom_data=["species", "sex", "bill_depth_mm", "body_mass_g", "island"],
    )
    fig.update_traces(
        marker=dict(size=12, opacity=0.86, line=dict(width=0.8, color="white")),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Sex: %{customdata[1]}<br>"
            "Island: %{customdata[4]}<br>"
            "Flipper Length: %{x:.0f} mm<br>"
            "Bill Length: %{y:.1f} mm<br>"
            "Bill Depth: %{customdata[2]:.1f} mm<br>"
            "Body Mass: %{customdata[3]:.0f} g<extra></extra>"
        ),
    )
    for trace in fig.data:
        mode = getattr(trace, "mode", "")
        if mode == "lines":
            trace.line = dict(width=2.2, dash="solid")
            trace.hovertemplate = f"<b>{trace.name}</b><br>Group-wise regression line<extra></extra>"
    return color_hoverlabels(style_figure(fig, 490))


def build_sex_difference_figure(data: pd.DataFrame) -> go.Figure:
    features = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
    feature_labels = {
        "bill_length_mm": "Bill Length (mm)",
        "bill_depth_mm": "Bill Depth (mm)",
        "flipper_length_mm": "Flipper Length (mm)",
        "body_mass_g": "Body Mass (g)",
    }
    facet_order = [feature_labels[key] for key in features]
    melted = (
        data[data["sex"].isin(["MALE", "FEMALE"])]
        .melt(
            id_vars=["species_short", "sex"],
            value_vars=features,
            var_name="feature",
            value_name="value",
        )
    )
    melted["feature"] = melted["feature"].map(feature_labels)
    melted["feature"] = pd.Categorical(
        melted["feature"],
        categories=facet_order,
        ordered=True,
    )
    summary = (
        melted.groupby(["species_short", "sex", "feature"], observed=False)["value"]
        .mean()
        .reset_index()
    )
    fig = px.bar(
        summary,
        x="value",
        y="species_short",
        color="sex",
        facet_col="feature",
        barmode="group",
        color_discrete_map={
            "MALE": "#0D47A5",
            "FEMALE": "#F3CFE0",
        },
        category_orders={"feature": facet_order, "species_short": SPECIES_ORDER, "sex": ["MALE", "FEMALE"]},
        title="Average Measurements by Sex Across Features",
        labels={"species_short": "Species", "value": "Average", "sex": "Sex"},
    )
    fig.update_xaxes(matches=None)
    fig.update_traces(
        cliponaxis=False,
        texttemplate="%{x:.1f}",
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Sex: %{fullData.name}<br>"
            "Average Value: %{x:.1f}<extra></extra>"
        ),
    )
    facet_max = {
        feature_name: summary.loc[summary["feature"] == feature_name, "value"].max()
        for feature_name in facet_order
    }
    
    fig.for_each_annotation(lambda ann: ann.update(text=f"<b>{ann.text.split('=')[-1]}</b>"))
    for i, feature_name in enumerate(facet_order, start=1):
        axis_name = "xaxis" if i == 1 else f"xaxis{i}"
        axis_max = facet_max.get(feature_name, 0)

        fig.layout[axis_name].update(
            range=[0, axis_max * 1.1],  
            dtick=axis_max / 4,
        )
    fig.update_layout(title_x=0)
    return color_hoverlabels(style_figure(fig, 520))


def build_body_mass_violin(data: pd.DataFrame) -> go.Figure:
    fig = px.violin(
        data,
        x="species_short",
        y="body_mass_g",
        color="species_short",
        color_discrete_map=SPECIES_COLORS,
        category_orders={"species_short": SPECIES_ORDER},
        box=True,
        points="all",
        title="Gentoo mass sits in a clearly heavier band",
        labels={"species_short": "Species", "body_mass_g": "Body Mass (g)"},
    )
    fig.update_traces(
        meanline_visible=True,
        pointpos=-1.00,
        jitter=0.16,
        hovertemplate="<b>%{x}</b><br>Body Mass: %{y:.0f} g<extra></extra>",
    )
    fig.update_layout(showlegend=False)
    return color_hoverlabels(style_figure(fig, 390))


def build_bill_profile_figure(data: pd.DataFrame) -> go.Figure:
    grouped = (
        data.groupby("species_short", observed=False)[["bill_length_mm", "bill_depth_mm"]]
        .mean()
        .reset_index()
    )
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.12,
        subplot_titles=("Bill Length (mm)", "Bill Depth (mm)"),
    )
    fig.add_trace(
        go.Bar(
            x=-grouped["bill_length_mm"],
            y=grouped["species_short"],
            orientation="h",
            marker=dict(color=[SPECIES_COLORS[item] for item in grouped["species_short"]]),
            text=[f"{value:.1f}" for value in grouped["bill_length_mm"]],
            cliponaxis=False,
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Bill Length: %{text} mm<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=grouped["bill_depth_mm"],
            y=grouped["species_short"],
            orientation="h",
            marker=dict(color=[SPECIES_COLORS[item] for item in grouped["species_short"]]),
            text=[f"{value:.1f}" for value in grouped["bill_depth_mm"]],
            cliponaxis=False,
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Bill Depth: %{text} mm<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(showticklabels=False)
    max_left = grouped["bill_length_mm"].max() if not grouped.empty else 0
    max_right = grouped["bill_depth_mm"].max() if not grouped.empty else 0
    fig.update_xaxes(
        tickvals=[-60, -40, -20, 0],
        ticktext=["60", "40", "20", "0"],
        range=[-(max_left * 1.22 if max_left else 1), 0],
        row=1,
        col=1,
    )
    fig.update_xaxes(range=[0, max_right * 1.22 if max_right else 1], row=1, col=2)
    for species in grouped["species_short"]:
        fig.add_annotation(
            x=0.5,
            y=species,
            xref="paper",
            yref="y",
            text=f"<b>{species}</b>",
            showarrow=False,
            font=dict(size=13, color=SPECIES_COLORS[species]),
        )
    fig.update_layout(title="Chinstrap bills stretch longest and deepest")
    return color_hoverlabels(style_figure(fig, 390))


def build_ecology_figure(data: pd.DataFrame) -> go.Figure:
    eco = data.dropna(subset=["delta_13_c", "delta_15_n"]).copy()
    fig = px.scatter(
        eco,
        x="delta_13_c",
        y="delta_15_n",
        color="species_short",
        color_discrete_map=SPECIES_COLORS,
        category_orders={"species_short": SPECIES_ORDER},
        title="Diet signatures hint at ecological separation",
        labels={"delta_13_c": "Delta 13 C", "delta_15_n": "Delta 15 N", "species_short": "Species"},
        custom_data=["island", "sex"],
    )
    fig.update_traces(
        marker=dict(size=10, opacity=0.9, line=dict(width=0.8, color="white")),
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Island: %{customdata[0]}<br>"
            "Sex: %{customdata[1]}<br>"
            "Delta 13 C: %{x:.2f}<br>"
            "Delta 15 N: %{y:.2f}<extra></extra>"
        ),
    )
    return color_hoverlabels(style_figure(fig, 320))


def prepare_ml_objects() -> dict:
    features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "delta_13_c", "delta_15_n"]
    X = df[features].copy().fillna(df[features].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X_scaled)

    ml_df = df.copy()
    ml_df["PC1"] = X_pca[:, 0]
    ml_df["PC2"] = X_pca[:, 1]
    ml_df["TSNE1"] = X_tsne[:, 0]
    ml_df["TSNE2"] = X_tsne[:, 1]
    ml_df["cluster"] = [f"Cluster {cluster + 1}" for cluster in clusters]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["species_short"])
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=["PC1", "PC2"],
    ).reset_index(names="feature")
    loadings["feature_label"] = loadings["feature"].map(
        {
            "bill_length_mm": "Bill Length",
            "bill_depth_mm": "Bill Depth",
            "flipper_length_mm": "Flipper Length",
            "body_mass_g": "Body Mass",
            "delta_13_c": "Delta 13 C",
            "delta_15_n": "Delta 15 N",
        }
    )

    return {
        "ml_df": ml_df,
        "accuracy": accuracy,
        "pc1": pca.explained_variance_ratio_[0],
        "pc2": pca.explained_variance_ratio_[1],
        "matrix": matrix,
        "labels": list(label_encoder.classes_),
        "loadings": loadings,
    }


ml = prepare_ml_objects()


def build_pca_figure(color_mode: str, species_values: list[str] | None, island_values: list[str] | None) -> go.Figure:
    ml_view = ml["ml_df"].copy()
    if species_values:
        ml_view = ml_view[ml_view["species_short"].isin(species_values)]
    if island_values:
        ml_view = ml_view[ml_view["island"].isin(island_values)]
    color_column = "species_short" if color_mode == "species" else "cluster"
    color_map = SPECIES_COLORS if color_mode == "species" else CLUSTER_COLORS
    if ml_view.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No penguins match the current filters in PCA space.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16, color=MUTED),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title="Two principal components preserve the species story")
        return style_figure(fig, 500)
    fig = px.scatter(
        ml_view,
        x="PC1",
        y="PC2",
        color=color_column,
        color_discrete_map=color_map,
        title="Two principal components preserve the species story",
        labels={
            "PC1": f"PC1 ({ml['pc1'] * 100:.1f}% variance)",
            "PC2": f"PC2 ({ml['pc2'] * 100:.1f}% variance)",
            "species_short": "Species",
            "cluster": "Cluster",
        },
        custom_data=["species", "island", "sex", "body_mass_g", "bill_length_mm"],
    )
    fig.update_traces(
        marker=dict(size=13, opacity=0.9, line=dict(width=0.8, color="white")),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Island: %{customdata[1]}<br>"
            "Sex: %{customdata[2]}<br>"
            "PC1: %{x:.2f}<br>"
            "PC2: %{y:.2f}<br>"
            "Body Mass: %{customdata[3]:.0f} g<br>"
            "Bill Length: %{customdata[4]:.1f} mm<extra></extra>"
        ),
    )
    return color_hoverlabels(style_figure(fig, 500))


def build_loadings_figure(selected_component: str | None = None) -> go.Figure:
    loadings = ml["loadings"].copy()
    top_features = (
        loadings.assign(strength=loadings["PC1"].abs())
        .sort_values("strength", ascending=False)
        .head(2)["feature_label"]
        .tolist()
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=loadings["PC1"],
            y=loadings["feature_label"],
            name="PC1",
            orientation="h",
            marker_color=ACCENT,
            text=[f"{value:.2f}{' *' if label in top_features else ''}" for value, label in zip(loadings["PC1"], loadings["feature_label"])],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>PC1 Loading: %{x:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=loadings["PC2"],
            y=loadings["feature_label"],
            name="PC2",
            orientation="h",
            marker_color="#7FD1C6",
            text=[f"{value:.2f}" for value in loadings["PC2"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>PC2 Loading: %{x:.3f}<extra></extra>",
        )
    )
    if selected_component in {"PC1", "PC2"}:
        for trace in fig.data:
            if trace.name == selected_component:
                trace.marker.line = dict(color=INK, width=1.6)
                trace.opacity = 1.0
            else:
                trace.opacity = 0.22
    else:
        for trace in fig.data:
            trace.opacity = 0.95
    lower = min(loadings["PC1"].min(), loadings["PC2"].min(), 0) if not loadings.empty else 0
    upper = max(loadings["PC1"].max(), loadings["PC2"].max(), 0) if not loadings.empty else 0
    fig.update_layout(
        barmode="group",
        title="Flipper length and body mass dominate the first axis",
        xaxis_title="Loading",
        yaxis_title=None,
    )
    fig.update_xaxes(range=[lower * 1.18 if lower else -0.1, upper * 1.18 if upper else 0.1])
    fig.update_yaxes(categoryorder="array", categoryarray=list(loadings["feature_label"])[::-1])
    return color_hoverlabels(style_figure(fig, 420))


def build_confusion_matrix_figure() -> go.Figure:
    fig = px.imshow(
        ml["matrix"],
        x=ml["labels"],
        y=ml["labels"],
        text_auto=True,
        color_continuous_scale=[[0.0, "#E7F5F3"], [0.4, "#7AD3C4"], [1.0, ACCENT]],
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        title="Predictions align almost perfectly with the real labels",
    )
    fig.update_traces(hovertemplate="<b>Actual: %{y}</b><br>Predicted: %{x}<br>Count: %{z}<extra></extra>")
    fig.update_layout(coloraxis_colorbar=dict(outlinewidth=0))
    return style_figure(fig, 360)


def build_tsne_figure(species_values: list[str] | None, island_values: list[str] | None) -> go.Figure:
    ml_view = ml["ml_df"].copy()
    if species_values:
        ml_view = ml_view[ml_view["species_short"].isin(species_values)]
    if island_values:
        ml_view = ml_view[ml_view["island"].isin(island_values)]
    if ml_view.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No penguins match the current filters in t-SNE space.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16, color=MUTED),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title="t-SNE offers a compact nonlinear view of the same separation")
        return style_figure(fig, 500)
    fig = px.scatter(
        ml_view,
        x="TSNE1",
        y="TSNE2",
        color="species_short",
        color_discrete_map=SPECIES_COLORS,
        category_orders={"species_short": SPECIES_ORDER},
        title="t-SNE offers a compact nonlinear view of the same separation",
        labels={"species_short": "Species", "TSNE1": "t-SNE 1", "TSNE2": "t-SNE 2"},
        custom_data=["species", "island", "sex"],
    )
    fig.update_traces(
        marker=dict(size=10, opacity=0.9, line=dict(width=0.8, color="white")),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Island: %{customdata[1]}<br>"
            "Sex: %{customdata[2]}<br>"
            "t-SNE 1: %{x:.2f}<br>"
            "t-SNE 2: %{y:.2f}<extra></extra>"
        ),
    )
    return color_hoverlabels(style_figure(fig, 500))


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
server = app.server
app.title = "Palmer Penguins Story Dashboard"

app.layout = html.Div(
    [
        dbc.Container(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "Penguin Intelligence Dashboard",
                                    style={
                                        "display": "inline-block",
                                        "padding": "8px 14px",
                                        "borderRadius": "999px",
                                        "background": ACCENT_SOFT,
                                        "color": ACCENT,
                                        "fontWeight": "700",
                                        "letterSpacing": "0.08em",
                                        "textTransform": "uppercase",
                                        "fontSize": "0.76rem",
                                        "marginBottom": "18px",
                                    },
                                ),
                                html.H1(
                                    "Palmer penguins, retold as a product-grade data story.",
                                    style={
                                        "fontFamily": TITLE_FONT,
                                        "fontSize": "3.25rem",
                                        "lineHeight": 1.08,
                                        "margin": "0 0 14px 0",
                                        "color": INK,
                                        "maxWidth": "780px",
                                    },
                                ),
                                html.P(
                                    "Explore how species differ across islands, body measurements, and machine learning space without losing the familiar penguin palette from your notebooks.",
                                    style={
                                        "color": MUTED,
                                        "fontSize": "1.06rem",
                                        "maxWidth": "760px",
                                        "margin": 0,
                                        "lineHeight": 1.65,
                                    },
                                ),
                            ],
                            style={"flex": "1 1 0"},
                        ),
                        html.Div(
                            [
                                html.Img(
                                    src=logo_src,
                                    style={
                                        "width": "170px",
                                        "height": "170px",
                                        "objectFit": "contain",
                                        "filter": "drop-shadow(0 18px 30px rgba(17, 37, 49, 0.10))",
                                    },
                                )
                            ],
                            style={
                                "width": "220px",
                                "display": "flex",
                                "justifyContent": "center",
                                "alignItems": "center",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "24px",
                        "alignItems": "center",
                        "justifyContent": "space-between",
                        "padding": "36px 0 26px 0",
                        "flexWrap": "wrap",
                    },
                ),
                card(
                    "Filters",
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Species", style={"fontWeight": "700", "marginBottom": "10px", "color": INK}),
                                    dcc.Dropdown(
                                        id="species-filter",
                                        options=[{"label": species, "value": species} for species in SPECIES_ORDER],
                                        value=SPECIES_ORDER,
                                        multi=True,
                                        clearable=False,
                                    ),
                                ],
                                md=6,
                                xs=12,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Island", style={"fontWeight": "700", "marginBottom": "10px", "color": INK}),
                                    dcc.Dropdown(
                                        id="island-filter",
                                        options=[{"label": island, "value": island} for island in sorted(df["island"].dropna().unique())],
                                        value=sorted(df["island"].dropna().unique()),
                                        multi=True,
                                        clearable=False,
                                    ),
                                ],
                                md=6,
                                xs=12,
                            ),
                        ],
                        className="g-4",
                    ),
                    extra_style={"marginBottom": "24px"},
                ),
                dbc.Tabs(
                    [
                        dbc.Tab(label="Exploration", tab_id="tab-exploration", label_style={"fontWeight": "700"}),
                        dbc.Tab(label="Machine Learning", tab_id="tab-ml", label_style={"fontWeight": "700"}),
                    ],
                    active_tab="tab-exploration",
                    id="main-tabs",
                ),
                html.Div(id="tab-content"),
            ],
            fluid=True,
            style={"maxWidth": "1380px", "padding": "0 26px 36px 26px"},
        )
    ],
    style={
        "minHeight": "100vh",
        "background": "linear-gradient(180deg, #EEF7F4 0%, #F6F9F8 18%, #F5F8F7 100%)",
        "fontFamily": BODY_FONT,
    },
)


def exploration_layout() -> html.Div:
    return html.Div(
        [
            dbc.Row(
                [
                    kpi_card("Total Penguins", "kpi-total", "Filter-aware population size"),
                    kpi_card("Avg Body Mass (g)", "kpi-mass", "Average weight across visible penguins"),
                    kpi_card("Avg Flipper Length (mm)", "kpi-flipper", "Mean flipper span of the current cohort"),
                    kpi_card("# Species", "kpi-species", "Distinct species visible in the filter state"),
                ],
                className="g-4 mt-2",
            ),
            html.Div(
                "Quick overview of population and key physical characteristics",
                style={"marginTop": "-4px", "marginBottom": "24px", "color": MUTED, "fontSize": "0.96rem"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            "Adelie is the main population anchor",
                            "Species counts are sorted from most to least common.",
                            "species-count-fig",
                            "Adelie is the most common species.",
                            350,
                            note_id="species-count-note",
                            title_id="species-count-title",
                        ),
                        md=6,
                        className="mb-4",
                    ),
                    dbc.Col(
                        graph_card(
                            "Islands reveal strong occupancy patterns",
                            "Counts are stacked so composition is visible at a glance.",
                            "island-species-fig",
                            "Species occupy distinct islands, with Gentoo primarily found on Biscoe.",
                            350,
                            note_id="island-species-note",
                        ),
                        md=6,
                        className="mb-4",
                    ),
                ],
                className="g-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            "Morphology separates species without forcing the view",
                            "The hero scatter compares flipper and bill length across the full cohort.",
                            "measurement-scatter-fig",
                            "Species form clearly separable clusters based on physical measurements.",
                            490,
                            note_id="measurement-scatter-note",
                        ),
                        md=12,
                        className="mb-4",
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            "Average Measurements by Sex Across Features",
                            "A faceted comparison makes the sex gap readable across bill, flipper, and body mass together.",
                            "sex-difference-fig",
                            "Males consistently exhibit larger measurements across nearly every feature and species.",
                            520,
                            note_id="sex-difference-note",
                        ),
                        md=12,
                        className="mb-4",
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            "Gentoo mass occupies a higher distribution",
                            "Violin shapes show the spread, density, and outliers by species.",
                            "body-mass-violin-fig",
                            "Gentoo penguins show a distinctly higher body mass distribution.",
                            390,
                            note_id="body-mass-note",
                        ),
                        md=6,
                        className="mb-4",
                    ),
                    dbc.Col(
                        graph_card(
                            "Bill profiles show Chinstrap at the extreme",
                            "Average bill length and depth are mirrored around the species labels.",
                            "bill-profile-fig",
                            "Chinstrap combines long and deep bills more strongly than the others.",
                            390,
                            note_id="bill-profile-note",
                        ),
                        md=6,
                        className="mb-4",
                    ),
                ],
                className="g-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            "Ecology adds a quieter but useful extra layer",
                            "This optional panel uses isotopic signatures to hint at foraging and dietary differences.",
                            "ecology-fig",
                            "Isotopic signatures suggest ecological and dietary differences between species.",
                            340,
                            note_id="ecology-note",
                        ),
                        md=12,
                        className="mb-2",
                    )
                ]
            ),
        ],
        style={"padding": "18px 0 4px 0"},
    )


def ml_layout() -> html.Div:
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(card("Model Accuracy", html.Div([html.Div(f"{ml['accuracy'] * 100:.1f}%", style={"fontSize": "2.15rem", "fontWeight": "800", "color": INK}), html.Div("Random forest classification on the six-feature morphology and isotope set", style={"marginTop": "10px", "color": MUTED, "fontSize": "0.93rem"})]), subtle=True), md=3, sm=6, xs=12, className="mb-4"),
                    dbc.Col(card("PC1 Variance", html.Div([html.Div(f"{ml['pc1'] * 100:.1f}%", style={"fontSize": "2.15rem", "fontWeight": "800", "color": INK}), html.Div("The main separation axis explains most of the visible structure", style={"marginTop": "10px", "color": MUTED, "fontSize": "0.93rem"})]), subtle=True), md=3, sm=6, xs=12, className="mb-4"),
                    dbc.Col(card("PC2 Variance", html.Div([html.Div(f"{ml['pc2'] * 100:.1f}%", style={"fontSize": "2.15rem", "fontWeight": "800", "color": INK}), html.Div("A second axis still contributes meaningful species structure", style={"marginTop": "10px", "color": MUTED, "fontSize": "0.93rem"})]), subtle=True), md=3, sm=6, xs=12, className="mb-4"),
                    dbc.Col(card("# Clusters", html.Div([html.Div("3", style={"fontSize": "2.15rem", "fontWeight": "800", "color": INK}), html.Div("Set to the three known species for interpretability", style={"marginTop": "10px", "color": MUTED, "fontSize": "0.93rem"})]), subtle=True), md=3, sm=6, xs=12, className="mb-4"),
                ],
                className="g-4 mt-2",
            ),
            html.Div(
                "Model performance and dimensionality reduction summary",
                style={"marginTop": "-4px", "marginBottom": "24px", "color": MUTED, "fontSize": "0.96rem"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        card(
                            "",
                            html.Div(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.H3("Species separation stays crisp in PCA space", style={"fontFamily": TITLE_FONT, "fontWeight": "700", "fontSize": "1.6rem", "margin": "0 0 8px 0", "color": INK}),
                                                    html.P("Switch between known species labels and unsupervised clusters to compare the story.", style={"margin": 0, "color": MUTED}),
                                                ],
                                                md=8,
                                            ),
                                            dbc.Col(
                                                dcc.RadioItems(
                                                    id="pca-color-mode",
                                                    options=[
                                                        {"label": "Color by Species", "value": "species"},
                                                        {"label": "Color by Cluster", "value": "cluster"},
                                                    ],
                                                    value="species",
                                                    inline=True,
                                                    style={"textAlign": "right", "color": INK},
                                                    inputStyle={"marginRight": "6px", "marginLeft": "12px"},
                                                ),
                                                md=4,
                                                style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end"},
                                            ),
                                        ],
                                        className="g-3",
                                    ),
                                    dcc.Graph(id="pca-fig", config={"displayModeBar": False}, style={"height": "500px"}),
                                    html.Div(id="pca-note", children=annotation_block("Species separation is captured in just two principal components.")),

                                ]
                            ),
                        ),
                        md=12,
                        className="mb-4",
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            "Feature loadings explain why PCA works so well",
                            "Click a PC1 or PC2 bar to spotlight that component across every feature.",
                            "loadings-fig",
                            "Flipper length and body mass drive the primary separation between species.",
                            420,
                        ),
                        md=7,
                        className="mb-4",
                    ),
                    dbc.Col(
                        graph_card(
                            "Validation confirms the visual separation",
                            "The model rarely confuses one species for another on the held-out test split.",
                            "confusion-matrix-fig",
                            "Model evaluated on full dataset (not affected by filters).",
                            420,
                        ),
                        md=5,
                        className="mb-4",
                    ),
                ],
                className="g-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        graph_card(
                            "t-SNE keeps the clusters compact",
                            "This nonlinear embedding acts as a full-width companion view to PCA.",
                            "tsne-fig",
                            "Even with nonlinear embedding, the three species remain cleanly separated.",
                            500,
                        ),
                        md=12,
                        className="mb-4",
                    )
                ]
            ),
        ],
        style={"padding": "18px 0 4px 0"},
    )


@app.callback(Output("tab-content", "children"), Input("main-tabs", "active_tab"))
def render_tab(tab: str):
    return exploration_layout() if tab == "tab-exploration" else ml_layout()


@app.callback(
    Output("kpi-total", "children"),
    Output("kpi-mass", "children"),
    Output("kpi-flipper", "children"),
    Output("kpi-species", "children"),
    Output("species-count-fig", "figure"),
    Output("island-species-fig", "figure"),
    Output("measurement-scatter-fig", "figure"),
    Output("sex-difference-fig", "figure"),
    Output("body-mass-violin-fig", "figure"),
    Output("bill-profile-fig", "figure"),
    Output("ecology-fig", "figure"),
    Output("species-count-title", "children"),
    Output("species-count-note", "children"),
    Output("island-species-note", "children"),
    Output("measurement-scatter-note", "children"),
    Output("sex-difference-note", "children"),
    Output("body-mass-note", "children"),
    Output("bill-profile-note", "children"),
    Output("ecology-note", "children"),
    Input("species-filter", "value"),
    Input("island-filter", "value"),
)
def update_exploration(species_values: list[str], island_values: list[str]):
    data = filtered_df(species_values, island_values)
    total_penguins = f"{len(data):,}"
    avg_mass = f"{data['body_mass_g'].mean():,.0f}" if len(data) else "0"
    avg_flipper = f"{data['flipper_length_mm'].mean():.1f}" if len(data) else "0.0"
    species_count = f"{data['species_short'].nunique()}"

    return (
        total_penguins,
        avg_mass,
        avg_flipper,
        species_count,
        build_species_count_figure(data),
        build_island_species_figure(data),
        build_measurement_scatter(data),
        build_sex_difference_figure(data),
        build_body_mass_violin(data),
        build_bill_profile_figure(data),
        build_ecology_figure(data),
        summarize_species_title(data),
        annotation_container(summarize_species_count(data)),
        annotation_container(summarize_island_mix(data)),
        annotation_container(summarize_scatter(data)),
        annotation_container(summarize_sex_difference(data)),
        annotation_container(summarize_body_mass(data)),
        annotation_container(summarize_bill_profile(data)),
        annotation_container(summarize_ecology(data)),
    )


@app.callback(
    Output("pca-fig", "figure"),
    Output("loadings-fig", "figure"),
    Output("confusion-matrix-fig", "figure"),
    Output("tsne-fig", "figure"),
    Output("pca-note", "children"),
    Input("pca-color-mode", "value"),
    Input("loadings-fig", "clickData"),
    Input("species-filter", "value"),
    Input("island-filter", "value"),
)
def update_ml(color_mode: str, loadings_click_data, species_values: list[str], island_values: list[str]):
    selected_component = None
    if loadings_click_data and loadings_click_data.get("points"):
        curve_number = loadings_click_data["points"][0].get("curveNumber")
        if curve_number == 0:
            selected_component = "PC1"
        elif curve_number == 1:
            selected_component = "PC2"
    return (
        build_pca_figure(color_mode, species_values, island_values),
        build_loadings_figure(selected_component),
        build_confusion_matrix_figure(),
        build_tsne_figure(species_values, island_values),
        annotation_container(summarize_pca_subset(filtered_df(species_values, island_values))),
    )


if __name__ == "__main__":
    app.run(debug=False, port=8052)
