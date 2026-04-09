"""
Palmer Penguins One-Click Pipeline + Interactive Dash Dashboard
Run:  python penguin_dashboard.py
Then open http://127.0.0.1:8050 in your browser.

Pipeline stages (auto-executed on startup):
  1. Load 
  2. Preprocess  
  3. Visualizations
  4. ML
  5. Dashboard
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from palmerpenguins import load_penguins

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

print("All libraries imported.")

# STAGE 1: LOAD
print("[Stage 1] Loading Palmer Penguins dataset")
preprocessed_path = Path("nam_branch/penguins_preprocessed.csv")

if preprocessed_path.exists():
    raw_penguins = pd.read_csv(preprocessed_path)
    df_raw = pd.DataFrame({
        "species": raw_penguins["Species"].replace({
            "Adelie Penguin (Pygoscelis adeliae)": "Adelie",
            "Gentoo penguin (Pygoscelis papua)": "Gentoo",
            "Chinstrap penguin (Pygoscelis antarctica)": "Chinstrap",
        }),
        "island": raw_penguins["Island"],
        "bill_length_mm": raw_penguins["Culmen Length (mm)"],
        "bill_depth_mm": raw_penguins["Culmen Depth (mm)"],
        "flipper_length_mm": raw_penguins["Flipper Length (mm)"],
        "body_mass_g": raw_penguins["Body Mass (g)"],
        "sex": raw_penguins["Sex"].str.lower(),
        "delta_15": raw_penguins["Delta 15 N (o/oo)"],
        "delta_13": raw_penguins["Delta 13 C (o/oo)"],
        "year": pd.to_datetime(raw_penguins["Date Egg"]).dt.year,
    })
else:
    df_raw = load_penguins()
    df_raw["delta_15"] = np.nan
    df_raw["delta_13"] = np.nan

print(f"   Raw shape: {df_raw.shape}  |  Missing values:\n{df_raw.isnull().sum().to_string()}")

# STAGE 2: PRE-PROCESSING
print("[Stage 2] Pre-processing")

numeric_cols = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

# Drop the 2 rows where ALL numeric measurements are missing
df = df_raw.dropna(subset=numeric_cols, how="all").copy()
print(f"Dropped {len(df_raw) - len(df)} rows with all-NA measurements.")

# Force numeric columns to be numeric, then fill remaining gaps.
# If a column is entirely missing (for example isotope fields in the fallback dataset),
# use 0 so downstream ML steps always receive finite values.
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    fill_value = df[col].mean()
    if pd.isna(fill_value):
        fill_value = 0.0
    df[col] = df[col].fillna(fill_value)

# Fill sex NA with mode
df["sex"] = df["sex"].fillna(df["sex"].mode()[0])

print(f"   Clean shape: {df.shape}  |  Remaining NAs: {df.isnull().sum().sum()}")

# Friendly colour palette (matches visualization notebook)
COLORS = {
    "Adelie":   "#FF8C00",
    "Gentoo":   "#9932CC",
    "Chinstrap":"#057076",
}

CLUSTER_COLORS_2 = {"0": "#A0AEC0", "1": "#057076"}
CLUSTER_COLORS_3 = {"0": "#A0AEC0", "1": "#4E79A7", "2": "#057076"}


def display_label(column_name):
    return column_name.replace("_mm", "").replace("_g", "").replace("_", " ").title()


def format_axis_label(column_name):
    custom_labels = {
        "bill_length_mm": "Bill Length",
        "bill_depth_mm": "Bill Depth",
        "flipper_length_mm": "Flipper Length",
        "body_mass_g": "Body Mass",
        "delta_15": "Delta 15",
        "delta_13": "Delta 13",
    }
    return custom_labels.get(column_name, column_name.replace("_", " ").title())

# STAGE 3 VISUALIZATIONS  (Plotly figures stored for the dashboard)
print("\n[Stage 3] Building visualisation figures")

# 3-A  Species distribution bar chart
species_counts = df["species"].value_counts().reset_index()
species_counts.columns = ["species", "count"]

fig_species_bar = px.bar(
    species_counts, x="species", y="count", color="species",
    color_discrete_map=COLORS, text="count",
    title="Species Distribution",
)
fig_species_bar.update_traces(
    textposition="outside", marker_line_color="black", marker_line_width=1
)
fig_species_bar.update_layout(
    showlegend=False, plot_bgcolor="white",
    xaxis_title="Species", yaxis_title="Count",
    bargap=0.35,
    annotations=[dict(
        text="Adelie is the most common species, found across all three islands.",
        xref="paper", yref="paper", x=0.5, y=-0.22,
        showarrow=False, font=dict(size=12, color="#555"),
    )],
    margin=dict(b=80),
)

# 3-B  Species — Island grouped bar
cross = df.groupby(["island", "species"]).size().reset_index(name="count")

fig_island_bar = px.bar(
    cross, x="island", y="count", color="species",
    barmode="group", color_discrete_map=COLORS,
    title="Species vs. Island",
    text="count",
)
fig_island_bar.update_traces(textposition="outside")
fig_island_bar.update_layout(
    plot_bgcolor="white",
    xaxis_title="Island", yaxis_title="Count",
    legend_title="Species",
    annotations=[dict(
        text="Gentoo penguins are exclusive to Biscoe island. Dream hosts Adelie & Chinstrap.",
        xref="paper", yref="paper", x=0.5, y=-0.22,
        showarrow=False, font=dict(size=12, color="#555"),
    )],
    margin=dict(b=80),
)

# 3-C  Flipper length vs Body mass scatter
fig_scatter = px.scatter(
    df,
    x="flipper_length_mm",
    y="body_mass_g",
    color="species",
    color_discrete_map=COLORS,
    trendline="ols",
    title="Flipper Length vs Body Mass",
    labels={
        "flipper_length_mm": "Flipper Length (mm)",
        "body_mass_g": "Body Mass (g)",
    },
    hover_data=["island", "sex", "bill_length_mm", "bill_depth_mm"],
)
fig_scatter.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color="white")))
fig_scatter.update_layout(title_x=0)
fig_violin = px.violin(
    df, y="body_mass_g", x="species", color="species",
    box=True, points=False, color_discrete_map=COLORS,
    title="Body Mass Distribution by Species",
    labels={"body_mass_g": "Body Mass (g)", "species": "Species"},
)
fig_violin.update_layout(showlegend=False)

# 3-E  Correlation heatmap
corr = df[numeric_cols].corr()
corr.index = [format_axis_label(col) for col in corr.index]
corr.columns = [format_axis_label(col) for col in corr.columns]
fig_corr = px.imshow(
    corr, text_auto=".2f", aspect="auto",
    title="Correlation Matrix of Physical Measurements",
    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
)
fig_corr.update_layout(title_x=0.5)

# 3-F  Body mass by island + sex horizontal bar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df_grouped = df.groupby(["island", "species", "sex"])["body_mass_g"].mean().reset_index()
fig_mass_bar = px.bar(
    df_grouped, x="body_mass_g", y="species", color="sex",
    orientation="h", barmode="group", facet_col="island",
    color_discrete_map={"male": "#3B82F6", "female": "#FF92C8"},
    
    title="Average Body Mass by Species, Island, and Sex",
    labels={"body_mass_g": "Avg Body Mass (g)", "species": "Species"},
)
fig_mass_bar.update_layout(title_x=0.5, height=420,
    legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"))

# 3-G  3-D scatter of bill length, flipper length, and body mass
fig_3d = px.scatter_3d(
    df, x="bill_length_mm", y="flipper_length_mm", z="body_mass_g",
    color="species", color_discrete_map=COLORS, opacity=0.8,
    title="3D Morphological Clusters",
    labels={"bill_length_mm": "Bill (mm)", "flipper_length_mm": "Flipper (mm)", "body_mass_g": "Mass (g)"},
)
fig_3d.update_traces(marker=dict(size=4, line=dict(width=1, color="DarkSlateGrey")))
fig_3d.update_layout(title_x=0.5, height=500)

# 3-H  Donut by island
fig_donut = px.pie(
    df, names="species", facet_col="island",
    color="species", color_discrete_map=COLORS,
    hole=0.4, title="Species Composition per Island",
)
fig_donut.update_layout(title_x=0.5)

print("All visualisation figures ready.")

# STAGE 4 MACHINE LEARNING
print("\n[Stage 4] Running Machine Learning")

features = numeric_cols.copy()
X = df[features].copy()
X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow analysis and fixed clustering decisions
inertia_df = pd.DataFrame({
    "k": list(range(1, 7)),
    "inertia": [KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_scaled).inertia_ for k in range(1, 7)],
})

kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init="auto")
df["cluster_k2"] = kmeans_2.fit_predict(X_scaled).astype(str)
df["cluster_k3"] = kmeans_3.fit_predict(X_scaled).astype(str)
print("   K-Means: fixed models fitted for k=2 and k=3.")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]
var1, var2 = pca.explained_variance_ratio_ * 100
print(f"   PCA: PC1={var1:.1f}%  PC2={var2:.1f}% variance explained.")

fig_pca_cluster = px.scatter(
    df, x="PCA1", y="PCA2", color="cluster_k3",
    symbol="species",
    title=f"K=3 Clusters on PCA Space (PC1={var1:.1f}%, PC2={var2:.1f}%)",
    labels={"PCA1": f"PC1 ({var1:.1f}%)", "PCA2": f"PC2 ({var2:.1f}%)"},
    hover_data=["species", "island"],
    color_discrete_map=CLUSTER_COLORS_3,
)
fig_pca_cluster.update_layout(title_x=0.5)

fig_elbow = px.line(
    inertia_df,
    x="k",
    y="inertia",
    markers=True,
    title="The elbow softens after k=2",
    labels={"k": "Number Of Clusters", "inertia": "Inertia"},
)
fig_elbow.add_annotation(
    x=2,
    y=float(inert2ia_df.loc[inertia_df["k"] == 2, "inertia"].iloc[0]),
    text="Elbow suggests k=2, but biology points to 3 species.",
    showarrow=True,
    arrowhead=2,
    ax=40,
    ay=-50,
    bgcolor="white"
)

fig_pca_species = px.scatter(
    df, x="PCA1", y="PCA2", color="species",
    color_discrete_map=COLORS,
    title=f"True Species on PCA Space",
    labels={"PCA1": f"PC1 ({var1:.1f}%)", "PCA2": f"PC2 ({var2:.1f}%)"},
    hover_data=["island", "sex"],
)
fig_pca_species.update_layout(title_x=0.5)

# 4-C  Random Forest Classifier
le = LabelEncoder()
y_enc = le.fit_transform(df["species"])
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.3, random_state=42, stratify=y_enc
)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"   Random Forest Accuracy: {accuracy*100:.1f}%")

# Feature importances
importances = pd.Series(rf.feature_importances_, index=[format_axis_label(col) for col in features]).sort_values(ascending=True)
fig_feat_imp = px.bar(
    importances, orientation="h",
    title=f"Feature Importances (RF Accuracy: {accuracy*100:.1f}%)",
    labels={"value": "Importance", "index": "Feature"},
    color=importances.values,
    color_continuous_scale="Viridis",
)
fig_feat_imp.update_layout(showlegend=False, coloraxis_showscale=False)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
species_names = le.classes_
fig_cm = px.imshow(
    cm, text_auto=True, x=species_names, y=species_names,
    color_continuous_scale="Blues",
    title="Confusion Matrix (Test Set)",
    labels={"x": "Predicted", "y": "Actual"},
)
fig_cm.update_layout(title_x=0.5)

# Classification report as a DataFrame for display
report_dict = classification_report(y_test, y_pred, target_names=species_names, output_dict=True)
report_df = pd.DataFrame(report_dict).T.round(2).reset_index()
report_df.columns = ["Class"] + list(report_df.columns[1:])

print("ML complete.")

# STAGE 5: DASH DASHBOARD
print("\n[Stage 5] Launching Dash dashboard...")
PLOT_TEMPLATE = "simple_white"
px.defaults.template = PLOT_TEMPLATE
THEME_BG = "#F7F9FC"
CARD_BG = "#FFFFFF"
ACCENT = "#057076"
ACCENT_LIGHT = "#E7F3F4"
TEXT_DARK = "#18323D"
TEXT_MUTED = "#667784"
DIVIDER = "#DCE4EA"
PAGE_STYLE = {
    "backgroundColor": THEME_BG,
    "minHeight": "100vh",
    "fontFamily": "'Segoe UI', system-ui, sans-serif",
}
PAGE_WIDTH = {"maxWidth": "1320px", "margin": "0 auto", "padding": "0 24px"}
CARD_STYLE = {
    "background": CARD_BG,
    "border": f"1px solid {DIVIDER}",
    "borderRadius": "18px",
    "padding": "24px",
    "boxShadow": "0 12px 30px rgba(16, 24, 40, 0.04)",
}
SOFT_CARD_STYLE = {
    **CARD_STYLE,
    "background": "#FBFDFD",
}
INSIGHT_STYLE = {
    "background": ACCENT_LIGHT,
    "border": f"1px solid rgba(5, 112, 118, 0.15)",
    "borderRadius": "16px",
    "padding": "20px",
    "color": TEXT_DARK,
    "lineHeight": "1.7",
}
SECTION_LABEL = {
    "color": ACCENT,
    "fontWeight": "700",
    "letterSpacing": "0.12em",
    "fontSize": "11px",
    "textTransform": "uppercase",
    "marginBottom": "8px",
}
MUTED = {"color": TEXT_MUTED, "fontSize": "13px", "lineHeight": "1.7", "margin": "0"}
CONTROL_LABEL = {"fontWeight": "600", "fontSize": "13px", "color": TEXT_DARK, "marginBottom": "8px"}
def apply_figure_theme(fig, height=None):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Segoe UI, system-ui, sans-serif", color=TEXT_DARK),
        title=dict(font=dict(size=20), x=0),
        margin=dict(l=20, r=20, t=64, b=40),
        hoverlabel=dict(font=dict(family="Segoe UI, system-ui, sans-serif")),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EDF2F7", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EDF2F7", zeroline=False)
    if height is not None:
        fig.update_layout(height=height)
    return fig
def stat_card(title, value, subtitle=""):
    return html.Div([
        html.P(title, style={**MUTED, "fontSize": "12px", "marginBottom": "8px"}),
        html.H3(value, style={
            "color": TEXT_DARK,
            "fontWeight": "800",
            "fontSize": "30px",
            "margin": "0 0 6px 0",
        }),
        html.P(subtitle, style={**MUTED, "fontSize": "11px"}),
    ], style={**CARD_STYLE, "height": "100%"})
def section_header(label, title, subtitle=""):
    return html.Div([
        html.P(label, style=SECTION_LABEL),
        html.H2(title, style={
            "color": TEXT_DARK,
            "fontWeight": "800",
            "fontSize": "36px",
            "margin": "0 0 10px 0",
        }),
        html.P(subtitle, style={**MUTED, "fontSize": "15px", "maxWidth": "780px"}) if subtitle else None,
    ], style={"marginBottom": "24px"})
def graph_card(figure=None, title="", subtitle="", insight="", graph_id=None, height=420):
    graph_component = dcc.Graph(id=graph_id, figure=figure, style={"height": f"{height}px"}) if graph_id else dcc.Graph(figure=figure, style={"height": f"{height}px"})
    children = []
    if title or subtitle:
        children.append(html.Div([
            html.H4(title, style={"color": TEXT_DARK, "fontWeight": "700", "fontSize": "20px", "margin": "0 0 4px 0"}),
            html.P(subtitle, style=MUTED) if subtitle else None,
        ], style={"marginBottom": "16px"}))
    children.append(graph_component)
    if insight:
        children.append(html.Div(insight, style={**INSIGHT_STYLE, "marginTop": "14px"}))
    return html.Div(children, style=CARD_STYLE)
def info_box(title, body):
    return html.Div([
        html.P(title, style=SECTION_LABEL),
        html.P(body, style={**MUTED, "fontSize": "15px", "color": TEXT_DARK}),
    ], style=INSIGHT_STYLE)


def filter_badge(label):
    return html.Div(
        label,
        style={
            "display": "inline-block",
            "padding": "8px 12px",
            "borderRadius": "999px",
            "background": ACCENT_LIGHT,
            "color": ACCENT,
            "fontWeight": "700",
            "fontSize": "12px",
            "marginBottom": "14px",
        },
    )
def largest_species_label():
    return df.groupby("species")["body_mass_g"].mean().idxmax()
def build_species_distribution_figure(selected_species=None):
    fig = px.bar(
        species_counts,
        x="species",
        y="count",
        color="species",
        color_discrete_map=COLORS,
        text="count",
        labels={"species": "Species", "count": "Penguins"},
    )
    fig.update_traces(textposition="outside", marker_line_color="white", marker_line_width=1.5, opacity=1.0)
    fig.update_traces(hovertemplate="<b>%{x}</b><br>Penguins: %{y}<extra></extra>")
    if selected_species:
        fig.for_each_trace(lambda trace: trace.update(opacity=1.0 if trace.name == selected_species else 0.25))
    fig.update_layout(showlegend=False, clickmode="event+select", bargap=0.35)
    return apply_figure_theme(fig, height=340)
def filter_overview_df(selected_species=None, islands=None):
    dff = df.copy()
    if selected_species:
        dff = dff[dff["species"] == selected_species]
    if islands:
        dff = dff[dff["island"].isin(islands)]
    return dff
def build_overview_scatter(dff, selected_species=None):
    title = f"Showing: {selected_species} Penguins" if selected_species else "Showing: All Penguin Species"
    if len(dff) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No penguins match the current drill-down and filters.",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            font=dict(size=16, color=TEXT_MUTED),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title=title)
        return apply_figure_theme(fig, height=520)
    fig = px.scatter(
        dff,
        x="flipper_length_mm",
        y="body_mass_g",
        color="species",
        color_discrete_map=COLORS,
        trendline="ols",
        labels={
            "flipper_length_mm": "Flipper Length (mm)",
            "body_mass_g": "Body Mass (g)",
        },
        hover_data=["island", "sex", "bill_length_mm", "bill_depth_mm"],
        title=title,
    )
    fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color="white")))
    fig.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Island: %{customdata[0]}<br>"
            "Sex: %{customdata[1]}<br>"
            "Flipper length: %{x:.0f} mm<br>"
            "Body mass: %{y:.0f} g<br>"
            "Bill length: %{customdata[2]:.1f} mm<br>"
            "Bill depth: %{customdata[3]:.1f} mm<extra></extra>"
        )
    )
    return apply_figure_theme(fig, height=520)
def build_overview_insight(dff, selected_species=None):
    if len(dff) == 0:
        return info_box("Key takeaway", "No records match the current selection, so there is nothing to compare yet.")
    corr_value = dff["flipper_length_mm"].corr(dff["body_mass_g"])
    selection_label = f"Filter: {selected_species}" if selected_species else "Filter: All species"
    if selected_species:
        body = (
            f"{selected_species} stays internally consistent: longer flippers still map to heavier bodies (r = {corr_value:.2f}), "
            "so the species keeps a clear physical signature even on its own."
        )
    else:
        body = (
            f"Gentoo penguins are visibly larger, while Adelie and Chinstrap occupy lighter regions. "
            f"Across all penguins, flipper length and body mass move together strongly (r = {corr_value:.2f}), making the species separable by measurement alone."
        )
    return html.Div([
        filter_badge(selection_label),
        info_box("Dominant insight", body),
    ])
def build_detail_list(title, items):
    return html.Div([
        html.H5(title, style={"fontWeight": "700", "fontSize": "18px", "marginBottom": "14px", "color": TEXT_DARK}),
        html.Div(items, style={"display": "grid", "gap": "10px"}),
    ], style=CARD_STYLE)
def filtered_explorer_df(species=None, island=None, sex=None):
    dff = df.copy()
    if species:
        dff = dff[dff["species"].isin(species)]
    if island:
        dff = dff[dff["island"].isin(island)]
    if sex:
        dff = dff[dff["sex"].isin(sex)]
    return dff


def build_ml_decision_figure():
    fig = px.scatter(
        df,
        x="PCA1",
        y="PCA2",
        color="cluster_k3",
        symbol="species",
        color_discrete_map=CLUSTER_COLORS_3,
        title="K=3 preserves the biological story",
        labels={"PCA1": f"PC1 ({var1:.1f}%)", "PCA2": f"PC2 ({var2:.1f}%)"},
        hover_data=["species", "island"],
    )
    fig.add_annotation(
        x=0.98,
        y=0.02,
        xref="paper",
        yref="paper",
        text="Why k=3? It separates the three known species most cleanly.",
        showarrow=False,
        align="right",
        bgcolor="white",
        bordercolor=ACCENT,
        borderwidth=1,
        font=dict(color=TEXT_DARK, size=12),
    )
    return apply_figure_theme(fig, 420)


def build_ml_species_alignment_figure(color_mode):
    if color_mode == "cluster_k3":
        fig = px.scatter(
            df,
            x="PCA1",
            y="PCA2",
            color="cluster_k3",
            symbol="species",
            color_discrete_map=CLUSTER_COLORS_3,
            title="K=3 clusters mirror the main species boundaries",
            labels={"PCA1": f"PC1 ({var1:.1f}%)", "PCA2": f"PC2 ({var2:.1f}%)"},
            hover_data=["species", "island"],
        )
    else:
        fig = px.scatter(
            df,
            x="PCA1",
            y="PCA2",
            color="species",
            symbol="island",
            color_discrete_map=COLORS,
            title="True species separate in the same PCA space",
            labels={"PCA1": f"PC1 ({var1:.1f}%)", "PCA2": f"PC2 ({var2:.1f}%)"},
            hover_data=["island", "sex"],
        )
    return apply_figure_theme(fig, 420)


def build_ml_interactive_figure(species=None, islands=None, compare_mode="cluster_k3"):
    dff = df.copy()
    if species:
        dff = dff[dff["species"].isin(species)]
    if islands:
        dff = dff[dff["island"].isin(islands)]

    title_prefix = "K=2 merges species structure" if compare_mode == "cluster_k2" else "K=3 keeps species structure visible"

    if compare_mode == "cluster_k2":
        color_map = CLUSTER_COLORS_2
    else:
        color_map = CLUSTER_COLORS_3

    fig = px.scatter(
        dff,
        x="PCA1",
        y="PCA2",
        color=compare_mode,
        symbol="species",
        color_discrete_map=color_map,
        title=f"{title_prefix} ({len(dff)} penguins shown)",
        labels={"PCA1": f"PC1 ({var1:.1f}%)", "PCA2": f"PC2 ({var2:.1f}%)"},
        hover_data=["species", "island"],
    )
    fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color="white")))
    return apply_figure_theme(fig, 460)
for figure in [
    fig_species_bar,
    fig_island_bar,
    fig_violin,
    fig_corr,
    fig_mass_bar,
    fig_3d,
    fig_donut,
    fig_pca_species,
    fig_pca_cluster,
    fig_feat_imp,
    fig_cm,
    fig_scatter,
]:
    apply_figure_theme(figure, figure.layout.height if figure.layout.height else None)
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Palmer Penguins Dashboard",
    suppress_callback_exceptions=True,
)
total_penguins = len(df)
n_species = df["species"].nunique()
n_islands = df["island"].nunique()
largest_species = largest_species_label()
TAB_STYLE = {
    "fontWeight": "600",
    "padding": "16px 18px",
    "border": "none",
    "color": TEXT_MUTED,
    "background": CARD_BG,
}
TAB_SELECTED_STYLE = {
    "fontWeight": "700",
    "color": ACCENT,
    "border": "none",
    "borderBottom": f"3px solid {ACCENT}",
    "background": CARD_BG,
    "padding": "16px 18px 13px",
}
def hero_panel():
    return html.Div([
        html.Div([
            html.P("Palmer Penguins Analysis", style=SECTION_LABEL),
            html.H1("What distinguishes penguin species?", style={
                "fontSize": "42px",
                "fontWeight": "800",
                "color": TEXT_DARK,
                "margin": "0 0 12px 0",
            }),
            html.P(
                "This dashboard is organized as an analytical story: start with the broad species patterns, test the separation in measurement space, then move into model evidence and free-form exploration.",
                style={**MUTED, "fontSize": "16px", "maxWidth": "860px"},
            ),
        ], style={**CARD_STYLE, "margin": "24px 0"}),
    ])
app.layout = html.Div(style=PAGE_STYLE, children=[
    html.Div(style=PAGE_WIDTH, children=[hero_panel()]),
    html.Div(style=PAGE_WIDTH, children=[
        html.Div([
            dcc.Tabs(
                id="main-tabs",
                value="tab-overview",
                parent_style={"overflowX": "auto"},
                style={"border": "none"},
                colors={"border": CARD_BG, "primary": ACCENT, "background": CARD_BG},
                children=[
                    dcc.Tab(label="Overview", value="tab-overview", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    dcc.Tab(label="Comparisons", value="tab-compare", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    dcc.Tab(label="Machine Learning", value="tab-ml", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    dcc.Tab(label="Explorer", value="tab-explorer", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                ],
            ),
        ], style={**CARD_STYLE, "padding": "0 18px", "marginBottom": "24px"}),
        html.Div(id="tab-content", style={"paddingBottom": "32px"}),
    ]),
])
@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "tab-overview":
        return html.Div([
            dcc.Store(id="overview-selected-species"),
            section_header(
                "Overview",
                "Guided story",
                "Follow the evidence from species counts into measurement space. Clicking a species bar will drill the rest of the tab down to that group.",
            ),
            dbc.Row([
                dbc.Col(stat_card("Total penguins", total_penguins, "records after preprocessing"), md=3),
                dbc.Col(stat_card("Species", n_species, "distinct penguin species"), md=3),
                dbc.Col(stat_card("Islands", n_islands, "sampling locations"), md=3),
                dbc.Col(stat_card("Largest species", largest_species, "highest average body mass"), md=3),
            ], className="g-4 mb-4"),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.P("Drill-down controls", style=SECTION_LABEL),
                        html.H4("Click a species to drill down", style={"fontWeight": "700", "color": TEXT_DARK, "marginBottom": "8px"}),
                        html.P(
                            "The species bars drive the rest of this tab. You can also narrow the story to one or more islands, then reset the interaction at any time.",
                            style=MUTED,
                        ),
                    ], md=8),
                    dbc.Col([
                        html.Label("Island filter", style=CONTROL_LABEL),
                        dcc.Dropdown(
                            id="overview-island-filter",
                            options=[{"label": island, "value": island} for island in sorted(df["island"].unique())],
                            multi=True,
                            placeholder="All islands",
                        ),
                        dbc.Button("Reset drill-down", id="overview-reset", color="light", className="mt-3", style={
                            "width": "100%",
                            "border": f"1px solid {DIVIDER}",
                            "fontWeight": "600",
                        }),
                    ], md=4),
                ], className="g-4"),
            ], style={**SOFT_CARD_STYLE, "marginBottom": "24px"}),
            dbc.Row([
                dbc.Col(graph_card(
                    title="Species distribution",
                    subtitle="This is the entry point for the drill-down interaction.",
                    graph_id="overview-species-bar",
                    height=360,
                ), md=5),
                dbc.Col(graph_card(
                    title="Physical separation in measurement space",
                    subtitle="Flipper length and body mass alone already create visible species boundaries.",
                    graph_id="overview-main-scatter",
                    height=560,
                ), md=7),
            ], className="g-4 mb-4"),
            dbc.Row([
                dbc.Col(graph_card(
                    fig_donut,
                    title="Species composition by island",
                    subtitle="Each island hosts a distinct mix of species.",
                    height=420,
                ), md=6),
                dbc.Col(graph_card(
                    fig_island_bar,
                    title="Species distribution across islands",
                    subtitle="Each island hosts a distinct mix of penguin species.",
                    height=420,
                ), md=6),
            ], className="g-4 mt-2"),
            html.Div(id="overview-insight-box"),
        ])
    if tab == "tab-compare":
        return html.Div([
            section_header(
                "Comparisons",
                "Clean comparative views",
                "This tab keeps only the highest-value diagnostics: one distribution view and one relationship view.",
            ),
            html.Div(info_box(
                "Dominant insight",
                "Species differ because size-related measurements move together. Gentoo sits clearly above the others in mass, while the heatmap shows why flipper length and body mass are so informative."
            ), style={"marginBottom": "24px"}),
            dbc.Row([
                dbc.Col(graph_card(
                    fig_violin,
                    title="Body mass distribution by species",
                    subtitle="Gentoo separates cleanly on size, while Adelie and Chinstrap are closer together.",
                    height=460,
                ), md=5),
                dbc.Col(graph_card(
                    fig_corr,
                    title="Correlation heatmap",
                    subtitle="Physical measurements reinforce each other rather than competing for signal.",
                    height=460,
                ), md=7),
            dbc.Row([
                dbc.Col(graph_card(
                    fig_mass_bar,
                    title="Male penguins are heavier across all species",
                    subtitle="Sex-based size difference is consistent across islands.",
                    height=420,
                ), md=12),
            ], className="g-4 mt-2"),
            ], className="g-4"),
        ])
    if tab == "tab-ml":
        return html.Div([
            section_header(
                "Machine Learning",
                "Can we prove the separation?",
                "This tab keeps the models fixed and uses them to explain why morphology and isotopes separate species so well.",
            ),
            dbc.Row([
                dbc.Col(stat_card("RF accuracy", f"{accuracy*100:.1f}%", "holdout test accuracy"), md=4),
                dbc.Col(stat_card("K-means clusters", "3", "same count as known species"), md=4),
                dbc.Col(stat_card("Model features", len(features), "morphology + isotopes"), md=4),
            ], className="g-4 mb-4"),
            html.Div(info_box(
                "Dominant insight",
                "Once isotopes join the body measurements, the structure becomes even cleaner: clustering lines up closely with species and the classifier stays highly accurate."
            ), style={"marginBottom": "24px"}),
            dbc.Row([
                dbc.Col(graph_card(
                    fig_elbow,
                    title="Choosing the number of clusters",
                    subtitle="The elbow softens after k=2, but that shortcut misses the full biological structure.",
                    height=360,
                ), width=12),
            ], className="g-4 mb-4"),
            dbc.Row([
                dbc.Col(graph_card(
                    build_ml_decision_figure(),
                    title="Final clustering decision",
                    subtitle="We fix k=3 because it is the most interpretable choice for the known three-species system.",
                    height=460,
                ), width=12),
            ], className="g-4 mb-4"),
            dbc.Row([
                dbc.Col(graph_card(
                    build_ml_species_alignment_figure("cluster_k3"),
                    title="Cluster view",
                    subtitle="Unsupervised groups already recover most of the species pattern.",
                    height=420,
                ), md=6),
                dbc.Col(graph_card(
                    build_ml_species_alignment_figure("species"),
                    title="True labels",
                    subtitle="The biological classes occupy nearly the same PCA regions.",
                    height=420,
                ), md=6),
            ], className="g-4 mb-4"),
            dbc.Row([
                dbc.Col(graph_card(
                    fig_feat_imp,
                    title="Feature importance",
                    subtitle="The classifier leans most heavily on the same measurements that stand out in the visual analysis.",
                    height=380,
                ), md=6),
                dbc.Col(graph_card(
                    fig_cm,
                    title="Confusion matrix",
                    subtitle="Prediction errors are rare across the holdout set.",
                    height=380,
                ), md=6),
            ], className="g-4 mb-4"),
            html.Div([
                html.P("Section 4", style=SECTION_LABEL),
                html.H4("Interactive exploration", style={"fontWeight": "700", "fontSize": "22px", "color": TEXT_DARK, "marginBottom": "6px"}),
                html.P(
                    "Filtering affects visualization only; models remain fixed for consistency.",
                    style={**MUTED, "marginBottom": "18px"},
                ),
                dbc.Row([
                    dbc.Col([
                        html.Label("Species", style=CONTROL_LABEL),
                        dcc.Dropdown(
                            id="ml-filter-species",
                            options=[{"label": s, "value": s} for s in sorted(df["species"].unique())],
                            multi=True,
                            placeholder="All species",
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Label("Island", style=CONTROL_LABEL),
                        dcc.Dropdown(
                            id="ml-filter-island",
                            options=[{"label": i, "value": i} for i in sorted(df["island"].unique())],
                            multi=True,
                            placeholder="All islands",
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Label("Compare", style=CONTROL_LABEL),
                        dcc.RadioItems(
                            id="ml-compare-mode",
                            options=[
                                {"label": "k=3", "value": "cluster_k3"},
                                {"label": "k=2 vs k=3 stress test", "value": "cluster_k2"},
                            ],
                            value="cluster_k3",
                            inline=False,
                            labelStyle={"display": "block", "marginBottom": "6px"},
                        ),
                    ], md=4),
            html.Div([
                html.H4("Classification Report", style={"fontWeight": "700", "marginBottom": "12px"}),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th(col) for col in report_df.columns
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(report_df.iloc[i][col]) for col in report_df.columns
                        ]) for i in range(len(report_df))
                    ])
                ], style={"width": "100%", "fontSize": "13px", "color": TEXT_DARK, "borderCollapse": "collapse", "marginTop": "12px"})
            ], style=CARD_STYLE),
            ], className="g-4"),])
        ])
    return html.Div([
        section_header(
            "Explorer",
            "Smart interaction",
            "Filter the dataset, choose any two measurement axes, inspect the resulting relationship, and click a point for row-level detail.",
        ),
        html.Div(info_box(
            "Dominant insight",
            "If the story is real, it should survive custom slicing. This tab lets you stress-test that claim by changing filters and measurement pairs yourself."
        ), style={"marginBottom": "24px"}),
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Species", style=CONTROL_LABEL),
                    dcc.Dropdown(
                        id="filter-species",
                        options=[{"label": s, "value": s} for s in sorted(df["species"].unique())],
                        multi=True,
                        placeholder="All species",
                    ),
                ], md=4),
                dbc.Col([
                    html.Label("Island", style=CONTROL_LABEL),
                    dcc.Dropdown(
                        id="filter-island",
                        options=[{"label": i, "value": i} for i in sorted(df["island"].unique())],
                        multi=True,
                        placeholder="All islands",
                    ),
                ], md=4),
                dbc.Col([
                    html.Label("Sex", style=CONTROL_LABEL),
                    dcc.Dropdown(
                        id="filter-sex",
                        options=[{"label": s.title(), "value": s} for s in sorted(df["sex"].unique())],
                        multi=True,
                        placeholder="All sexes",
                    ),
                ], md=4),
            ], className="g-4 mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("X axis", style=CONTROL_LABEL),
                    dcc.Dropdown(
                        id="x-axis",
                        options=[{"label": format_axis_label(c), "value": c} for c in numeric_cols],
                        value="flipper_length_mm",
                    ),
                ], md=6),
                dbc.Col([
                    html.Label("Y axis", style=CONTROL_LABEL),
                    dcc.Dropdown(
                        id="y-axis",
                        options=[{"label": format_axis_label(c), "value": c} for c in numeric_cols],
                        value="body_mass_g",
                    ),
                ], md=6),
            ], className="g-4"),
        ], style={**CARD_STYLE, "marginBottom": "24px"}),
        dbc.Row([
            dbc.Col(graph_card(
                title="Dynamic scatter explorer",
                subtitle="Use filters and axis swaps to test how strongly different measurements separate species.",
                graph_id="explorer-scatter",
                height=600,
            ), lg=8),
            dbc.Col([
                html.Div(id="explorer-insight-panel"),
                html.Div(id="explorer-inspect-panel", className="mt-4"),
            ], lg=4),
        ], className="g-4"),
    ])
@app.callback(
    Output("overview-selected-species", "data"),
    Input("overview-species-bar", "clickData"),
    Input("overview-reset", "n_clicks"),
    State("overview-selected-species", "data"),
)
def update_overview_selection(click_data, reset_clicks, current_selection):
    trigger = dash.ctx.triggered_id
    if trigger == "overview-reset":
        return None
    if trigger == "overview-species-bar" and click_data:
        selected = click_data["points"][0]["x"]
        return None if current_selection == selected else selected
    return current_selection
@app.callback(
    Output("overview-species-bar", "figure"),
    Output("overview-main-scatter", "figure"),
    Output("overview-insight-box", "children"),
    Input("overview-selected-species", "data"),
    Input("overview-island-filter", "value"),
)
def update_overview_tab(selected_species, islands):
    dff = filter_overview_df(selected_species, islands)
    return (
        build_species_distribution_figure(selected_species),
        build_overview_scatter(dff, selected_species),
        build_overview_insight(dff, selected_species),
    )

@app.callback(
    Output("ml-interactive-pca", "figure"),
    Input("ml-filter-species", "value"),
    Input("ml-filter-island", "value"),
    Input("ml-compare-mode", "value"),
)
def update_ml_interactive(species, islands, compare_mode):
    return build_ml_interactive_figure(species, islands, compare_mode)


@app.callback(
    Output("explorer-scatter", "figure"),
    Output("explorer-insight-panel", "children"),
    Output("explorer-inspect-panel", "children"),
    Input("filter-species", "value"),
    Input("filter-island", "value"),
    Input("filter-sex", "value"),
    Input("x-axis", "value"),
    Input("y-axis", "value"),
    Input("explorer-scatter", "clickData"),
)
def update_explorer(species, island, sex, x_col, y_col, click_data):
    dff = filtered_explorer_df(species, island, sex)
    x_label = format_axis_label(x_col)
    y_label = format_axis_label(y_col)
    if len(dff) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No penguins match the current filters.",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            font=dict(size=16, color=TEXT_MUTED),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title=f"{y_label} vs {x_label} (0 penguins)")
        fig = apply_figure_theme(fig, 560)
        insight_panel = info_box("Automatic insight", "Change the filters to populate the scatter plot and compute a new relationship summary.")
        inspect_panel = build_detail_list("Click-to-inspect", [html.P("Select a point in the scatter plot to inspect its measurements.", style=MUTED)])
        return fig, insight_panel, inspect_panel
    explorer_fig = px.scatter(
        dff,
        x=x_col,
        y=y_col,
        color="species",
        symbol="island",
        color_discrete_map=COLORS,
        trendline="ols",
        hover_data=["species", "island", "sex", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
        labels={x_col: x_label, y_col: y_label},
        title=f"{y_label} vs {x_label} ({len(dff)} penguins)",
    )
    explorer_fig.update_traces(marker=dict(size=10, opacity=0.82, line=dict(width=0.5, color="white")))
    explorer_fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Island: %{customdata[1]}<br>"
            "Sex: %{customdata[2]}<br>"
            f"{x_label}: %{{x:.1f}}<br>"
            f"{y_label}: %{{y:.1f}}<br>"
            "Flipper length: %{customdata[5]:.0f} mm<br>"
            "Body mass: %{customdata[6]:.0f} g<extra></extra>"
        )
    )
    explorer_fig.update_layout(clickmode="event+select")
    explorer_fig = apply_figure_theme(explorer_fig, 560)
    if x_col == y_col:
        correlation_text = "The same variable is selected on both axes, so the relationship is perfectly linear by definition."
    else:
        corr_value = dff[x_col].corr(dff[y_col])
        strength = "strong" if abs(corr_value) >= 0.7 else "moderate" if abs(corr_value) >= 0.4 else "weak"
        direction = "positive" if corr_value >= 0 else "negative"
        correlation_text = f"{strength.title()} {direction} correlation (r = {corr_value:.2f}) between {x_label.lower()} and {y_label.lower()}."
    insight_panel = html.Div([
        info_box("Automatic insight", correlation_text),
        html.Div([
            html.P("Selection summary", style=SECTION_LABEL),
            html.P(
                f"{len(dff)} penguins remain after filtering across {dff['species'].nunique()} species and {dff['island'].nunique()} islands.",
                style={**MUTED, "fontSize": "14px", "color": TEXT_DARK},
            ),
        ], style={**CARD_STYLE, "marginTop": "16px"}),
    ])
    if click_data and click_data.get("points"):
        point = click_data["points"][0]
        custom_data = point.get("customdata", [])
        inspect_items = [
            html.Div(f"Species: {custom_data[0]}", style={**MUTED, "color": TEXT_DARK}),
            html.Div(f"Island: {custom_data[1]}", style={**MUTED, "color": TEXT_DARK}),
            html.Div(f"Sex: {custom_data[2]}", style={**MUTED, "color": TEXT_DARK}),
            html.Div(f"Bill length: {custom_data[3]} mm", style={**MUTED, "color": TEXT_DARK}),
            html.Div(f"Bill depth: {custom_data[4]} mm", style={**MUTED, "color": TEXT_DARK}),
            html.Div(f"Flipper length: {custom_data[5]} mm", style={**MUTED, "color": TEXT_DARK}),
            html.Div(f"Body mass: {custom_data[6]} g", style={**MUTED, "color": TEXT_DARK}),
        ]
    else:
        inspect_items = [html.P("Click a point in the scatter plot to inspect the selected penguin.", style=MUTED)]
    inspect_panel = build_detail_list("Click-to-inspect", inspect_items)
    return explorer_fig, insight_panel, inspect_panel
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Palmer Penguins Dashboard is live!")
    print("  Open http://127.0.0.1:8050 in your browser")
    print("="*60 + "\n")
    app.run(debug=False, port=8050)
