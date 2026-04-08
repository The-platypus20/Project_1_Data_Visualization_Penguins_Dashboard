"""
Palmer Penguins — One-Click Pipeline + Interactive Dash Dashboard
Run:  python penguin_dashboard.py
Then open http://127.0.0.1:8050 in your browser.

Pipeline stages (auto-executed on startup):
  1. Load  →  2. Preprocess  →  3. Visualizations  →  4. ML  →  5. Dashboard
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

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
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

print("✅ All libraries imported.")

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 ─ LOAD
# ═════════════════════════════════════════════════════════════════════════════
print("\n📦 [Stage 1] Loading Palmer Penguins dataset …")
df_raw = load_penguins()
print(f"   Raw shape: {df_raw.shape}  |  Missing values:\n{df_raw.isnull().sum().to_string()}")

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 ─ PRE-PROCESSING
#   • Drop 2 rows that are completely empty of physical measurements
#   • Fill remaining NA values (numeric → mean, sex → mode)
# ═════════════════════════════════════════════════════════════════════════════
print("\n🔧 [Stage 2] Pre-processing …")

numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

# Drop the 2 rows where ALL numeric measurements are missing
df = df_raw.dropna(subset=numeric_cols, how="all").copy()
print(f"   Dropped {len(df_raw) - len(df)} rows with all-NA measurements.")

# Fill remaining numeric NAs with column mean
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Fill sex NA with mode
df["sex"] = df["sex"].fillna(df["sex"].mode()[0])

print(f"   Clean shape: {df.shape}  |  Remaining NAs: {df.isnull().sum().sum()}")

# Friendly colour palette (matches visualization notebook)
COLORS = {
    "Adelie":    "#FF8C00",
    "Chinstrap": "#9932CC",
    "Gentoo":    "#057076",
}

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 ─ VISUALIZATIONS  (Plotly figures stored for the dashboard)
# ═════════════════════════════════════════════════════────────────────────────
print("\n📊 [Stage 3] Building visualisation figures …")

# ── 3-A  Species distribution bar chart ──────────────────────────────────────
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

# ── 3-B  Species × Island grouped bar ────────────────────────────────────────
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

# ── 3-C  Flipper length vs Body mass scatter ─────────────────────────────────
fig_scatter = px.scatter(
    df, x="flipper_length_mm", y="body_mass_g",
    color="species", color_discrete_map=COLORS,
    marginal_x="histogram", marginal_y="violin",
    trendline="ols",
    title="Flipper Length vs Body Mass",
    labels={
        "flipper_length_mm": "Flipper Length (mm)",
        "body_mass_g": "Body Mass (g)",
    },
    hover_data=["island", "sex", "bill_length_mm"],
)
fig_scatter.update_layout(
    title_x=0.5,
    annotations=[dict(
        text="Strong positive correlation: larger flippers → heavier penguins. Gentoo are biggest.",
        xref="paper", yref="paper", x=0.5, y=-0.18,
        showarrow=False, font=dict(size=12, color="#555"),
    )],
    margin=dict(b=70),
)

# ── 3-D  Weight distribution violin ──────────────────────────────────────────
fig_violin = px.violin(
    df, y="body_mass_g", x="species", color="species",
    box=True, points="all", color_discrete_map=COLORS,
    title="Body Mass Distribution by Species",
    labels={"body_mass_g": "Body Mass (g)", "species": "Species"},
)
fig_violin.update_layout(showlegend=False)

# ── 3-E  Correlation heatmap ─────────────────────────────────────────────────
corr = df[numeric_cols].corr()
fig_corr = px.imshow(
    corr, text_auto=".2f", aspect="auto",
    title="Correlation Matrix of Physical Measurements",
    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
)
fig_corr.update_layout(title_x=0.5)

# ── 3-F  Body mass by island + sex horizontal bar ────────────────────────────
df_grouped = df.groupby(["island", "species", "sex"])["body_mass_g"].mean().reset_index()
fig_mass_bar = px.bar(
    df_grouped, x="body_mass_g", y="species", color="sex",
    orientation="h", barmode="group", facet_col="island",
    color_discrete_map={"male": "#7f7f7f", "female": "#bcbd22"},
    title="Average Body Mass by Species, Island, and Sex",
    labels={"body_mass_g": "Avg Body Mass (g)", "species": "Species"},
)
fig_mass_bar.update_layout(title_x=0.5, height=420,
    legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"))

# ── 3-G  3-D scatter ─────────────────────────────────────────────────────────
fig_3d = px.scatter_3d(
    df, x="bill_length_mm", y="flipper_length_mm", z="body_mass_g",
    color="species", color_discrete_map=COLORS, opacity=0.8,
    title="3D Morphological Clusters",
    labels={"bill_length_mm": "Bill (mm)", "flipper_length_mm": "Flipper (mm)", "body_mass_g": "Mass (g)"},
)
fig_3d.update_traces(marker=dict(size=4, line=dict(width=1, color="DarkSlateGrey")))
fig_3d.update_layout(title_x=0.5, height=500)

# ── 3-H  Donut by island ─────────────────────────────────────────────────────
fig_donut = px.pie(
    df, names="species", facet_col="island",
    color="species", color_discrete_map=COLORS,
    hole=0.4, title="Species Composition per Island",
)
fig_donut.update_layout(title_x=0.5)

print("   ✅ All visualisation figures ready.")

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 ─ MACHINE LEARNING
# ═════════════════════════════════════════════════════════════════════════════
print("\n🤖 [Stage 4] Running Machine Learning …")

features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X = df[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4-A  K-Means clustering ───────────────────────────────────────────────────
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
df["cluster"] = kmeans.fit_predict(X_scaled).astype(str)
print("   K-Means: 3 clusters fitted.")

# ── 4-B  PCA ─────────────────────────────────────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]
var1, var2 = pca.explained_variance_ratio_ * 100
print(f"   PCA: PC1={var1:.1f}%  PC2={var2:.1f}% variance explained.")

fig_pca_cluster = px.scatter(
    df, x="PCA1", y="PCA2", color="cluster",
    symbol="species",
    title=f"K-Means Clusters on PCA Space (PC1={var1:.1f}%, PC2={var2:.1f}%)",
    labels={"PCA1": f"PC1 ({var1:.1f}%)", "PCA2": f"PC2 ({var2:.1f}%)"},
    hover_data=["species", "island"],
)
fig_pca_cluster.update_layout(title_x=0.5)

fig_pca_species = px.scatter(
    df, x="PCA1", y="PCA2", color="species",
    color_discrete_map=COLORS,
    title=f"True Species on PCA Space",
    labels={"PCA1": f"PC1 ({var1:.1f}%)", "PCA2": f"PC2 ({var2:.1f}%)"},
    hover_data=["island", "sex"],
)
fig_pca_species.update_layout(title_x=0.5)

# ── 4-C  Random Forest Classifier ────────────────────────────────────────────
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
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
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

print("   ✅ ML complete.")

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5 ─ DASH DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
print("\n🚀 [Stage 5] Launching Dash dashboard …")

# ── Colour palette & typography ───────────────────────────────────────────────
THEME_BG      = "#F4F7FA"
CARD_BG       = "#FFFFFF"
ACCENT        = "#057076"
ACCENT_LIGHT  = "#E8F4F5"
TEXT_DARK     = "#1A2B3C"
TEXT_MUTED    = "#6B7A8D"
DIVIDER       = "#DEE3EC"

CARD_STYLE = {
    "background": CARD_BG,
    "borderRadius": "12px",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.06)",
    "padding": "20px",
    "marginBottom": "20px",
}
INSIGHT_STYLE = {
    "background": ACCENT_LIGHT,
    "borderLeft": f"4px solid {ACCENT}",
    "borderRadius": "0 8px 8px 0",
    "padding": "12px 16px",
    "marginTop": "8px",
    "fontSize": "13px",
    "color": TEXT_DARK,
}
SECTION_TITLE = {
    "color": ACCENT,
    "fontWeight": "700",
    "letterSpacing": "0.08em",
    "fontSize": "11px",
    "textTransform": "uppercase",
    "marginBottom": "4px",
}
H2 = {"color": TEXT_DARK, "fontWeight": "700", "marginBottom": "4px"}
MUTED = {"color": TEXT_MUTED, "fontSize": "13px"}


def stat_card(title, value, subtitle=""):
    return html.Div([
        html.P(title, style={**MUTED, "marginBottom": "2px"}),
        html.H3(value, style={"color": ACCENT, "fontWeight": "800", "margin": "0"}),
        html.P(subtitle, style={**MUTED, "marginTop": "2px", "fontSize": "11px"}),
    ], style={**CARD_STYLE, "textAlign": "center", "padding": "16px"})


def section_header(label, title, subtitle=""):
    return html.Div([
        html.P(label, style=SECTION_TITLE),
        html.H2(title, style=H2),
        html.P(subtitle, style=MUTED) if subtitle else None,
        html.Hr(style={"borderColor": DIVIDER, "margin": "10px 0 20px"}),
    ])


def graph_card(figure, insight="", height=420):
    children = [dcc.Graph(figure=figure, style={"height": f"{height}px"})]
    if insight:
        children.append(html.Div(f"💡 {insight}", style=INSIGHT_STYLE))
    return html.Div(children, style=CARD_STYLE)


# ── Layout ────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Palmer Penguins Dashboard",
)

# KPI stats
total_penguins = len(df)
n_species      = df["species"].nunique()
n_islands      = df["island"].nunique()
heaviest       = int(df["body_mass_g"].max())

app.layout = html.Div(style={"background": THEME_BG, "minHeight": "100vh",
                              "fontFamily": "'Segoe UI', system-ui, sans-serif"}, children=[

    # ── NAV BAR ──────────────────────────────────────────────────────────────
    html.Div(style={
        "background": ACCENT, "padding": "0 32px",
        "display": "flex", "alignItems": "center",
        "justifyContent": "space-between", "height": "60px",
    }, children=[
        html.Span("🐧 Palmer Penguins", style={
            "color": "white", "fontWeight": "700", "fontSize": "18px",
        }),
        html.Span("Explore · Compare · Predict", style={
            "color": "rgba(255,255,255,0.75)", "fontSize": "13px",
        }),
    ]),

    # ── TAB NAVIGATION ───────────────────────────────────────────────────────
    html.Div(style={"padding": "0 32px", "background": "white",
                    "borderBottom": f"1px solid {DIVIDER}"}, children=[
        dcc.Tabs(id="main-tabs", value="tab-overview", style={"border": "none"},
                 colors={"border": "white", "primary": ACCENT, "background": "white"},
                 children=[
            dcc.Tab(label="📊  Overview", value="tab-overview",
                    style={"fontWeight": "600"}, selected_style={"fontWeight": "700", "color": ACCENT}),
            dcc.Tab(label="🔬  Comparisons", value="tab-compare",
                    style={"fontWeight": "600"}, selected_style={"fontWeight": "700", "color": ACCENT}),
            dcc.Tab(label="🤖  Machine Learning", value="tab-ml",
                    style={"fontWeight": "600"}, selected_style={"fontWeight": "700", "color": ACCENT}),
            dcc.Tab(label="🔍  Explorer", value="tab-explorer",
                    style={"fontWeight": "600"}, selected_style={"fontWeight": "700", "color": ACCENT}),
        ]),
    ]),

    # ── TAB CONTENT ──────────────────────────────────────────────────────────
    html.Div(id="tab-content", style={"padding": "24px 32px", "maxWidth": "1400px", "margin": "0 auto"}),
])


# ── Tab renderer ─────────────────────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):

    # ════════════ TAB 1 — OVERVIEW ══════════════════════════════════════════
    if tab == "tab-overview":
        return html.Div([
            section_header("Chapter 1", "Overview",
                           "Primary insights visible at a glance — no interaction required."),

            # KPI row
            dbc.Row([
                dbc.Col(stat_card("Total Penguins", total_penguins, "after preprocessing"), md=3),
                dbc.Col(stat_card("Species", n_species, "Adelie · Chinstrap · Gentoo"), md=3),
                dbc.Col(stat_card("Islands", n_islands, "Biscoe · Dream · Torgersen"), md=3),
                dbc.Col(stat_card("Heaviest Individual", f"{heaviest:,} g", "Gentoo species"), md=3),
            ], className="mb-2"),

            # Narrative intro
            html.Div([
                dcc.Markdown("""
The **Palmer Penguins** dataset contains morphological measurements of **344 penguins** 
from three species collected on three islands of the Palmer Archipelago, Antarctica.

**Story arc:** Adelie is the most abundant species. Gentoo penguins are the largest 
and are exclusively found on Biscoe Island. Chinstrap penguins live only on Dream Island.

Explore the tabs above to dive deeper into comparisons and machine-learning predictions.
                """, style={"lineHeight": "1.8"}),
            ], style={**CARD_STYLE, "borderLeft": f"4px solid {ACCENT}"}),

            dbc.Row([
                dbc.Col(graph_card(
                    fig_species_bar, height=400,
                    insight="Adelie (152) is the most represented species — nearly 50% of the dataset.",
                ), md=6),
                dbc.Col(graph_card(
                    fig_island_bar, height=400,
                    insight="Gentoo is exclusive to Biscoe. Dream hosts both Adelie & Chinstrap.",
                ), md=6),
            ]),

            graph_card(
                fig_scatter, height=480,
                insight=(
                    "Strong positive correlation between flipper length and body mass "
                    "(r ≈ 0.87). Gentoo penguins occupy the top-right cluster, indicating "
                    "they are consistently larger."
                ),
            ),
        ])

    # ════════════ TAB 2 — COMPARISONS ═══════════════════════════════════════
    elif tab == "tab-compare":
        return html.Div([
            section_header("Chapter 2", "Comparisons",
                           "Drill into physical-trait differences between species and islands."),

            dbc.Row([
                dbc.Col(graph_card(fig_violin, height=420,
                    insight="Gentoo has the highest median body mass (~5,100 g) and the widest spread."), md=6),
                dbc.Col(graph_card(fig_corr, height=420,
                    insight="Flipper length & body mass are highly correlated (+0.87). Bill depth shows negative correlation with bill length."), md=6),
            ]),

            graph_card(fig_mass_bar, height=440,
                insight="Males are consistently heavier than females across all species. Torgersen has only Adelie penguins."),

            dbc.Row([
                dbc.Col(graph_card(fig_donut, height=400,
                    insight="Biscoe is dominated by Gentoo (73%). Dream is split ~50/50 Adelie & Chinstrap."), md=6),
                dbc.Col(graph_card(fig_3d, height=480,
                    insight="In 3D space, the three species form well-separated morphological clusters, suggesting high classifiability."), md=6),
            ]),
        ])

    # ════════════ TAB 3 — MACHINE LEARNING ═══════════════════════════════════
    elif tab == "tab-ml":
        # Build classification report table
        report_rows = []
        for _, row in report_df.iterrows():
            report_rows.append(
                html.Tr([html.Td(str(v), style={"padding": "6px 12px", "borderBottom": f"1px solid {DIVIDER}",
                                                 "fontWeight": "700" if row["Class"] == "accuracy" else "400"})
                         for v in row], style={"background": ACCENT_LIGHT if row["Class"] in le.classes_ else "white"})
            )

        return html.Div([
            section_header("Chapter 3", "Machine Learning",
                           "Unsupervised clustering (K-Means + PCA) and supervised classification (Random Forest)."),

            # ML KPIs
            dbc.Row([
                dbc.Col(stat_card("RF Accuracy", f"{accuracy*100:.1f}%", "on 30% holdout test set"), md=4),
                dbc.Col(stat_card("K-Means Clusters", "3", "matching species count"), md=4),
                dbc.Col(stat_card("PCA Variance", f"{var1+var2:.1f}%", f"PC1={var1:.1f}%  PC2={var2:.1f}%"), md=4),
            ]),

            dbc.Row([
                dbc.Col(graph_card(fig_pca_species, height=420,
                    insight="True species separate cleanly in 2D PCA space, validating that morphology alone is highly discriminative."), md=6),
                dbc.Col(graph_card(fig_pca_cluster, height=420,
                    insight="K-Means clusters align closely with actual species — confirming natural clustering in morphological data."), md=6),
            ]),

            dbc.Row([
                dbc.Col(graph_card(fig_feat_imp, height=380,
                    insight="Flipper length and body mass are the most predictive features. Bill depth also contributes meaningfully."), md=6),
                dbc.Col(graph_card(fig_cm, height=380,
                    insight="Almost zero misclassifications. The Random Forest model achieves near-perfect species identification."), md=6),
            ]),

            # Classification report table
            html.Div([
                html.H5("Classification Report", style={"color": TEXT_DARK, "fontWeight": "700", "marginBottom": "12px"}),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th(col, style={"padding": "8px 12px", "background": ACCENT, "color": "white",
                                            "textAlign": "left"})
                        for col in report_df.columns
                    ])),
                    html.Tbody(report_rows),
                ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "13px"}),
            ], style=CARD_STYLE),
        ])

    # ════════════ TAB 4 — EXPLORER ═══════════════════════════════════════════
    elif tab == "tab-explorer":
        return html.Div([
            section_header("Chapter 4", "Interactive Explorer",
                           "Filter and drill down into the data using the controls below."),

            # Controls
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Species", style={"fontWeight": "600", "fontSize": "13px"}),
                        dcc.Dropdown(
                            id="filter-species",
                            options=[{"label": s, "value": s} for s in df["species"].unique()],
                            multi=True,
                            placeholder="All species",
                        ),
                    ], md=3),
                    dbc.Col([
                        html.Label("Island", style={"fontWeight": "600", "fontSize": "13px"}),
                        dcc.Dropdown(
                            id="filter-island",
                            options=[{"label": i, "value": i} for i in df["island"].unique()],
                            multi=True,
                            placeholder="All islands",
                        ),
                    ], md=3),
                    dbc.Col([
                        html.Label("Sex", style={"fontWeight": "600", "fontSize": "13px"}),
                        dcc.Dropdown(
                            id="filter-sex",
                            options=[{"label": s.title(), "value": s} for s in df["sex"].unique()],
                            multi=True,
                            placeholder="All sexes",
                        ),
                    ], md=3),
                    dbc.Col([
                        html.Label("X Axis", style={"fontWeight": "600", "fontSize": "13px"}),
                        dcc.Dropdown(
                            id="x-axis",
                            options=[{"label": c.replace("_mm", " (mm)").replace("_g", " (g)").replace("_", " ").title(), "value": c}
                                     for c in numeric_cols],
                            value="flipper_length_mm",
                        ),
                    ], md=1.5),
                    dbc.Col([
                        html.Label("Y Axis", style={"fontWeight": "600", "fontSize": "13px"}),
                        dcc.Dropdown(
                            id="y-axis",
                            options=[{"label": c.replace("_mm", " (mm)").replace("_g", " (g)").replace("_", " ").title(), "value": c}
                                     for c in numeric_cols],
                            value="body_mass_g",
                        ),
                    ], md=1.5),
                ]),
            ], style={**CARD_STYLE, "marginBottom": "12px"}),

            # Dynamic scatter
            html.Div(id="explorer-scatter", style=CARD_STYLE),

            # Dynamic data table summary
            html.Div(id="explorer-stats", style=CARD_STYLE),
        ])


# ── Explorer callbacks ────────────────────────────────────────────────────────
@app.callback(
    Output("explorer-scatter", "children"),
    Output("explorer-stats", "children"),
    Input("filter-species", "value"),
    Input("filter-island", "value"),
    Input("filter-sex", "value"),
    Input("x-axis", "value"),
    Input("y-axis", "value"),
)
def update_explorer(species, island, sex, x_col, y_col):
    dff = df.copy()
    if species:
        dff = dff[dff["species"].isin(species)]
    if island:
        dff = dff[dff["island"].isin(island)]
    if sex:
        dff = dff[dff["sex"].isin(sex)]

    x_label = x_col.replace("_mm", " (mm)").replace("_g", " (g)").replace("_", " ").title()
    y_label = y_col.replace("_mm", " (mm)").replace("_g", " (g)").replace("_", " ").title()

    fig = px.scatter(
        dff, x=x_col, y=y_col, color="species", symbol="island",
        color_discrete_map=COLORS,
        trendline="ols",
        title=f"{y_label} vs {x_label}  ({len(dff)} penguins)",
        labels={x_col: x_label, y_col: y_label},
        hover_data=["island", "sex", "bill_length_mm", "bill_depth_mm"],
    )
    fig.update_layout(height=420)

    # Summary stats table
    if len(dff) == 0:
        stats_content = html.P("No penguins match the current filters.", style=MUTED)
    else:
        summary = dff.groupby("species")[numeric_cols].mean().round(1).reset_index()
        header_cols = ["Species"] + [c.replace("_mm", " (mm)").replace("_g", " (g)").replace("_", " ").title()
                                     for c in numeric_cols]
        rows = []
        for _, row in summary.iterrows():
            rows.append(html.Tr([
                html.Td(str(v), style={"padding": "6px 12px", "borderBottom": f"1px solid {DIVIDER}",
                                       "color": COLORS.get(str(row["species"]), TEXT_DARK),
                                       "fontWeight": "600" if _ == 0 else "400"})
                for v in row
            ]))
        stats_content = [
            html.H6(f"Mean measurements — {len(dff)} penguins selected",
                    style={"fontWeight": "700", "color": TEXT_DARK, "marginBottom": "10px"}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th(h, style={"padding": "8px 12px", "background": ACCENT,
                                      "color": "white", "textAlign": "left"})
                    for h in header_cols
                ])),
                html.Tbody(rows),
            ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "13px"}),
        ]

    return dcc.Graph(figure=fig), stats_content


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🐧  Palmer Penguins Dashboard is live!")
    print("  👉  Open http://127.0.0.1:8050 in your browser")
    print("="*60 + "\n")
    app.run(debug=False, port=8050)
