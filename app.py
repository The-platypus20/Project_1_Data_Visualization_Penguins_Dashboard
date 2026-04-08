import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIGURATION ---
# Set the page title and layout to 'wide' to utilize the full screen width
st.set_page_config(page_title="Penguin Dashboard", layout="wide")

# --- DATA LOADING ---
# Using streamlit's cache decorator to prevent reloading data on every user interaction
@st.cache_data
def load_data():
    # Use relative path for cross-platform compatibility
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'data', 'penguins_preprocessed.csv')
    return pd.read_csv(path)

# Initialize the raw dataset
df_raw = load_data()
df_raw.columns = (
    df_raw.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
)

df_raw.rename(columns={
    "culmen_length_mm": "bill_length_mm",
    "culmen_depth_mm": "bill_depth_mm"
}, inplace=True)

# --- THEME & GLOBAL COLOR MAPPING ---
# Set a consistent white background theme and predefined colors for each penguin species
pio.templates.default = "plotly_white"
COLORS = {
    "Adelie Penguin (Pygoscelis adeliae)": '#FF8C00', 
    "Chinstrap penguin (Pygoscelis antarctica)": '#9932CC', 
    "Gentoo penguin (Pygoscelis papua)": '#057076'
}

# --- SIDEBAR FILTERS (GLOBAL CONTROLS) ---
# These sidebar widgets allow users to filter the entire dashboard by Species or Island
st.sidebar.header("Global Controls")
st.sidebar.write("Select what to display on ALL charts:")

# Species Multi-select: Default is all species selected
selected_species = st.sidebar.multiselect(
    "Select Penguin Species:",
    options=df_raw['species'].unique(),
    default=df_raw['species'].unique()
)

# Island Multi-select: Default is all islands selected
selected_islands = st.sidebar.multiselect(
    "Select Islands:",
    options=df_raw['island'].unique(),
    default=df_raw['island'].unique()
)

# --- APPLY FILTERING LOGIC ---
# The 'df' variable is the filtered version used by all figures below
df = df_raw[
    (df_raw['species'].isin(selected_species)) & 
    (df_raw['island'].isin(selected_islands))
]

# Dashboard Title and Summary Metrics
st.title("🐧 Palmer Penguins Data Visualization")
st.write(f"Showing **{df.shape[0]}** samples based on your filters.")

# Safety check: If no data is selected, stop the app to prevent errors
if df.empty:
    st.warning("No data matches the selected filters. Please select at least one species and one island.")
    st.stop()

# --- NAVIGATION TABS ---
# Organize charts into tabs to avoid a cluttered single-page view
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Distributions", "Correlations", "Counts & Comparison", "3D Analysis", "Machine Learning"])

# --- TAB 1: DISTRIBUTIONS ---
with tab1:
    st.header("1. Distribution: Weight by Species")
    # Fig 1: Violin Plot showing data density, quartiles, and individual points
    fig1 = px.violin(df, y="body_mass_g", x="species", color="species", 
                     box=True, points="all", hover_data=df.columns,
                     title="Distribution of weights by species",
                     color_discrete_map=COLORS)
    st.plotly_chart(fig1, use_container_width=True)

# --- TAB 2: CORRELATIONS ---
with tab2:
    st.header("2. Correlation & Scatter Matrix")
    # Fig 2: Heatmap visualizing Pearson correlation between physical attributes
    corr = df.select_dtypes(include=[np.number]).corr()
    fig2 = px.imshow(corr, text_auto=True, aspect="auto",
                     title="Correlation matrix of physical figures",
                     color_continuous_scale='RdBu_r')
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Fig 3: High-level Scatter Matrix for all morphological dimensions
    fig3 = px.scatter_matrix(
        df, 
        dimensions=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
        color="species", 
        color_discrete_map=COLORS,
        title="Penguin Morphological Matrix",
        labels={col: col.replace('_', ' ').title() for col in df.columns}
    )
    fig3.update_traces(diagonal_visible=True)
    fig3.update_yaxes(tickangle=0, title_standoff=10, automargin=True)
    fig3.update_layout(
        height=900, width=1200, margin=dict(l=180, r=20, t=80, b=80), 
        autosize=True, title={'text': "Penguin Morphological Matrix", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}
    )
    st.plotly_chart(fig3, use_container_width=True)

# --- TAB 3: COUNTS & COMPARISONS ---
with tab3:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.header("4. Species Count")
        # Fig 4: Histogram showing the population size of each selected species
        fig4 = px.histogram(df, x="species", color="species", color_discrete_map=COLORS, title="Total Count", text_auto=True)
        fig4.update_layout(title_x=0.5, xaxis_title="Species", yaxis_title="Number", showlegend=False, plot_bgcolor="white", bargap=0.3, height=500)
        fig4.update_traces(textposition='outside', textfont_size=14, marker_line_color='black', marker_line_width=1)
        st.plotly_chart(fig4, use_container_width=True)

    with col_b:
        st.header("5. Marginal Distributions")
        # Fig 5: Scatter Plot with marginal distributions (histogram/violin) and trendlines
        fig5 = px.scatter(df, x="flipper_length_mm", y="bill_length_mm", color="species",
                          marginal_x="histogram", marginal_y="violin", color_discrete_map=COLORS,
                          trendline="ols", title="Flipper vs Bill Length")
        fig5.update_layout(title_x=0.5, height=600)
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    
    # Fig 6: Horizontal Bar Chart faceted by Island to compare average mass by Sex
    df_grouped = df.groupby(['island', 'species', 'sex'])['body_mass_g'].mean().reset_index()
    fig6_horizontal = px.bar(df_grouped, x="body_mass_g", y="species", color="sex", orientation='h', barmode="group",
                             facet_col="island", color_discrete_map={"Male": "#7f7f7f", "Female": "#bcbd22"},
                             title="Average Body Mass by Species, Island, and Sex")
    fig6_horizontal.update_yaxes(tickangle=0, ticksuffix="    ", griddash='dot')
    fig6_horizontal.update_layout(title_x=0.5, height=500, margin=dict(t=80, l=100, r=40, b=80),
                                  legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5))
    fig6_horizontal.update_traces(texttemplate='%{x:.0f}', textposition='outside')
    st.plotly_chart(fig6_horizontal, use_container_width=True)

    st.markdown("---")

    # Fig 7: Donut Chart showing species composition percentage per Island
    fig7 = px.pie(df, names="species", facet_col="island", color="species", color_discrete_map=COLORS, hole=0.4, title="Proportion")
    fig7.update_layout(title_x=0.5)
    st.plotly_chart(fig7, use_container_width=True)

# --- TAB 4: 3D ANALYSIS ---
with tab4:
    st.header("8. 3D Morphological Analysis")
    # Fig 8: Interactive 3D Scatter Plot to see species clustering in 3D space
    fig8 = px.scatter_3d(df, x='bill_length_mm', y='flipper_length_mm', z='body_mass_g',
                         color='species', color_discrete_map=COLORS, opacity=0.8, title="3D Clusters")
    fig8.update_traces(marker=dict(size=4, line=dict(width=1, color='DarkSlateGrey')))
    fig8.update_layout(
        height=950, width=1100, title_x=0.5, title_font_size=22,
        # Adjusting the 3D scene: domain pushes the cube up, camera sets the initial viewpoint
        scene=dict(domain=dict(y=[0.2, 1.0]), camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                   xaxis=dict(gridcolor='white', showbackground=True, backgroundcolor="rgb(230, 230, 230)"),
                   yaxis=dict(gridcolor='white', showbackground=True, backgroundcolor="rgb(230, 230, 230)"),
                   zaxis=dict(gridcolor='white', showbackground=True, backgroundcolor="rgb(230, 230, 230)")),
        margin=dict(l=0, r=0, b=50, t=100),
        legend=dict(yanchor="top", y=0.9, xanchor="center", x=0.05, bgcolor="rgba(255, 255, 255, 0.5)")
    )
    st.plotly_chart(fig8, use_container_width=True)

# --- TAB 5: MACHINE LEARNING ---
with tab5:
    st.header("Machine Learning Analysis")

    # Prepare data for ML
    features = [
    'bill_length_mm',
    'bill_depth_mm',
    'flipper_length_mm',
    'body_mass_g',
    'delta_15_n_o/oo',
    'delta_13_c_o/oo'
]
    X = df[features].copy()
    X.fillna(X.mean(), inplace=True)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means Clustering
    st.subheader("K-Means Clustering with PCA")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]

    fig_ml1 = px.scatter(df, x='PCA1', y='PCA2', color='Cluster',
                         title='K-Means Clustering (PCA-reduced)',
                         color_continuous_scale='viridis')
    fig_ml1.update_layout(
        xaxis_title=f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)',
        yaxis_title=f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)'
    )
    st.plotly_chart(fig_ml1, use_container_width=True)

    # t-SNE
    st.subheader("t-SNE Visualization")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42, learning_rate='auto', init='random')
    X_tsne = tsne.fit_transform(X_scaled)
    df['TSNE1'] = X_tsne[:, 0]
    df['TSNE2'] = X_tsne[:, 1]

    fig_ml2 = px.scatter(df, x='TSNE1', y='TSNE2', color='Species',
                         title='t-SNE Visualization (Colored by Species)',
                         color_discrete_map=COLORS)
    st.plotly_chart(fig_ml2, use_container_width=True)

    # Random Forest Classification
    st.subheader("Random Forest Classification")
    X_clf = X.copy()
    y_clf = df['Species'].copy()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_clf)

    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
    scaler_clf = StandardScaler()
    X_train_scaled = scaler_clf.fit_transform(X_train)
    X_test_scaled = scaler_clf.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Model Accuracy:** {accuracy:.2f}")

    st.write("**Classification Report:**")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.write("**Confusion Matrix:**")
    conf_mat = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(conf_mat, index=label_encoder.classes_, columns=label_encoder.classes_))

    # Feature Importances
    st.subheader("Feature Importances")
    importances = model.feature_importances_
    feature_names = X_clf.columns
    forest_importances = pd.Series(importances, index=feature_names)

    fig_ml3 = px.bar(forest_importances.reset_index(), x='index', y=0,
                     title='Feature Importances',
                     labels={'index': 'Feature', 0: 'Importance'})
    fig_ml3.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_ml3, use_container_width=True)
