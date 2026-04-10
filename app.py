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
st.set_page_config(page_title="Penguin Dashboard", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'data', 'penguins_preprocessed.csv')
    data = pd.read_csv(path)
    # Standardize column names
    data.columns = (
        data.columns.str.strip().str.lower()
        .str.replace(" ", "_").str.replace("(", "").str.replace(")", "")
    )
    data.rename(columns={
        "culmen_length_mm": "bill_length_mm",
        "culmen_depth_mm": "bill_depth_mm"
    }, inplace=True)
    return data

df_raw = load_data()

# --- THEME & GLOBAL COLOR MAPPING ---
pio.templates.default = "plotly_white"
COLORS = {
    "Adelie Penguin (Pygoscelis adeliae)": '#FF8C00', 
    "Chinstrap penguin (Pygoscelis antarctica)": '#9932CC', 
    "Gentoo penguin (Pygoscelis papua)": '#057076'
}

# Global label dictionary for consistency
LABEL_MAP = {
    "bill_length_mm": "Bill Length (mm)",
    "bill_depth_mm": "Bill Depth (mm)",
    "flipper_length_mm": "Flipper Length (mm)",
    "body_mass_g": "Body Mass (g)",
    "delta_15_n": "Delta 15 N (o/oo)",
    "delta_13_c": "Delta 13 C (o/oo)",
    "species": "Species",
    "island": "Island",
    "sex": "Sex"
}

# --- SIDEBAR FILTERS ---
st.sidebar.header("Global Controls")
selected_species = st.sidebar.multiselect(
    "Select Penguin Species:", options=df_raw['species'].unique(), default=df_raw['species'].unique()
)
selected_islands = st.sidebar.multiselect(
    "Select Islands:", options=df_raw['island'].unique(), default=df_raw['island'].unique()
)

df = df_raw[(df_raw['species'].isin(selected_species)) & (df_raw['island'].isin(selected_islands))]

st.title("🐧 Palmer Penguins Data Visualization")
st.write(f"Showing **{df.shape[0]}** samples based on your filters.")

if df.empty:
    st.warning("No data matches the selected filters. Please adjust your selection.")
    st.stop()

# --- NAVIGATION TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Distributions", "Correlations", "Counts & Comparison", 
    "3D Analysis", "Ecological Niche", "Machine Learning"
])

# --- TAB 1: DISTRIBUTIONS ---
with tab1:
    st.header("1. Weight Distribution")
    fig1 = px.violin(df, y="body_mass_g", x="species", color="species", 
                     box=True, points="all", hover_data=df.columns,
                     title="Distribution of Body Mass (g) by Species",
                     color_discrete_map=COLORS, labels=LABEL_MAP)
    st.plotly_chart(fig1, use_container_width=True)

# --- TAB 2: CORRELATIONS ---
with tab2:
    st.header("2. Correlation & Matrix")
    corr = df.select_dtypes(include=[np.number]).corr()
    fig2 = px.imshow(corr, text_auto=True, aspect="auto",
                     title="Correlation Matrix of Physical Figures",
                     color_continuous_scale='RdBu_r', labels=LABEL_MAP)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    fig3 = px.scatter_matrix(
        df, dimensions=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
        color="species", color_discrete_map=COLORS,
        title="Penguin Morphological Matrix", labels=LABEL_MAP
    )
    fig3.update_layout(height=800)
    st.plotly_chart(fig3, use_container_width=True)

# --- TAB 3: COUNTS & COMPARISONS ---
with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        st.header("4. Species Count")
        fig4 = px.histogram(df, x="species", color="species", color_discrete_map=COLORS, 
                            title="Total Count of Each Species", text_auto=True, labels=LABEL_MAP)
        st.plotly_chart(fig4, use_container_width=True)
    with col_b:
        st.header("5. Marginal Distributions")
        fig5 = px.scatter(df, x="flipper_length_mm", y="bill_length_mm", color="species",
                          marginal_x="histogram", marginal_y="violin", color_discrete_map=COLORS,
                          trendline="ols", title="Flipper Length (mm) vs Bill Length (mm)", labels=LABEL_MAP)
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    st.header("6. Multi-Feature Analysis")
    features_list = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    df_multi = df.groupby(['species', 'sex'])[features_list].mean().reset_index()
    df_melted = df_multi.melt(id_vars=["species", "sex"], value_vars=features_list, var_name="feature", value_name="average_value")

    fig6 = px.bar(df_melted, x="average_value", y="species", color="sex", orientation='h', barmode="group",
                  facet_col="feature", facet_col_wrap=2, color_discrete_map={"Male": "#7f7f7f", "Female": "#bcbd22"},
                  title="Average Physical Features by Species and Sex", labels=LABEL_MAP)
    fig6.update_yaxes(tickangle=0, ticksuffix="   ")
    fig6.update_xaxes(matches=None)
    fig6.for_each_annotation(lambda a: a.update(text=LABEL_MAP.get(a.text.split("=")[-1], a.text)))
    fig6.update_traces(texttemplate='%{x:.1f}', textposition='outside')
    fig6.update_layout(height=800, title_x=0.5, legend=dict(orientation="h", y=-0.1, x=0.5, xanchor='center'))
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")
    st.header("7. Population by Island")
    fig7 = px.histogram(df, x="island", color="species", color_discrete_map=COLORS, barmode="stack",
                        title="Species Distribution Across Islands (Stacked)", labels=LABEL_MAP, text_auto=True)
    fig7.update_layout(xaxis_title="Island Name", yaxis_title="Total Population", title_x=0.5)
    st.plotly_chart(fig7, use_container_width=True)

# --- TAB 4: 3D ANALYSIS ---
with tab4:
    st.header("8. 3D Morphological Analysis")
    fig8 = px.scatter_3d(df, x='bill_length_mm', y='flipper_length_mm', z='body_mass_g',
                         color='species', color_discrete_map=COLORS, opacity=0.8, 
                         title="3D Clusters of Physical Metrics", labels=LABEL_MAP)
    fig8.update_layout(height=800, title_x=0.5)
    st.plotly_chart(fig8, use_container_width=True)

# --- TAB 5: ECOLOGICAL NICHE ---
with tab5:
    st.header("9. Isotopic & Morphological Niche")
    col_c, col_d = st.columns(2)
    with col_c:
        fig9 = px.scatter(df, x="delta_13_c", y="delta_15_n", color="species", color_discrete_map=COLORS,
                          marginal_x="box", marginal_y="box", title="Nitrogen vs Carbon Isotopes",
                          labels={
                              "delta_13_c": "Delta 13 C (o/oo) - Habitat",
                              "delta_15_n": "Delta 15 N (o/oo) - Diet",
                              "species": "Species"
                          })
        st.plotly_chart(fig9, use_container_width=True)
    with col_d:
        df_ternary = df.copy()
        for col in ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]:
            df_ternary[col+"_scaled"] = (df_ternary[col] - df_ternary[col].min()) / (df_ternary[col].max() - df_ternary[col].min())
        fig10 = px.scatter_ternary(df_ternary, a="bill_length_mm_scaled", b="bill_depth_mm_scaled", c="flipper_length_mm_scaled",
                                   color="species", color_discrete_map=COLORS, size="body_mass_g", size_max=10,
                                   title="Relative Morphological Balance (Scaled Features)",
                                   labels={
                                       "bill_length_mm_scaled": "Bill Length",
                                       "bill_depth_mm_scaled": "Bill Depth",
                                       "flipper_length_mm_scaled": "Flipper Length"
                                   })
        st.plotly_chart(fig10, use_container_width=True)

# --- TAB 6: MACHINE LEARNING ---
with tab6:
    st.header("Machine Learning Analysis")
    ml_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'delta_15_n', 'delta_13_c']
    X = df[ml_features].copy().fillna(df[ml_features].mean())
    X_scaled = StandardScaler().fit_transform(X)

    # K-Means
    st.subheader("K-Means Clustering with PCA")
    df['Cluster'] = KMeans(n_clusters=3, random_state=42, n_init='auto').fit_predict(X_scaled)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'], df['PCA2'] = X_pca[:, 0], X_pca[:, 1]
    fig_pca = px.scatter(df, x='PCA1', y='PCA2', color='Cluster', title='K-Means Clusters (PCA-reduced)', color_continuous_scale='viridis')
    st.plotly_chart(fig_pca, use_container_width=True)

    # t-SNE (Colored by Species for comparison)
    st.subheader("t-SNE Visualization")
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)
    df['TSNE1'], df['TSNE2'] = X_tsne[:, 0], X_tsne[:, 1]
    fig_tsne = px.scatter(df, x='TSNE1', y='TSNE2', color='species', color_discrete_map=COLORS, title='t-SNE (Colored by Species)', labels=LABEL_MAP)
    st.plotly_chart(fig_tsne, use_container_width=True)

    # Random Forest
    st.subheader("Random Forest Classification")
    le = LabelEncoder()
    y = le.fit_transform(df['species'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    # Random Forest Classification
    st.subheader("Random Forest Classification")
    le = LabelEncoder()
    y = le.fit_transform(df['species'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # Hiển thị Accuracy nổi bật
    st.metric("Model Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    
    # --- THÊM BẢNG F1, PRECISION, RECALL ---
    st.write("**Detailed Classification Report:**")
    report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Làm tròn số để bảng trông sạch hơn
    st.dataframe(report_df.style.format("{:.2f}").background_gradient(cmap='Greens', subset=['f1-score', 'precision', 'recall']))
    
    # Feature Importance (Standardized colors)
    importances = pd.Series(rf.feature_importances_, index=[LABEL_MAP.get(f, f) for f in ml_features]).sort_values(ascending=True)
    fig_imp = px.bar(importances, orientation='h', title="Feature Importance for Species Classification", labels={'value': 'Importance Score', 'index': 'Feature'})
    fig_imp.update_traces(marker_color='#057076')
    st.plotly_chart(fig_imp, use_container_width=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, x=le.classes_, y=le.classes_, text_auto=True, title="Confusion Matrix", labels=dict(x="Predicted", y="Actual"))
    st.plotly_chart(fig_cm, use_container_width=True)