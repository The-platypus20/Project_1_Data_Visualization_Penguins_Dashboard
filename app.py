import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# --- PAGE CONFIGURATION ---
# Set the page title and layout to 'wide' to utilize the full screen width
st.set_page_config(page_title="Penguin Dashboard", layout="wide")

# --- DATA LOADING ---
# Using streamlit's cache decorator to prevent reloading data on every user interaction
@st.cache_data
def load_data():
    # Ensure the path matches your local directory structure
    path = '/Users/anngothesoloist/Project_1_Data_Visualization_Penguins_Dashboard/data/penguins_clean.csv'
    return pd.read_csv(path)

# Initialize the raw dataset
df_raw = load_data()

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
tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlations", "Counts & Comparison", "3D Analysis"])

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