import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import requests
import sqlite3
import time
import gc
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import random
from scipy.stats import poisson
from datetime import datetime, timedelta
import matplotlib.cm as cm
import seaborn as sns




# --- Define translations ---
T = {
    "dashboard_title": "🌍 Earthquake analysis and prediction",
    "dashboard_desc": "Analyze and visualize global seismic activity data from multiple sources.",
    "filters": "Filters",
    "date_range": "Date Range",
    "magnitude_range": "Magnitude Range",
    "depth_range": "Depth Range",
    "event_types": "Event Types",
    "region_filter": "Region Filter",
    "download_csv": "Download as CSV",
    "selected_events": "Selected Events",
    "advanced_options": "Advanced Options",
    "show_clusters": "Show Cluster Analysis",
    "show_advanced_charts": "Show Advanced Charts",
    "reset_filters": "Reset Filters",
    "upload_csv": "Upload CSV Data",
    "help": "This dashboard analyzes earthquake data from the USGS. Use filters to refine your analysis.",
    "about": "Created by Alfonso Cifuentes Alonso. Data from USGS, IGN España, and EMSC."
}

# --- Page Configuration --- (must be before any Streamlit UI code)
st.set_page_config(
    page_title="🌍 Interactive Seismic Activity Dashboard",  # Use default language
    page_icon="🌍",
    layout="wide"
)

# --- Page title ---
st.title(T["dashboard_title"])
st.markdown(T["dashboard_desc"])
df = None

st.markdown("""
<style>
/* Google Fonts import */
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Fira+Mono&display=swap');

html, body, [class*="css"]  {
    font-family: 'Montserrat', sans-serif !important;
    background: linear-gradient(120deg, #232526 0%, #2c2c2e 100%) !important;
    color: #f7f7f7 !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    color: #ffb347 !important; /* yellowish orange */
}

.stApp {
    background: linear-gradient(120deg, #232526 0%, #2c2c2e 100%) !important;
}

/* Hacer que la barra de navegación tenga fondo transparente */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-radius: 12px 12px 0 0;
    padding: 0.5rem 1rem;
    border: none !important;
    border-bottom: none;
    box-shadow: none !important;
}

.stTabs [data-baseweb="tab"] {
    color: #ffb347 !important;
    font-weight: 600;
    font-size: 1.1rem;
    padding: 0.7rem 1.5rem;
    border-radius: 8px 8px 0 0;
    margin-right: 0.5rem;
    transition: all 0.3s ease;
    background: rgba(35, 37, 38, 0.5) !important; /* Semi-transparente */
    border: 1px solid rgba(255, 179, 71, 0.2);
    border-bottom: none;
    position: relative;
    top: 0;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.1), 
                0 -2px 3px rgba(0,0,0,0.2);
    backdrop-filter: blur(5px); /* Efecto de desenfoque para dar profundidad */
    -webkit-backdrop-filter: blur(5px);
}

.stTabs [data-baseweb="tab"]:hover {
    box-shadow: 0 0 15px rgba(255,220,50,0.6); /* yellow glow effect */
    transform: translateY(-2px);
    border-color: rgba(255, 179, 71, 0.5);
    background: rgba(35, 37, 38, 0.7) !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(35, 37, 38, 0.9) !important;
    color: #fff !important;
    box-shadow: inset 0 2px 0 #ffb347, 
                inset 2px 0 0 rgba(255,255,255,0.1),
                inset -2px 0 0 rgba(255,255,255,0.1),
                0 -3px 5px rgba(0,0,0,0.3);
    border-color: rgba(255, 179, 71, 0.5);
    position: relative;
    top: -3px;
    z-index: 1;
}

/* El resto del CSS se mantiene igual */
.stButton>button {
    background: #e67300 !important; /* dark orange, no gradient */
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: 700 !important;
    font-size: 1.1rem;
    padding: 0.6rem 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    transition: box-shadow 0.3s, transform 0.2s;
}
.stButton>button:hover {
    box-shadow: 0 0 15px rgba(255,220,50,0.6); /* yellow glow effect */
    transform: translateY(-2px);
}

.stDownloadButton>button {
    background: #e67300 !important; /* dark orange, no gradient */
    color: #ffffff;
    border-radius: 8px;
    font-weight: 700 !important;
    font-size: 1.1rem;
    padding: 0.6rem 1.5rem;
    margin: 0.5rem 0;
    border: none;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    transition: box-shadow 0.3s, transform 0.2s;
}
.stDownloadButton>button:hover {
    box-shadow: 0 0 15px rgba(255,220,50,0.6); /* yellow glow effect */
    transform: translateY(-2px);
}

.stSidebar {
    background: rgba(40, 40, 45, 0.98) !important;
    border-radius: 0 16px 16px 0;
    box-shadow: 2px 0 12px rgba(255,179,71,0.08);
    color: #f7f7f7 !important;
}

.stSidebar .stButton>button, .stSidebar .stDownloadButton>button {
    width: 100%;
}

.stMetric {
    background: rgba(255,179,71,0.10);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 8px rgba(255,179,71,0.04);
    color: #ffb347 !important;
}

.stDataFrame, .stTable {
    background: rgba(35,37,38,0.98) !important;
    border-radius: 12px !important;
    color: #f7f7f7 !important;
    font-family: 'Fira Mono', monospace !important;
    font-size: 1.05rem !important;
}

.stAlert, .stSuccess, .stWarning, .stError {
    border-radius: 10px !important;
    font-size: 1.1rem !important;
    color: #fff !important;
    background: #b22222 !important;
}

.stMarkdown code {
    background: #232526 !important;
    color: #ffb347 !important;
    border-radius: 6px;
    padding: 0.2em 0.4em;
    font-size: 1.05em;
}

.stExpanderHeader {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #ffb347 !important;
}

/* Slider track and handle */
.stSlider > div > div > div > div {
    background: #ff9800 !important; /* dark orange track */
}
.stSlider > div > div > div > div > div {
    background: #232526 !important; /* dark red handle */
}
            
.stSlider input {
    color: white !important;
    background-color: rgba(35, 37, 38, 0.9) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255, 179, 71, 0.3) !important;
    padding: 4px 8px !important;
    font-family: 'Fira Mono', monospace !important;
    text-align: center !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
}

/* Quitar el fondo rojo y mejorar el hover */
.stSlider input:hover, .stSlider input:focus {
    border-color: #ffb347 !important;
    box-shadow: 0 0 8px rgba(255, 179, 71, 0.5) !important;
    background-color: rgba(40, 40, 45, 0.95) !important;
}

/* Ajustar el ancho para que quepa mejor el texto */
.stSlider input[type="number"] {
    width: 80px !important;
    min-width: 80px !important;
}

/* Selectbox dropdown */
.stSelectbox>div>div>div>div {
    background: #232526 !important;
    color: #ffb347 !important;
}

/* Radio buttons */
.stRadio>div>label {
    font-size: 1.08rem !important;
    color: #ffb347 !important;
}

.stSubheader {
    color: #ffb347 !important;
    font-weight: 700 !important;
}

/* Arregla el problema de los gráficos que se salen de sus contenedores */
.stPlotlyChart {
    border-radius: 16px !important;
    box-shadow: 0 2px 16px rgba(255,179,71,0.07);
    background: #232526 !important;
    padding: 1rem;
    overflow: hidden !important; /* Evita que el contenido se salga */
    max-width: 100% !important; /* Garantiza que no sea más ancho que el contenedor */
}

/* Asegúrate de que cualquier elemento dentro del gráfico respeta el overflow */
.stPlotlyChart > div {
    border-radius: 16px !important;
    overflow: hidden !important;
    max-width: 100% !important;
}

/* Arregla específicamente los contenedores de SVG de Plotly */
.stPlotlyChart .js-plotly-plot, .stPlotlyChart .plot-container {
    border-radius: 16px !important;
    overflow: hidden !important;
    max-width: 100% !important;
}

/* También aplicar a iframe dentro de mapas */
.stPlotlyChart iframe {
    border-radius: 16px !important;
    overflow: hidden !important;
    max-width: 100% !important;
}

/* Ajustes para folium maps */
.stFolium {
    border-radius: 16px !important;
    overflow: hidden !important;
    max-width: 100% !important;
}

.stFolium iframe {
    border-radius: 16px !important;
}

.stTabs {
    margin-bottom: 1.5rem !important;
}

/* Adding 3D effect to tab content area */
.stTabs [data-baseweb="tab-panel"] {
    border: 1px solid rgba(255, 179, 71, 0.2);
    border-top: none;
    border-radius: 0 0 12px 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    padding: 1rem;
    background: rgba(35, 37, 38, 0.8);
    overflow: hidden; /* Evitar que el contenido se salga */
    max-width: 100%; /* Garantiza que no sea más ancho que el contenedor */
}

/* Asegura que los contenidos de panel respeten los bordes */
.stTabs [data-baseweb="tab-panel"] > div {
    overflow: hidden;
    max-width: 100%;
}

.stMarkdown h2, .stMarkdown h3 {
    color: #ffb347 !important;
}

.stMarkdown ul {
    margin-left: 1.5rem;
}

.stMarkdown li {
    margin-bottom: 0.3rem;
}

::-webkit-scrollbar {
    width: 10px;
    background: #232526;
}
::-webkit-scrollbar-thumb {
    background: #ffb347;
    border-radius: 8px;
}

/* Ajuste para 3D plots */
.stPlotlyChart .svg-container {
    border-radius: 16px !important;
    overflow: hidden !important;
}

/* Estilos específicos para gráficos que podrían desbordarse */
.js-plotly-plot .plotly {
    max-width: 100% !important;
    overflow: hidden !important;
}
            
            /* Make tabs more compact to prevent horizontal scrolling */
.stTabs [data-baseweb="tab"] {
    padding: 0.6rem 1.0rem !important;  /* Reduced padding */
    font-size: 0.95rem !important;      /* Slightly smaller font */
    margin-right: 0.3rem !important;    /* Less space between tabs */
    white-space: nowrap;                /* Prevent text wrapping */
}

/* Make tab content container take full width */
.stTabs [data-baseweb="tab-list"] {
    width: 100% !important;
    overflow-x: auto !important;        /* Allow horizontal scrolling only for tab bar if needed */
    flex-wrap: nowrap !important;       /* Prevent wrapping of tabs */
}

/* Make scrollable tab bar if needed */
.stTabs [data-baseweb="tab-list"] > div {
    display: flex !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    scrollbar-width: thin !important;
    max-width: 100% !important;
}

/* Reduce the size of the icons in tabs */
.stTabs [data-baseweb="tab"] [data-testid="stMarkdownContainer"] p {
    display: inline-flex !important;
    align-items: center !important;
}
            
/* Color negro para los textos en multiselect (Event Types) */
.stMultiSelect [data-baseweb="tag"] {
    color: black !important;
    font-weight: 500 !important;
}

/* Asegurarse de que el texto sea visible en los selectbox y multiselect */
.stMultiSelect span, .stMultiSelect div span {
    color: black !important;
}

/* Texto en el dropdown del multiselect */
.stMultiSelect [role="listbox"] span {
    color: black !important;
}

/* Color para el texto de las opciones seleccionadas */
.stMultiSelect [data-baseweb="tag"] span {
    color: black !important;
}

/* Asegurar que los textos en el menú desplegable sean negros */
div[data-baseweb="popover"] ul li {
    color: black !important;
}
            
            /* Centrar títulos de gráficos */
.stPlotlyChart .gtitle {
    text-align: center !important;
    width: 100% !important;
    left: 0 !important;
    margin: 0 auto !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #ffb347 !important;
}

/* Aplicar estilo a los contenedores de gráficos */
.stPlotlyChart, .stFolium {
    border-radius: 16px !important;
    box-shadow: 0 4px 12px rgba(255,179,71,0.15);
    padding: 1rem 1rem 0.5rem 1rem !important;
    margin-bottom: 1.5rem !important;
    background: rgba(35, 37, 38, 0.7) !important;
}

/* Mejorar la apariencia de los títulos dentro de los mapas de folium */
.stFolium h3, .stFolium h4, .folium-map h3, .folium-map h4 {
    text-align: center !important;
    margin: 10px auto !important;
    color: white !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
}

/* Asegurar que los títulos del mapa se centren correctamente */
.folium-map .leaflet-control {
    text-align: center !important;
    width: 100% !important;
}

/* Centrar títulos en todos los divs redondeados */
div[style*="border-radius"] h3, 
div[style*="border-radius"] h4,
div[class*="border-radius"] h3,
div[class*="border-radius"] h4 {
    text-align: center !important;
    width: 100% !important;
}
            
/* Mejorar visualización de gráficos y mapas para evitar leyendas ocultas */
.stPlotlyChart, .stFolium {
    border-radius: 16px !important;
    box-shadow: 0 4px 12px rgba(255,179,71,0.15);
    padding: 1.5rem !important;
    padding-bottom: 3.5rem !important;  /* Más espacio en la parte inferior para leyendas */
    margin-bottom: 2rem !important;
    background: rgba(35, 37, 38, 0.7) !important;
    max-height: 95vh !important;  /* Ligeramente más pequeño para dar espacio a leyendas */
    overflow: visible !important;  /* Permite que las leyendas sean visibles fuera del contenedor */
}

/* Asegurar que los contenedores de las leyendas son visibles */
.stPlotlyChart .legend, .stPlotlyChart .g-gtitle {
    overflow: visible !important;
}

/* Ajustar posición de las leyendas para mapas */
.stPlotlyChart .legend {
    transform: translateY(10px) !important;
}

/* Dar más espacio a gráficos 3D o complejos */
.stPlotlyChart.complex-chart {
    height: 600px !important;
    padding-bottom: 4rem !important;
}

/* Ajustar específicamente los mapas para que tengan más espacio para leyendas */
.stFolium .folium-map {
    margin-bottom: 30px !important;
}

/* Asegurar que las leyendas en Folium mapas son visibles */
.stFolium .leaflet-control-container .leaflet-bottom {
    bottom: 10px !important;
}

/* Añadir margen adicional al final de cada panel de tab para evitar corte */
.stTabs [data-baseweb="tab-panel"] > div {
    padding-bottom: 2rem !important;
}

/* Hacer los títulos de los gráficos más pequeños para ahorrar espacio vertical */
.stPlotlyChart .gtitle {
    font-size: 1rem !important;
}

/* Hacer que las leyendas tengan un fondo semi-transparente para mejorar legibilidad */
.stPlotlyChart .legend .bg {
    fill: rgba(35, 37, 38, 0.7) !important;
    stroke: rgba(255, 179, 71, 0.3) !important;
}

/* Corregir la visualización de tooltips */
.stPlotlyChart .hoverlayer {
    z-index: 1000 !important;
}
            
</style>
""", unsafe_allow_html=True)

# Function to load data
@st.cache_data(ttl=3600)
def load_data():
    try:
        url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv"
        df = pd.read_csv(url)

        # All processing inside cache
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        df['updated'] = pd.to_datetime(df['updated']).dt.tz_localize(None)
        df['day'] = df['time'].dt.date
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.day_name()
        df['week'] = df['time'].dt.isocalendar().week
        conditions = [
            (df['mag'] < 2.0),
            (df['mag'] >= 2.0) & (df['mag'] < 4.0),
            (df['mag'] >= 4.0) & (df['mag'] < 6.0),
            (df['mag'] >= 6.0)
        ]
        choices = ['Minor (<2)', 'Light (2-4)', 'Moderate (4-6)', 'Strong (6+)']
        df['magnitud_categoria'] = np.select(conditions, choices, default='Not classified')
        df['region'] = df['place'].str.split(', ').str[-1]
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Days of week translation
days_translation = {
    'Monday': 'Monday',
    'Tuesday': 'Tuesday',
    'Wednesday': 'Wednesday',
    'Thursday': 'Thursday',
    'Friday': 'Friday',
    'Saturday': 'Saturday',
    'Sunday': 'Sunday'
}

# Color scheme for magnitudes
magnitude_colors = {
    'Minor (<2)': 'blue',
    'Light (2-4)': 'green',
    'Moderate (4-6)': 'orange',
    'Strong (6+)': 'red'
}

# Function to ensure positive values for markers
def ensure_positive(values, min_size=3):
    if isinstance(values, (pd.Series, np.ndarray, list)):
        return np.maximum(np.abs(values), min_size)
    else:
        return max(abs(values), min_size)

# Function to plot magnitude distribution
def plot_magnitude_distribution(df):
    fig = px.histogram(
        df, x="mag", nbins=30, color="magnitud_categoria",
        color_discrete_map=magnitude_colors,
        labels={"mag": "Magnitude", "count": "Frequency"},
        title="Magnitude Distribution by Category"
    )
    fig.update_layout(bargap=0.1)
    return fig

# Function to create a titled chart with proper legend placement
def create_titled_chart(fig, title, height=None, use_container_width=True):
    """
    Creates a chart with proper title placement to avoid legend overlap
    """
    # Remove the original title from the figure
    fig.update_layout(title=None)
    
    # If height is specified, update it
    if height:
        fig.update_layout(height=height)
    
    # Add proper margin to ensure the plot elements don't crowd each other
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Display the title above the chart using Streamlit's markdown
    st.markdown(f"<h3 style='text-align: center; color: #ffb347;'>{title}</h3>", unsafe_allow_html=True)
    
    # Return the plotly chart
    return st.plotly_chart(fig, use_container_width=use_container_width)

# For maps and complex charts that might need special handling
def create_titled_map(fig, title, height=None, use_container_width=True):
    """
    Creates a map with proper title placement
    """
    # Remove the original title 
    fig.update_layout(title=None)
    
    # If height is specified, update it
    if height:
        fig.update_layout(height=height)
        
    # Add proper margin and adjust legend position for maps
    fig.update_layout(
        margin=dict(t=5, b=5, l=5, r=5),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(35, 37, 38, 0.7)",
            bordercolor="rgba(255, 179, 71, 0.3)",
            borderwidth=1
        )
    )
    
    # Display the title above the map using Streamlit's markdown
    st.markdown(f"<h3 style='text-align: center; color: #ffb347;'>{title}</h3>", unsafe_allow_html=True)
    
    # Return the plotly chart
    return st.plotly_chart(fig, use_container_width=use_container_width)

# Data filters
def filter_data(df, date_range, mag_range, depth_range, types, regions):
    mask = (
        (df['time'] >= pd.Timestamp(date_range[0])) &
        (df['time'] <= pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)) &
        (df['mag'] >= mag_range[0]) & (df['mag'] <= mag_range[1]) &
        (df['depth'] >= depth_range[0]) & (df['depth'] <= depth_range[1])
    )
    if types:
        mask &= df['type'].isin(types)
    if regions:
        mask &= df['region'].isin(regions)
    return df[mask].copy()

def reset_filters():
    st.session_state['date_range'] = [df['time'].min().date(), df['time'].max().date()]
    st.session_state['mag_range'] = (float(df['mag'].min()), float(df['mag'].max()))
    st.session_state['depth_range'] = (float(df['depth'].min()), float(df['depth'].max()))
    st.session_state['selected_types'] = df['type'].unique().tolist()
    st.session_state['selected_regions'] = []

def sidebar_filters(df, T):
    """Creates sidebar filters and returns their values"""
    st.sidebar.header(T["filters"])
    
    # Date range filter
    st.sidebar.subheader(T["date_range"])
    min_date = df['time'].min().date()
    max_date = df['time'].max().date()
    
    if 'date_range' not in st.session_state:
        st.session_state['date_range'] = [min_date, max_date]
        
    date_range = st.sidebar.date_input(
        "",
        value=st.session_state['date_range'],
        min_value=min_date,
        max_value=max_date
    )
    
    # Magnitude range filter
    st.sidebar.subheader(T["magnitude_range"])
    min_mag = float(round(df['mag'].min(), 1))
    max_mag = float(round(df['mag'].max(), 1))
    mag_min_val = min_mag - (min_mag % 0.1)  # Ajustar al múltiplo de 0.1 más cercano
    mag_max_val = max_mag + (0.1 - (max_mag % 0.1)) if max_mag % 0.1 != 0 else max_mag
    
    if 'mag_range' not in st.session_state:
        st.session_state['mag_range'] = (mag_min_val, mag_max_val)

    mag_range = st.sidebar.slider(
        "",
        min_value=mag_min_val,
        max_value=mag_max_val,
        value=(mag_min_val, mag_max_val),
        step=0.1,
        key="mag_range_slider"  # Clave única
    )
    
    # Depth range filter
    st.sidebar.subheader(T["depth_range"])
    min_depth = float(round(df['depth'].min(), 1))
    max_depth = float(round(df['depth'].max(), 1))
    depth_min_val = min_depth - (min_depth % 0.5)  # Ajustar al múltiplo de 0.5 más cercano
    depth_max_val = max_depth + (0.5 - (max_depth % 0.5)) if max_depth % 0.5 != 0 else max_depth

    if 'depth_range' not in st.session_state:
        st.session_state['depth_range'] = (depth_min_val, depth_max_val)
    
    # Mover esta línea FUERA del bloque condicional    
    depth_range = st.sidebar.slider(
        "",
        min_value=depth_min_val,
        max_value=depth_max_val,
        value=(depth_min_val, depth_max_val),
        step=0.5,
        key="depth_range_slider"  # Clave única
    )
    
    # Event types filter
    st.sidebar.subheader(T["event_types"])
    all_types = df['type'].unique().tolist()
    
    if 'selected_types' not in st.session_state:
        st.session_state['selected_types'] = all_types
        
    selected_types = st.sidebar.multiselect(
        "",
        options=all_types,
        default=st.session_state['selected_types']
    )
    
    # Region filter
    st.sidebar.subheader(T["region_filter"])
    regions = df['place'].str.split(', ').str[-1].unique().tolist()
    
    if 'selected_regions' not in st.session_state:
        st.session_state['selected_regions'] = []
        
    selected_regions = st.sidebar.multiselect(
        "",
        options=regions,
        default=st.session_state['selected_regions']
    )
    
    return date_range, mag_range, depth_range, selected_types, selected_regions

def download_filtered_data(df, T):
    csv = df.to_csv(index=False)
    st.download_button(
        label=T["download_csv"],
        data=csv,
        file_name="filtered_earthquake_data.csv",
        mime="text/csv",
        key="download_csv"
    )

def reset_filters():
    st.session_state['date_range'] = [df['time'].min().date(), df['time'].max().date()]
    st.session_state['mag_range'] = (float(df['mag'].min()), float(df['mag'].max()))
    st.session_state['depth_range'] = (float(df['depth'].min()), float(df['depth'].max()))
    st.session_state['selected_types'] = df['type'].unique().tolist()
    st.session_state['selected_regions'] = []

def plot_interactive_map(df):
    df_plot = df[df['mag'] > 0].copy()
    fig = px.scatter_mapbox(
        df_plot,  # Use the filtered DataFrame everywhere
        lat="latitude",
        lon="longitude",
        color="magnitud_categoria",
        size=ensure_positive(df_plot['mag']),
        hover_data=["place", "mag", "depth", "time"],
        zoom=1,
        height=500,
        mapbox_style="carto-positron"
    )
    return fig

# --- Help Section ---
with st.sidebar.expander("ℹ️ Help / Ayuda"):
    st.markdown(T["help"])

# --- User CSV Upload ---
uploaded_file = st.sidebar.file_uploader(T["upload_csv"], type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Custom data loaded!")
else:
    with st.spinner("Loading data..."):
        df = load_data()

if df is not None and not df.empty:
    # --- Dynamic Filtering ---
    if st.sidebar.button(T["reset_filters"]):
        reset_filters(df)
    date_range, mag_range, depth_range, selected_types, selected_regions = sidebar_filters(df, T)

    # Filtering logic
    filtered_df = df.copy()
    if len(date_range) == 2:
        start_date, end_date = date_range
        start_datetime = pd.Timestamp(start_date)
        end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered_df = filtered_df[(filtered_df['time'] >= start_datetime) & (filtered_df['time'] <= end_datetime)]
    min_mag, max_mag = mag_range
    filtered_df = filtered_df[(filtered_df['mag'] >= min_mag) & (filtered_df['mag'] <= max_mag)]
    min_depth, max_depth = depth_range
    filtered_df = filtered_df[(filtered_df['depth'] >= min_depth) & (filtered_df['depth'] <= max_depth)]
    if selected_types:
        filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]
    if selected_regions:
        region_mask = filtered_df['place'].str.contains('|'.join(selected_regions), case=False)
        filtered_df = filtered_df[region_mask]

    st.sidebar.metric(T["selected_events"], len(filtered_df))

    # --- Advanced Options ---
    st.sidebar.markdown("---")
    st.sidebar.header(T["advanced_options"])
    show_clusters = st.sidebar.checkbox(T["show_clusters"], value=False)
    show_advanced_charts = st.sidebar.checkbox(T["show_advanced_charts"], value=False)

    # --- Main Tabs ---
    if len(filtered_df) == 0:
        st.warning("No data available with the selected filters. Please adjust the filters.")
    else:
        main_tabs = st.tabs(["📊 General Summary", "🌐 Geographic Analysis", "⏱️ Temporal Analysis", "📈 Advanced Analysis", "🚨 Alert Center", "📚 Historical Analysis (2005-2025)"], )

        # --- General Summary Tab ---
        with main_tabs[0]:
            with st.container():
                
                col1, col2, col3, col4 = st.columns(4)

                # Calculate metrics
                total_events = len(filtered_df)
                avg_mag = filtered_df['mag'].mean()
                max_mag = filtered_df['mag'].max()
                avg_depth = filtered_df['depth'].mean()
                
                with col1:
                    st.metric(
                        label="Total Events",
                        value=f"{total_events}",
                        delta=None,
                        delta_color="normal"
                    )

                with col2:
                    st.metric(
                        label="Average Magnitude",
                        value=f"{avg_mag:.2f}",
                        delta=None,
                        delta_color="normal"
                    )

                with col3:
                    st.metric(
                        label="Maximum Magnitude",
                        value=f"{max_mag:.2f}",
                        delta=None,
                        delta_color="normal"
                    )
                with col4:
                    st.metric(
                        label="Average Depth (km)",
                        value=f"{avg_depth:.2f}",
                        delta=None,
                        delta_color="normal"
                    )

            col_dist1, col_dist2 = st.columns(2)
            with col_dist1:
                st.subheader("Magnitude Distribution")
                st.plotly_chart(plot_magnitude_distribution(filtered_df), use_container_width=True)
            with col_dist2:
                st.subheader("Depth Distribution")
                fig_depth = px.histogram(
                    filtered_df,
                    x="depth",
                    nbins=30,
                    color="magnitud_categoria",
                    color_discrete_map=magnitude_colors,
                    labels={"depth": "Depth (km)", "count": "Frequency"},
                    title="Depth Distribution by Magnitude Category"
                )
                fig_depth.update_layout(bargap=0.1)
                st.plotly_chart(fig_depth, use_container_width=True)

            st.subheader("Relationship between Magnitude and Depth")
            size_values = ensure_positive(filtered_df['mag'])
            fig_scatter = px.scatter(
                filtered_df,
                x="depth",
                y="mag",
                color="magnitud_categoria",
                size=size_values,
                size_max=15,
                opacity=0.7,
                hover_name="place",
                color_discrete_map=magnitude_colors,
                labels={"depth": "Depth (km)", "mag": "Magnitude"}
            )
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.subheader("Top 10 Regions with Highest Seismic Activity")
            top_places = filtered_df['place'].value_counts().head(10).reset_index()
            top_places.columns = ['Region', 'Number of Events']
            fig_top = px.bar(
                top_places,
                x='Number of Events',
                y='Region',
                orientation='h',
                text='Number of Events',
                color='Number of Events',
                color_continuous_scale='Viridis'
            )
            fig_top.update_traces(textposition='outside')
            fig_top.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
            st.plotly_chart(fig_top, use_container_width=True)

        # --- Geographic Analysis Tab ---
        with main_tabs[1]:
            geo_tabs = st.tabs(["Events Map", "Heat Map", "Cluster Analysis", "🌋 Volcanology"])
            with geo_tabs[0]:
                st.subheader("Geographic Distribution of Earthquakes")
                fig_map = px.scatter_geo(
                    filtered_df,
                    lat="latitude",
                    lon="longitude",
                    color="magnitud_categoria",
                    size=ensure_positive(filtered_df['mag']),
                    size_max=15,
                    hover_name="place",
                    hover_data={
                        "latitude": False,
                        "longitude": False,
                        "magnitud_categoria": False,
                        "mag": ":.2f",
                        "depth": ":.2f km",
                        "time": True,
                        "type": True
                    },
                    color_discrete_map=magnitude_colors,
                    projection="natural earth"
                )
                fig_map.update_layout(
                    margin={"r": 0, "t": 0, "l": 0, "b": 0},
                    height=600,
                    geo=dict(
                        showland=True,
                        landcolor="#d2b48c",      # yellowish brown for continents
                        showocean=True,
                        oceancolor="#002244",     # darker blue for oceans
                        showcountries=True,
                        countrycolor="white",
                        showcoastlines=True,
                        coastlinecolor="white",
                        bgcolor="black"           # black background for the map itself
                    ),
                    paper_bgcolor="black",
                    plot_bgcolor="black"
                )
                st.plotly_chart(fig_map, use_container_width=True)
                st.subheader("Significant Events (Magnitude ≥ 4.0)")
                significant_events = filtered_df[filtered_df['mag'] >= 4.0].sort_values(by='mag', ascending=False)
                if not significant_events.empty:
                    st.dataframe(
                        significant_events[['time', 'place', 'mag', 'depth', 'type']],
                        use_container_width=True
                    )
                else:
                    st.info("No events with magnitude ≥ 4.0 in the selected range.")

            with geo_tabs[3]:
                st.header("🌋 Volcanology and Seismic Relationships")
                st.markdown("""
                This section analyzes the relationship between volcanic activity and seismic events. 
                Volcanoes and earthquakes are closely related geological phenomena that often occur in the same regions
                due to tectonic plate boundaries and other geological factors.
                """)
                
                # Load volcano dataset
                @st.cache_data(ttl=3600)
                def load_volcano_data():
                    try:
                        # Intenta cargar con diferentes codificaciones
                        encodings_to_try = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
                        
                        for encoding in encodings_to_try:
                            try:
                                # Intenta cargar desde el archivo local primero
                                try:
                                    df_volcano = pd.read_csv('data/volcano_data.csv', encoding=encoding)
                                    st.success(f"Volcano data loaded successfully using {encoding} encoding")
                                    return df_volcano
                                except FileNotFoundError:
                                    # Si el archivo no existe, descarga un conjunto de datos de muestra
                                    url = "https://raw.githubusercontent.com/plotly/datasets/master/volcano_db.csv"
                                    df_volcano = pd.read_csv(url, encoding=encoding)
                                    st.success(f"Volcano data downloaded from GitHub using {encoding} encoding")
                                    return df_volcano
                            except UnicodeDecodeError:
                                continue  # Prueba con la siguiente codificación
                        
                        # Si todas las codificaciones fallan, crea un conjunto de datos mínimo como respaldo
                        st.warning("Could not load volcano data with any encoding. Using minimal sample dataset.")
                        data = {
                            'Volcano Name': ['Kilauea', 'Mount St. Helens', 'Krakatoa', 'Vesuvius', 'Fuji'],
                            'Country': ['United States', 'United States', 'Indonesia', 'Italy', 'Japan'],
                            'Region': ['Hawaii', 'Washington', 'Java', 'Naples', 'Honshu'],
                            'Latitude': [19.421, 46.200, -6.102, 40.821, 35.361],
                            'Longitude': [-155.287, -122.180, 105.423, 14.426, 138.731],
                            'Elevation (m)': [1222, 2549, 813, 1281, 3776],
                            'Type': ['Shield', 'Stratovolcano', 'Caldera', 'Stratovolcano', 'Stratovolcano'],
                            'Status': ['Active', 'Active', 'Active', 'Active', 'Active'],
                            'Last Known Eruption': ['2023', '2008', '2018', '1944', '1707']
                        }
                        return pd.DataFrame(data)
                    except Exception as e:
                        st.error(f"Error loading volcano data: {e}")
                        # Crear un conjunto de datos mínimo de respaldo
                        data = {
                            'volcano_name': ['Kilauea', 'Mount St. Helens', 'Krakatoa', 'Vesuvius', 'Fuji'],
                            'country': ['United States', 'United States', 'Indonesia', 'Italy', 'Japan'],
                            'latitude': [19.421, 46.200, -6.102, 40.821, 35.361],
                            'longitude': [-155.287, -122.180, 105.423, 14.426, 138.731],
                            'elevation': [1222, 2549, 813, 1281, 3776]
                        }
                        return pd.DataFrame(data)

                # Load and preprocess volcano data
                with st.spinner("Loading volcano data..."):
                    df_volcano = load_volcano_data()
                
                # Clean and transform volcano data
                if df_volcano is not None:
                    # Standardize column names
                    df_volcano.columns = [col.strip().replace(' ', '_').lower() for col in df_volcano.columns]
                    
                    # Ensure we have the necessary columns, rename if needed
                    column_map = {
                        'volcano_name': 'volcano_name',
                        'name': 'volcano_name',
                        'latitude': 'latitude',
                        'longitude': 'longitude',
                        'lat': 'latitude',
                        'lon': 'longitude',
                        'elev': 'elevation',
                        'elevation': 'elevation',
                        'elevation_(m)': 'elevation',
                        'type': 'volcano_type',
                        'volcano_type': 'volcano_type',
                        'country': 'country',
                        'region': 'region',
                        'status': 'status',
                        'last_known_eruption': 'last_eruption',
                        'last_eruption': 'last_eruption'
                    }
                    
                    # Rename columns that exist in the dataframe
                    for old_col, new_col in column_map.items():
                        if old_col in df_volcano.columns and old_col != new_col:
                            df_volcano.rename(columns={old_col: new_col}, inplace=True)
                    
                    # Ensure required columns exist
                    required_cols = ['volcano_name', 'latitude', 'longitude', 'elevation', 'country']
                    missing_cols = [col for col in required_cols if col not in df_volcano.columns]
                    
                    if missing_cols:
                        st.warning(f"Missing required columns in volcano dataset: {', '.join(missing_cols)}")
                        # Create minimal columns if missing
                        for col in missing_cols:
                            if col in ['latitude', 'longitude', 'elevation']:
                                df_volcano[col] = 0
                            else:
                                df_volcano[col] = "Unknown"
                    
                    # Clean numeric data
                    for col in ['latitude', 'longitude', 'elevation']:
                        if col in df_volcano.columns:
                            df_volcano[col] = pd.to_numeric(df_volcano[col], errors='coerce')
                    
                    # Handle last eruption dates
                    if 'last_eruption' in df_volcano.columns:
                        # Convert various date formats to standardized years
                        df_volcano['last_eruption'] = df_volcano['last_eruption'].astype(str)
                        df_volcano['last_eruption_year'] = df_volcano['last_eruption'].str.extract(r'(\d{4})', expand=False)
                        df_volcano['last_eruption_year'] = pd.to_numeric(df_volcano['last_eruption_year'], errors='coerce')
                        
                        # Create activity recency categories
                        current_year = datetime.now().year
                        conditions = [
                            df_volcano['last_eruption_year'] >= current_year - 10,
                            df_volcano['last_eruption_year'].between(current_year - 100, current_year - 11),
                            df_volcano['last_eruption_year'].between(current_year - 1000, current_year - 101),
                            df_volcano['last_eruption_year'] < current_year - 1000
                        ]
                        choices = ['Recent (Last 10 years)', 'Historical (11-100 years)', 
                                  'Ancient (101-1000 years)', 'Prehistoric (>1000 years)']
                        df_volcano['activity_category'] = np.select(conditions, choices, default='Unknown')
                    else:
                        df_volcano['activity_category'] = 'Unknown'
                    
                    # Create volcano type categories if available
                    if 'volcano_type' in df_volcano.columns:
                        # Standardize volcano types
                        df_volcano['volcano_type'] = df_volcano['volcano_type'].astype(str).str.lower()
                        
                        # Categorize volcano types
                        type_conditions = [
                            df_volcano['volcano_type'].str.contains('strato|composite', case=False, na=False),
                            df_volcano['volcano_type'].str.contains('shield', case=False, na=False),
                            df_volcano['volcano_type'].str.contains('caldera', case=False, na=False),
                            df_volcano['volcano_type'].str.contains('cinder|scoria', case=False, na=False),
                            df_volcano['volcano_type'].str.contains('submarine', case=False, na=False),
                        ]
                        type_choices = ['Stratovolcano', 'Shield', 'Caldera', 'Cinder cone', 'Submarine']
                        df_volcano['volcano_type_category'] = np.select(type_conditions, type_choices, default='Other')
                    else:
                        df_volcano['volcano_type_category'] = 'Unknown'
                    
                    # Ensure we have a status column
                    if 'status' not in df_volcano.columns:
                        df_volcano['status'] = 'Unknown'
                    
                    # Standardize status categories
                    status_conditions = [
                        df_volcano['status'].str.contains('active|erupting|historical', case=False, na=False),
                        df_volcano['status'].str.contains('dormant', case=False, na=False),
                        df_volcano['status'].str.contains('extinct', case=False, na=False)
                    ]
                    status_choices = ['Active', 'Dormant', 'Extinct']
                    df_volcano['status_category'] = np.select(status_conditions, status_choices, default='Unknown')
                    
                    # Remove volcanoes with invalid coordinates
                    df_volcano = df_volcano[
                        (df_volcano['latitude'].between(-90, 90)) & 
                        (df_volcano['longitude'].between(-180, 180))
                    ].reset_index(drop=True)
                    
                    # Create volcano subtabs
                    volcano_tabs = st.tabs([
                        "📊 Volcano Overview", 
                        "🗺️ Global Distribution", 
                        "🔥 Activity Analysis",
                        "⚡ Earthquake-Volcano Relationship"
                    ])
                    
                    # Tab 1: Volcano Overview
                    with volcano_tabs[0]:
                        st.subheader("Global Volcano Dataset Overview")
                        
                        # Display basic metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Volcanoes", f"{len(df_volcano):,}")
                        with col2:
                            if 'status_category' in df_volcano.columns:
                                active_count = df_volcano[df_volcano['status_category'] == 'Active'].shape[0]
                                st.metric("Active Volcanoes", f"{active_count:,}")
                        with col3:
                            if 'elevation' in df_volcano.columns:
                                max_elev = df_volcano['elevation'].max()
                                st.metric("Highest Elevation", f"{max_elev:,.0f} m")
                        with col4:
                            if 'last_eruption_year' in df_volcano.columns:
                                recent_eruptions = df_volcano[df_volcano['last_eruption_year'] >= 2000].shape[0]
                                st.metric("Eruptions Since 2000", f"{recent_eruptions:,}")
                        
                        # Distribution by volcano type
                        if 'volcano_type_category' in df_volcano.columns:
                            st.subheader("Volcano Types Distribution")
                            
                            type_counts = df_volcano['volcano_type_category'].value_counts()
                            fig_types = px.pie(
                                values=type_counts.values,
                                names=type_counts.index,
                                title="Distribution by Volcano Type",
                                color_discrete_sequence=px.colors.sequential.Inferno,
                                hole=0.4
                            )
                            fig_types.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_types, use_container_width=True)
                        
                        # Distribution by activity status
                        if 'status_category' in df_volcano.columns and 'activity_category' in df_volcano.columns:
                            col1, col2 = st.columns(2)
                            
                            # Definir choices para la ordenación de categorías
                            choices = ['Recent (Last 10 years)', 'Historical (11-100 years)', 
                                      'Ancient (101-1000 years)', 'Prehistoric (>1000 years)']
                            
                            with col1:
                                status_counts = df_volcano['status_category'].value_counts()
                                fig_status = px.bar(
                                    x=status_counts.index, 
                                    y=status_counts.values,
                                    labels={'x': 'Status', 'y': 'Count'},
                                    title='Volcano Status Distribution',
                                    color=status_counts.values,
                                    color_continuous_scale=px.colors.sequential.Inferno
                                )
                                st.plotly_chart(fig_status, use_container_width=True)
                            
                            with col2:
                                activity_counts = df_volcano['activity_category'].value_counts()
                                fig_activity = px.bar(
                                    x=activity_counts.index, 
                                    y=activity_counts.values,
                                    labels={'x': 'Last Eruption', 'y': 'Count'},
                                    title='Last Eruption Recency',
                                    color=activity_counts.values,
                                    color_continuous_scale=px.colors.sequential.Inferno
                                )
                                fig_activity.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': choices})
                                st.plotly_chart(fig_activity, use_container_width=True)
                        
                        # Top countries by volcano count
                        if 'country' in df_volcano.columns:
                            st.subheader("Countries with Most Volcanoes")
                            country_counts = df_volcano['country'].value_counts().reset_index().head(15)
                            country_counts.columns = ['Country', 'Number of Volcanoes']
                            
                            fig_countries = px.bar(
                                country_counts,
                                y='Country',
                                x='Number of Volcanoes',
                                orientation='h',
                                color='Number of Volcanoes',
                                title='Top 15 Countries by Volcano Count',
                                color_continuous_scale=px.colors.sequential.Inferno
                            )
                            fig_countries.update_layout(yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig_countries, use_container_width=True)
                        
                        # Elevation distribution
                        if 'elevation' in df_volcano.columns:
                            st.subheader("Elevation Distribution")
                            
                            fig_elev = px.histogram(
                                df_volcano,
                                x='elevation',
                                nbins=50,
                                title='Volcano Elevation Distribution',
                                color_discrete_sequence=['rgba(255, 102, 0, 0.8)']
                            )
                            fig_elev.update_layout(
                                xaxis_title='Elevation (meters)',
                                yaxis_title='Number of Volcanoes'
                            )
                            st.plotly_chart(fig_elev, use_container_width=True)
                            
                            # Elevation by volcano type
                            if 'volcano_type_category' in df_volcano.columns:
                                fig_elev_type = px.box(
                                    df_volcano,
                                    x='volcano_type_category', 
                                    y='elevation',
                                    title='Elevation by Volcano Type',
                                    color='volcano_type_category',
                                    color_discrete_sequence=px.colors.sequential.Inferno
                                )
                                fig_elev_type.update_layout(
                                    xaxis_title='Volcano Type',
                                    yaxis_title='Elevation (meters)',
                                    showlegend=False
                                )
                                st.plotly_chart(fig_elev_type, use_container_width=True)
                    
                    # Tab 2: Global Distribution
                    with volcano_tabs[1]:
                        st.subheader("Global Volcano Distribution")
                        
                        # Create a color map for volcano types
                        volcano_type_colors = {
                            'Stratovolcano': '#FF5733',
                            'Shield': '#DAF7A6',
                            'Caldera': '#FFC300',
                            'Cinder cone': '#C70039',
                            'Submarine': '#1E8BC3',
                            'Other': '#8E44AD',
                            'Unknown': '#7B7D7D'
                        }
                        
                        # Use status for color if type is not available
                        color_column = 'volcano_type_category' if 'volcano_type_category' in df_volcano.columns else 'status_category'
                        
                        # Create the map
                        fig_volcano_map = px.scatter_geo(
                            df_volcano,
                            lat='latitude',
                            lon='longitude',
                            color=color_column,
                            hover_name='volcano_name',
                            hover_data={
                                'country': True,
                                'elevation': ':.0f m',
                                'status_category': True,
                                'last_known': True,
                                'latitude': False,
                                'longitude': False
                            },
                            title='Global Volcano Distribution',
                            size_max=15,
                            projection='natural earth'
                        )
                        
                        # Customize marker appearance
                        fig_volcano_map.update_traces(
                            marker=dict(
                                symbol='triangle-up',
                                size=10,
                                opacity=0.8,
                                line=dict(width=1, color='white')
                            )
                        )
                        
                        # Update map layout to match the earthquake map style
                        fig_volcano_map.update_layout(
                            margin={"r": 0, "t": 50, "l": 0, "b": 0},
                            height=600,
                            geo=dict(
                                showland=True,
                                landcolor="#d2b48c",
                                showocean=True,
                                oceancolor="#002244",
                                showcountries=True,
                                countrycolor="white",
                                showcoastlines=True,
                                coastlinecolor="white",
                                bgcolor="black",
                                projection_scale=1
                            ),
                            paper_bgcolor="black",
                            plot_bgcolor="black",
                            title_font=dict(size=20, color='white')
                        )
                        
                        st.plotly_chart(fig_volcano_map, use_container_width=True)
                        
                        # Ring of Fire highlight map
                        st.subheader("Pacific Ring of Fire")
                        
                        # Define Ring of Fire boundaries (approximate)
                        def is_in_ring_of_fire(lat, lon):
                            # Pacific rim coordinates (very approximate)
                            if (lon > 120 or lon < -60) and (lat > -60 and lat < 70):
                                return 'Ring of Fire'
                            else:
                                return 'Other Regions'
                        
                        # Add Ring of Fire classification
                        df_volcano['tectonic_zone'] = df_volcano.apply(
                            lambda x: is_in_ring_of_fire(x['latitude'], x['longitude']), axis=1
                        )
                        
                        # Create focused map for Ring of Fire
                        ring_of_fire_map = px.scatter_geo(
                            df_volcano,
                            lat='latitude',
                            lon='longitude',
                            color='tectonic_zone',
                            hover_name='volcano_name',
                            hover_data={
                                'country': True,
                                'elevation': ':.0f m',
                                'status_category': True,
                                'last_known': ':.0f m',  # Usa la columna correcta
                                'latitude': False,
                                'longitude': False
                            },
                            labels={'last_known': 'Last Known'},
                            title='Volcanoes in the Pacific Ring of Fire',
                            color_discrete_map={
                                'Ring of Fire': '#FF4500',  # Bright orange-red
                                'Other Regions': '#707070'  # Gray
                            },
                            size_max=15,
                            projection='orthographic'  # 3D-like globe projection
                        )
                        
                        # Set initial view to Pacific
                        ring_of_fire_map.update_geos(
                            projection_rotation=dict(lon=-150, lat=30, roll=0),
                            showcountries=True
                        )
                        
                        # Customize marker appearance
                        ring_of_fire_map.update_traces(
                            marker=dict(
                                symbol='triangle-up',
                                size=8,
                                opacity=0.8,
                                line=dict(width=1, color='white')
                            )
                        )
                        
                        # Style the map
                        ring_of_fire_map.update_layout(
                            height=600,
                            geo=dict(
                                showland=True,
                                landcolor="#552211",  # Dark brown land
                                showocean=True,
                                oceancolor="#003366",  # Dark blue ocean
                                showcountries=True,
                                countrycolor="white",
                                showcoastlines=True,
                                coastlinecolor="white",
                                bgcolor="black"
                            ),
                            paper_bgcolor="black",
                            plot_bgcolor="black"
                        )
                        
                        st.plotly_chart(ring_of_fire_map, use_container_width=True)
                        
                        # Volcano density heatmap
                        st.subheader("Volcano Density Heatmap")
                        
                        fig_volcano_heat = px.density_mapbox(
                            df_volcano,
                            lat='latitude',
                            lon='longitude',
                            z=df_volcano['elevation'] if 'elevation' in df_volcano.columns else None,
                            radius=20,
                            center=dict(lat=0, lon=0),
                            zoom=0,
                            mapbox_style="carto-darkmatter",
                            title='Global Volcano Density',
                            color_continuous_scale='Inferno'
                        )
                        
                        fig_volcano_heat.update_layout(height=600)
                        st.plotly_chart(fig_volcano_heat, use_container_width=True)
                        
                        # Educational information
                        with st.expander("📚 About Volcanic Distribution Patterns"):
                            st.markdown("""
                            ### Global Volcanic Distribution
                            
                            Volcanoes are not randomly distributed around the world. Their locations are directly 
                            tied to tectonic plate boundaries and hotspots:
                            
                            1. **Pacific Ring of Fire**: Contains approximately 75% of the world's active volcanoes,
                               following tectonic plate boundaries around the Pacific Ocean.
                            
                            2. **Mid-Ocean Ridges**: Underwater mountain ranges formed by divergent tectonic plates,
                               where magma rises to create new oceanic crust.
                            
                            3. **Continental Rift Zones**: Areas where the Earth's crust is being pulled apart,
                               allowing magma to rise to the surface.
                            
                            4. **Hotspots**: Volcanic regions created by mantle plumes that remain relatively fixed
                               while tectonic plates move over them (e.g., Hawaiian Islands).
                            
                            5. **Subduction Zones**: Where one tectonic plate is forced beneath another, creating
                               conditions for explosive volcanism (e.g., Andes Mountains).
                            
                            The visualization above clearly shows these patterns, particularly highlighting the 
                            Pacific Ring of Fire and other major tectonic boundaries.
                            """)
                    
                    # Tab 3: Activity Analysis
                    with volcano_tabs[2]:
                        st.subheader("Volcanic Activity Analysis")
                        
                        # Check if we have eruption data
                        has_eruption_data = 'last_eruption_year' in df_volcano.columns and df_volcano['last_eruption_year'].notna().sum() > 10
                        
                        if has_eruption_data:
                            # Eruption timeline
                            recent_cutoff = 1500  # Show eruptions since 1500 CE
                            recent_eruptions = df_volcano[df_volcano['last_eruption_year'] >= recent_cutoff].copy()
                            
                            if len(recent_eruptions) > 0:
                                # Sort by eruption year
                                recent_eruptions = recent_eruptions.sort_values('last_eruption_year')
                                
                                # Create a timeline of eruptions
                                fig_timeline = px.scatter(
                                    recent_eruptions,
                                    x='last_eruption_year',
                                    y='volcano_name',
                                    color='volcano_type_category' if 'volcano_type_category' in recent_eruptions.columns else 'country',
                                    size='elevation' if 'elevation' in recent_eruptions.columns else None,
                                    hover_data=['country', 'elevation', 'last_eruption'],
                                    title=f'Timeline of Volcanic Eruptions (since {recent_cutoff})',
                                    labels={'last_eruption_year': 'Year of Eruption', 'volcano_name': 'Volcano'}
                                )
                                
                                fig_timeline.update_layout(
                                    xaxis_title="Year",
                                    yaxis_title="Volcano",
                                    height=800,
                                    yaxis={'categoryorder': 'trace'}
                                )
                                
                                st.plotly_chart(fig_timeline, use_container_width=True)
                                
                                # Historical eruption frequency
                                st.subheader("Historical Eruption Frequency")
                                
                                # Create century bins
                                century_bins = list(range(1000, 2100, 100))
                                century_labels = [f"{b}-{b+99}" for b in century_bins[:-1]]
                                
                                # Bin eruptions by century
                                recent_eruptions['century'] = pd.cut(
                                    recent_eruptions['last_eruption_year'], 
                                    bins=century_bins, 
                                    labels=century_labels, 
                                    right=False
                                )
                                
                                # Count eruptions by century
                                century_counts = recent_eruptions['century'].value_counts().sort_index()
                                
                                fig_centuries = px.bar(
                                    x=century_counts.index,
                                    y=century_counts.values,
                                    labels={'x': 'Century', 'y': 'Number of Eruptions'},
                                    title='Volcanic Eruptions by Century',
                                    color=century_counts.values,
                                    color_continuous_scale='Inferno'
                                )
                                
                                st.plotly_chart(fig_centuries, use_container_width=True)
                                
                                # Note about data bias
                                st.info("""
                                **Note on Historical Data**: The apparent increase in eruptions in recent centuries 
                                is largely due to better record-keeping and reporting, not necessarily an actual 
                                increase in volcanic activity.
                                """)
                            
                        # Active volcanoes
                        if 'status_category' in df_volcano.columns:
                            st.subheader("Currently Active Volcanoes")
                            
                            # Filter for active volcanoes
                            active_volcanoes = df_volcano[df_volcano['status_category'] == 'Active'].copy()
                            
                            if len(active_volcanoes) > 0:
                                # Group by region/country
                                if 'region' in active_volcanoes.columns:
                                    active_by_region = active_volcanoes['region'].value_counts().reset_index().head(15)
                                    active_by_region.columns = ['Region', 'Number of Active Volcanoes']
                                else:
                                    active_by_region = active_volcanoes['country'].value_counts().reset_index().head(15)
                                    active_by_region.columns = ['Country', 'Number of Active Volcanoes']
                                
                                # Plot
                                fig_active = px.bar(
                                    active_by_region,
                                    y=active_by_region.columns[0],  # Region or Country
                                    x='Number of Active Volcanoes',
                                    orientation='h',
                                    color='Number of Active Volcanoes',
                                    title=f'Top 15 {active_by_region.columns[0]}s by Active Volcanoes',
                                    color_continuous_scale='Inferno'
                                )
                                fig_active.update_layout(yaxis={'categoryorder': 'total ascending'})
                                st.plotly_chart(fig_active, use_container_width=True)
                                
                                # Table of most notable active volcanoes
                                st.subheader("Notable Active Volcanoes")
                                
                                # Select columns to display
                                display_cols = [
                                    'volcano_name', 'country', 'elevation', 'last_eruption', 
                                    'volcano_type_category' if 'volcano_type_category' in active_volcanoes.columns else 'status'
                                ]
                                display_cols = [col for col in display_cols if col in active_volcanoes.columns]
                                
                                # Sort by elevation if available, otherwise by name
                                if 'elevation' in active_volcanoes.columns:
                                    notable_volcanoes = active_volcanoes.sort_values('elevation', ascending=False).head(20)
                                else:
                                    notable_volcanoes = active_volcanoes.sort_values('volcano_name').head(20)
                                
                                # Display table
                                st.dataframe(notable_volcanoes[display_cols], use_container_width=True)
                        
                        # Volcano types and characteristics
                        if 'volcano_type_category' in df_volcano.columns and 'elevation' in df_volcano.columns:
                            st.subheader("Volcano Types and Characteristics")
                            
                            # Create relationship plots
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Elevation by volcano type
                                fig_elev_box = px.box(
                                    df_volcano,
                                    x='volcano_type_category',
                                    y='elevation',
                                    color='volcano_type_category',
                                    title='Elevation Distribution by Volcano Type',
                                    labels={'volcano_type_category': 'Type', 'elevation': 'Elevation (m)'}
                                )
                                st.plotly_chart(fig_elev_box, use_container_width=True)
                            
                            with col2:
                                # Activity category by type if available
                                if 'activity_category' in df_volcano.columns:
                                    # Create a crosstab of type and activity
                                    type_activity = pd.crosstab(
                                        df_volcano['volcano_type_category'],
                                        df_volcano['activity_category']
                                    )
                                    
                                    # Convert to percentages
                                    type_activity_pct = type_activity.div(type_activity.sum(axis=1), axis=0) * 100
                                    
                                    # Plot
                                    fig_type_activity = px.imshow(
                                        type_activity_pct,
                                        text_auto='.1f',
                                        labels=dict(x="Activity Category", y="Volcano Type", color="Percentage"),
                                        title="Relationship Between Volcano Type and Recent Activity",
                                        color_continuous_scale='Inferno'
                                    )
                                    fig_type_activity.update_traces(
                                        hovertemplate="Type: %{y}<br>Activity: %{x}<br>Percentage: %{z:.1f}%<extra></extra>"
                                    )
                                    st.plotly_chart(fig_type_activity, use_container_width=True)
                        
                        # Educational content
                        with st.expander("📚 Learn About Different Volcano Types"):
                            st.markdown("""
                            ### Major Types of Volcanoes
                            
                            Different types of volcanoes are characterized by their shape, size, eruption style, and the materials they're built from:
                            
                            #### Stratovolcanoes (Composite Volcanoes)
                            - **Characteristics**: Steep-sided, symmetrical cones built of alternating layers of lava flows, ash, and blocks of stone
                            - **Examples**: Mount Fuji (Japan), Mount St. Helens (USA), Mount Vesuvius (Italy)
                            - **Eruption style**: Often explosive due to highly viscous lava
                            
                            #### Shield Volcanoes
                            - **Characteristics**: Broad, gently sloping domes built from fluid basaltic lava flows
                            - **Examples**: Mauna Loa and Kilauea (Hawaii), Nyamuragira (DR Congo)
                            - **Eruption style**: Generally non-explosive with flowing lava
                            
                            #### Calderas
                            - **Characteristics**: Large volcanic depressions formed after a magma chamber empties and collapses
                            - **Examples**: Yellowstone (USA), Lake Toba (Indonesia), Santorini (Greece)
                            - **Eruption style**: Can produce some of the most violent eruptions
                            
                            #### Cinder Cones
                            - **Characteristics**: Small, steep-sided cones built from ejected lava fragments
                            - **Examples**: Paricutin (Mexico), Sunset Crater (USA)
                            - **Eruption style**: Moderately explosive, building up through accumulation of ejected material
                            
                            #### Submarine Volcanoes
                            - **Characteristics**: Form underwater, sometimes creating islands when they grow large enough
                            - **Examples**: Loihi (Hawaii), Kick 'em Jenny (Caribbean)
                            - **Eruption style**: Modified by water pressure and cooling effects
                            """)
                    
                    # Tab 4: Earthquake-Volcano Relationship
                    with volcano_tabs[3]:
                        st.subheader("Exploring the Relationship Between Earthquakes and Volcanoes")
                        
                        st.markdown("""
                        This section analyzes the spatial and causal relationships between seismic activity and volcanic phenomena.
                        Both earthquakes and volcanoes are manifestations of Earth's tectonic processes, often occurring
                        in similar geographical regions.
                        """)
                        
                        # Distance calculation function
                        @st.cache_data(ttl=3600)
                        def calculate_distance(lat1, lon1, lat2, lon2):
                            # Haversine formula to calculate distance between two points on Earth
                            R = 6371  # Earth radius in km
                            dLat = np.radians(lat2 - lat1)
                            dLon = np.radians(lon2 - lon1)
                            a = np.sin(dLat/2) * np.sin(dLat/2) + \
                                np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * \
                                np.sin(dLon/2) * np.sin(dLon/2)
                            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                            distance = R * c
                            return distance
                        
                        # Create combined map of earthquakes and volcanoes
                        st.subheader("Combined Distribution Map")
                        
                        # Prepare data for plotting
                        fig_combined = go.Figure()
                        
                        # Add earthquakes as circles (if we have earthquake data)
                        if filtered_df is not None and len(filtered_df) > 0:
                            for category in filtered_df['magnitud_categoria'].unique():
                                category_df = filtered_df[filtered_df['magnitud_categoria'] == category]
                                fig_combined.add_trace(go.Scattergeo(
                                    lon=category_df['longitude'],
                                    lat=category_df['latitude'],
                                    text=category_df['place'],
                                    mode='markers',
                                    name=f'Earthquake - {category}',
                                    marker=dict(
                                        size=ensure_positive(category_df['mag']*1.5),
                                        color=magnitude_colors.get(category, '#FFA500'),
                                        opacity=0.7,
                                        line=dict(width=1, color='white')
                                    ),
                                    hovertemplate="<b>%{text}</b><br>Magnitude: %{marker.size:.1f}<br>Location: (%{lat:.2f}, %{lon:.2f})<extra></extra>"
                                ))
                        
                        # Add volcanoes as triangles
                        volcano_marker_size = 8
                        fig_combined.add_trace(go.Scattergeo(
                            lon=df_volcano['longitude'],
                            lat=df_volcano['latitude'],
                            text=df_volcano['volcano_name'],
                            mode='markers',
                            name='Volcano',
                            marker=dict(
                                size=volcano_marker_size,
                                symbol='triangle-up',
                                color='red',
                                opacity=0.8,
                                line=dict(width=1, color='white')
                            ),
                            hovertemplate="<b>Volcano: %{text}</b><br>Location: (%{lat:.2f}, %{lon:.2f})<extra></extra>"
                        ))
                        
                        # Add tectonic plate boundaries if available
                        # This would require additional data source
                        
                        # Update map layout
                        fig_combined.update_layout(
                            title="Global Distribution of Earthquakes and Volcanoes",
                            geo=dict(
                                showland=True,
                                landcolor="#d2b48c",
                                showocean=True,
                                oceancolor="#002244",
                                showcountries=True,
                                countrycolor="white",
                                showcoastlines=True,
                                coastlinecolor="white",
                                bgcolor="black",
                                projection_type="natural earth"
                            ),
                            height=700,
                            paper_bgcolor="black",
                            plot_bgcolor="black",
                            legend=dict(
                                itemsizing="constant",
                                bgcolor="rgba(0,0,0,0.5)",
                                bordercolor="white",
                                borderwidth=1
                            )
                        )
                        
                        st.plotly_chart(fig_combined, use_container_width=True)
                        
                        # Proximity Analysis
                        if filtered_df is not None and len(filtered_df) > 0:
                            st.subheader("Proximity Analysis")
                            
                            st.markdown("""
                            This analysis examines the spatial relationship between earthquakes and volcanoes,
                            calculating the distance from each earthquake to the nearest volcano.
                            """)
                            
                            # Parameters for analysis
                            distance_threshold = st.slider(
                                "Distance threshold for earthquake-volcano relationship (km)",
                                min_value=50,
                                max_value=500,
                                value=200,
                                step=50
                            )
                            
                            # Calculate distances (computationally intensive, so limit to recent earthquakes)
                            max_quakes_to_analyze = 1000  # Limit for performance
                            recent_earthquakes = filtered_df.sort_values('time', ascending=False).head(max_quakes_to_analyze)
                            
                            with st.spinner("Calculating proximity relationships..."):
                                # Prepare results container
                                proximity_results = []
                                
                                # Analyze each earthquake
                                for idx, quake in recent_earthquakes.iterrows():
                                    # Find distance to all volcanoes
                                    distances = []
                                    for _, volcano in df_volcano.iterrows():
                                        dist = calculate_distance(
                                            quake['latitude'], quake['longitude'],
                                            volcano['latitude'], volcano['longitude']
                                        )
                                        distances.append((dist, volcano['volcano_name'], volcano['country']))
                                    
                                    # Find nearest volcano
                                    nearest = min(distances, key=lambda x: x[0])
                                    
                                    proximity_results.append({
                                        'earthquake_time': quake['time'],
                                        'earthquake_mag': quake['mag'],
                                        'earthquake_place': quake['place'],
                                        'nearest_volcano': nearest[1],
                                        'volcano_country': nearest[2],
                                        'distance_km': nearest[0]
                                    })
                                
                                # Convert to DataFrame
                                proximity_df = pd.DataFrame(proximity_results)
                                
                                # Calculate statistics
                                near_volcano_count = sum(proximity_df['distance_km'] <= distance_threshold)
                                near_volcano_percentage = (near_volcano_count / len(proximity_df)) * 100
                                
                                # Display stats
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        "Earthquakes within threshold distance of a volcano", 
                                        f"{near_volcano_count} of {len(proximity_df)}"
                                    )
                                with col2:
                                    st.metric(
                                        "Percentage", 
                                        f"{near_volcano_percentage:.1f}%"
                                    )
                                
                                # Distribution of distances
                                fig_distances = px.histogram(
                                    proximity_df,
                                    x='distance_km',
                                    nbins=50,
                                    title='Distribution of Distances from Earthquakes to Nearest Volcano',
                                    labels={'distance_km': 'Distance (km)', 'count': 'Number of Earthquakes'}
                                )
                                fig_distances.add_vline(
                                    x=distance_threshold, 
                                    line_dash="dash", 
                                    annotation_text=f"Threshold: {distance_threshold} km"
                                )
                                st.plotly_chart(fig_distances, use_container_width=True)
                                
                                # Scatter plot of magnitude vs distance
                                fig_mag_dist = px.scatter(
                                    proximity_df,
                                    x='distance_km',
                                    y='earthquake_mag',
                                    color='earthquake_mag',
                                    size=ensure_positive(proximity_df['earthquake_mag']),
                                    hover_data=['earthquake_place', 'nearest_volcano'],
                                    title='Earthquake Magnitude vs Distance to Nearest Volcano',
                                    labels={
                                        'distance_km': 'Distance to Nearest Volcano (km)',
                                        'earthquake_mag': 'Magnitude'
                                    },
                                    color_continuous_scale='Plasma'
                                )
                                fig_mag_dist.add_vline(
                                    x=distance_threshold, 
                                    line_dash="dash", 
                                    annotation_text=f"Threshold: {distance_threshold} km"
                                )
                                st.plotly_chart(fig_mag_dist, use_container_width=True)
                                
                                # Top earthquake-volcano pairs
                                st.subheader("Closest Earthquake-Volcano Pairs")
                                
                                # Get earthquakes close to volcanoes
                                close_pairs = proximity_df[proximity_df['distance_km'] <= distance_threshold].sort_values('distance_km')
                                if len(close_pairs) > 0:
                                    # Format for display
                                    display_pairs = close_pairs[['earthquake_time', 'earthquake_mag', 'earthquake_place', 
                                                              'nearest_volcano', 'volcano_country', 'distance_km']].head(20)
                                    display_pairs.columns = ['Time', 'Magnitude', 'Earthquake Location', 
                                                            'Nearest Volcano', 'Volcano Country', 'Distance (km)']
                                    
                                    st.dataframe(display_pairs, use_container_width=True)
                                else:
                                    st.info(f"No earthquakes found within {distance_threshold} km of a volcano.")
                        
                        # Temporal correlation analysis
                        st.subheader("Temporal Correlation Analysis")
                        
                        st.markdown("""
                        This section explores potential temporal relationships between earthquake activity and volcanic eruptions.
                        Note that definitive causal relationships are complex and often require more detailed geophysical data.
                        """)
                        
                        # Simulated correlation data (since we don't have actual eruption dates)
                        # This would be replaced with real data analysis if eruption dates were available
                        
                        # Create synthetic eruption-earthquake correlation data for demonstration
                        np.random.seed(42)  # For reproducibility
                        
                        # Time periods
                        periods = ['1 week before', '1 day before', 'Same day', '1 day after', '1 week after', '1 month after']
                        
                        # Correlation values (with higher values close to eruption)
                        correlations = [0.15, 0.35, 0.65, 0.45, 0.25, 0.15]
                        
                        # Standard deviations
                        std_devs = [0.05, 0.08, 0.12, 0.10, 0.07, 0.05]
                        
                        # Create the chart
                        fig_temporal = go.Figure()
                        
                        fig_temporal.add_trace(go.Scatter(
                            x=periods,
                            y=correlations,
                            mode='lines+markers',
                            marker=dict(size=10, color='orange'),
                            line=dict(width=2, color='orange'),
                            name='Correlation'
                        ))
                        
                        # Add error bars
                        fig_temporal.add_trace(go.Scatter(
                            x=periods,
                            y=[c + s for c, s in zip(correlations, std_devs)],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False
                        ))
                        
                        fig_temporal.add_trace(go.Scatter(
                            x=periods,
                            y=[c - s for c, s in zip(correlations, std_devs)],
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(255, 165, 0, 0.2)',
                            showlegend=False
                        ))
                        
                        fig_temporal.update_layout(
                            title='Simulated Temporal Correlation Between Volcanic Eruptions and Earthquake Frequency',
                            xaxis_title='Time Relative to Eruption',
                            yaxis_title='Correlation Coefficient',
                            yaxis=dict(range=[0, 1]),
                            height=500
                        )
                        
                        st.plotly_chart(fig_temporal, use_container_width=True)
                        
                        st.warning("""
                        ⚠️ **Note**: This chart uses simulated data for educational purposes. In reality, the relationship
                        between earthquakes and volcanic eruptions is complex and varies significantly depending on
                        volcano type, tectonic setting, and other geological factors.
                        """)
                        
                        # Educational content on volcano-earthquake relationships
                        with st.expander("📚 The Science Behind Earthquake-Volcano Relationships"):
                            st.markdown("""
                            ### How Earthquakes and Volcanoes are Related
                            
                            Earthquakes and volcanoes are both manifestations of Earth's dynamic geologic processes, and they
                            often occur in similar regions due to shared underlying causes:
                            
                            #### Shared Tectonic Settings
                            
                            - **Plate Boundaries**: Most earthquakes and volcanoes occur along tectonic plate boundaries,
                              especially subduction zones where one plate slides beneath another.
                              
                            - **Magma Movement**: The movement of magma beneath volcanoes can trigger small earthquakes,
                              known as volcanic tremors.
                            
                            #### Types of Relationships
                            
                            1. **Volcanic Earthquakes**: Directly caused by volcanic processes such as magma movement,
                               gas pressure changes, or the fracturing of rock by magmatic heat and pressure.
                            
                            2. **Trigger Relationships**: Large tectonic earthquakes can sometimes trigger volcanic activity
                               by shaking loose magma chambers or opening new pathways for magma to reach the surface.
                            
                            3. **Shared Cause**: Both phenomena may be driven by the same underlying tectonic or mantle processes
                               without directly causing each other.
                            
                            #### Scientific Evidence
                            
                            - Studies have found statistical correlations between significant earthquakes and subsequent
                              eruptions at some volcanoes within days to weeks.
                              
                            - Monitoring both seismic activity and volcanic indicators (gas emissions, ground deformation)
                              provides the most comprehensive approach to understanding these complex relationships.
                            
                            #### Case Studies
                            
                            - The 1992 Landers earthquake in California coincided with increased activity at several nearby
                              volcanic systems.
                              
                            - After the 2011 Tohoku earthquake in Japan, scientists observed increased volcanic activity
                              in some areas of Japan in the following years.
                            
                            The relationship between earthquakes and volcanoes continues to be an active area of research
                            in geophysics and volcanology.
                            """)
                else:
                    st.error("Failed to load or process the volcano dataset properly.")

            with geo_tabs[1]:
                st.subheader("Seismic Activity Heat Map")
                st.markdown("This heat map shows areas with higher concentration of seismic activity. Brighter areas indicate higher density of events.")
                fig_heat = px.density_mapbox(
                    filtered_df,
                    lat="latitude",
                    lon="longitude",
                    z=ensure_positive(filtered_df['mag']),
                    radius=10,
                    center=dict(lat=filtered_df['latitude'].mean(), lon=filtered_df['longitude'].mean()),
                    zoom=1,
                    mapbox_style="open-street-map",
                    opacity=0.8
                )
                fig_heat.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=600)
                st.plotly_chart(fig_heat, use_container_width=True)
                st.subheader("Significant Events (Magnitude ≥ 4.0)")
                strong_events = filtered_df[filtered_df['mag'] >= 4.0].sort_values(by='mag', ascending=False)
                if not strong_events.empty:
                    st.dataframe(
                        strong_events[['time', 'place', 'mag', 'depth', 'type']],
                        use_container_width=True
                    )
                else:
                    st.info("No events with magnitude ≥ 4.0 in the selected range.")

                    # Tab 3: Temporal Analysis
        
            with geo_tabs[2]:
                st.subheader("Geographic Cluster Analysis")
                st.markdown("This analysis identifies groups of earthquakes that might be geographically related. It uses the DBSCAN algorithm which groups events based on their spatial proximity.")
                if len(filtered_df) > 10:
                    cluster_df = filtered_df[['latitude', 'longitude']].copy()
                    scaler = StandardScaler()
                    cluster_data = scaler.fit_transform(cluster_df)
                    col1, col2 = st.columns(2)
                    with col1:
                        eps_distance = st.slider(
                            "Maximum distance between events to consider them neighbors (eps)",
                            min_value=0.05,
                            max_value=1.0,
                            value=0.2,
                            step=0.05
                        )
                    with col2:
                        min_samples = st.slider(
                            "Minimum number of events to form a cluster",
                            min_value=2,
                            max_value=20,
                            value=5,
                            step=1
                        )
                    with st.spinner("Performing cluster analysis..."):
                        dbscan = DBSCAN(eps=eps_distance, min_samples=min_samples)
                        cluster_result = dbscan.fit_predict(cluster_data)
                        filtered_df.loc[:, 'cluster'] = cluster_result

                    n_clusters = len(set(filtered_df['cluster'])) - (1 if -1 in filtered_df['cluster'] else 0)
                    n_noise = list(filtered_df['cluster']).count(-1)
                    col1, col2 = st.columns(2)
                    col1.metric("Number of identified clusters", n_clusters)
                    col2.metric("Ungrouped events (noise)", n_noise)

                    # Assign geographical names to clusters
                    cluster_names = {}
                    for cluster_id in sorted(set(filtered_df['cluster'])):
                        if cluster_id == -1:
                            cluster_names[cluster_id] = "No Cluster"
                        else:
                            # Get the most common region name in this cluster
                            region_mode = (
                                filtered_df[filtered_df['cluster'] == cluster_id]['region']
                                .mode()
                            )
                            if not region_mode.empty:
                                cluster_names[cluster_id] = f"{region_mode.iloc[0]}"
                            else:
                                cluster_names[cluster_id] = f"Cluster {cluster_id}"

                    filtered_df['cluster_name'] = filtered_df['cluster'].map(cluster_names)

                    # Assign colors: black for "No Cluster", others use Plotly default palette
                    unique_names = filtered_df['cluster_name'].unique().tolist()
                    color_map = {}
                    palette = px.colors.qualitative.Plotly
                    color_idx = 0
                    for name in unique_names:
                        if name == "No Cluster":
                            color_map[name] = "black"
                        else:
                            color_map[name] = palette[color_idx % len(palette)]
                            color_idx += 1

                    st.markdown("### Clusters Map")
                    fig_cluster = px.scatter_geo(
                        filtered_df,
                        lat="latitude",
                        lon="longitude",
                        color="cluster_name",
                        size=ensure_positive(filtered_df['mag']),
                        size_max=15,
                        hover_name="place",
                        hover_data={
                            "latitude": False,
                            "longitude": False,
                            "cluster_name": True,
                            "mag": ":.2f",
                            "depth": ":.2f km",
                            "time": True
                        },
                        color_discrete_map=color_map,
                        projection="natural earth"
                    )

                    # Create starry background effect
                    n_stars = 500
                    random_stars = {
                        'type': 'scattergeo',
                        'lon': np.random.uniform(-180, 180, n_stars),
                        'lat': np.random.uniform(-90, 90, n_stars),
                        'mode': 'markers',
                        'marker': {
                            'size': np.random.uniform(0.1, 1, n_stars),
                            'opacity': np.random.uniform(0.3, 0.8, n_stars),
                            'color': 'white'
                        },
                        'showlegend': False,
                        'hoverinfo': 'none'
                    }

                    # Add stars to the map
                    fig_cluster.add_trace(random_stars)

                    fig_cluster.update_layout(
                        margin={"r": 0, "t": 0, "l": 0, "b": 0},
                        height=500,
                        paper_bgcolor="black",
                        geo=dict(
                            showland=True,
                            landcolor="#d2b48c",  # Yellowish-brown for continents
                            showocean=True,
                            oceancolor="#00356B",  # Slightly darker blue for oceans
                            showcountries=True,
                            countrycolor="white",
                            showcoastlines=True,
                            coastlinecolor="white",
                            bgcolor="black"  # Black background for the map itself
                        )
                    )
                    st.plotly_chart(fig_cluster, use_container_width=True)
                else:
                    st.warning("Not enough data to perform cluster analysis with current filters.")


                    # --- Temporal Analysis Tab ---
                    with main_tabs[2]:
                        st.subheader("Temporal Pattern Analysis")
                        # Create tabs for different temporal analyses
                        temp_tab1, temp_tab2, temp_tab3 = st.tabs([
                            "Daily Evolution", 
                            "Weekly Patterns",
                            "Hourly Patterns"
                        ])
                        
                        # Tab 1: Daily Evolution
                        with temp_tab1:
                            st.subheader("Daily Evolution of Seismic Activity")
                            
                            # Group by day
                            try:
                                daily_counts = filtered_df.groupby('day').agg({
                                    'id': 'count',
                                    'mag': ['mean', 'max']
                                }).reset_index()
                                
                                daily_counts.columns = ['Date', 'Count', 'Average Magnitude', 'Maximum Magnitude']
                                daily_counts['Date'] = pd.to_datetime(daily_counts['Date'])
                                
                                # Create chart
                                fig_daily = go.Figure()
                                
                                # Add bars for event count
                                fig_daily.add_trace(go.Bar(
                                    x=daily_counts['Date'],
                                    y=daily_counts['Count'],
                                    name='Number of Events',
                                    marker_color='lightblue',
                                    opacity=0.7
                                ))
                                
                                # Add line for maximum magnitude
                                fig_daily.add_trace(go.Scatter(
                                    x=daily_counts['Date'],
                                    y=daily_counts['Maximum Magnitude'],
                                    name='Maximum Magnitude',
                                    mode='lines+markers',
                                    marker=dict(color='red', size=6),
                                    line=dict(width=2, dash='solid'),
                                    yaxis='y2'
                                ))
                                
                                # Add line for average magnitude
                                fig_daily.add_trace(go.Scatter(
                                    x=daily_counts['Date'],
                                    y=daily_counts['Average Magnitude'],
                                    name='Average Magnitude',
                                    mode='lines',
                                    marker=dict(color='orange'),
                                    line=dict(width=2, dash='dot'),
                                    yaxis='y2'
                                ))
                                
                                # Configure axes and layout
                                fig_daily.update_layout(
                                    title='Daily Evolution of Seismic Events',
                                    xaxis=dict(title='Date', tickformat='%d-%b'),
                                    yaxis=dict(title='Number of Events', side='left'),
                                    yaxis2=dict(
                                        title='Magnitude',
                                        side='right',
                                        overlaying='y',
                                        range=[0, max(daily_counts['Maximum Magnitude']) + 0.5]
                                    ),
                                    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
                                    hovermode='x unified',
                                    height=500
                                )
                                
                                st.plotly_chart(fig_daily, use_container_width=True)
                                
                                # Add trend analysis
                                if len(daily_counts) > 5:
                                    st.subheader("Trend Analysis")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # Calculate the trend of events per day
                                        x = np.arange(len(daily_counts))
                                        y = daily_counts['Count']
                                        z = np.polyfit(x, y, 1)
                                        p = np.poly1d(z)
                                        
                                        trend_direction = "increasing" if z[0] > 0 else "decreasing"
                                        trend_value = abs(z[0])
                                        
                                        st.metric(
                                            "Event Trend", 
                                            f"{trend_direction} ({trend_value:.2f} events/day)",
                                            delta=f"{trend_value:.2f}" if z[0] > 0 else f"-{trend_value:.2f}"
                                        )
                                    
                                    with col2:
                                        # Calculate the trend of magnitude per day
                                        x = np.arange(len(daily_counts))
                                        y = daily_counts['Average Magnitude']
                                        z_mag = np.polyfit(x, y, 1)
                                        p_mag = np.poly1d(z_mag)
                                        
                                        trend_direction_mag = "increasing" if z_mag[0] > 0 else "decreasing"
                                        trend_value_mag = abs(z_mag[0])
                                        
                                        st.metric(
                                            "Magnitude Trend", 
                                            f"{trend_direction_mag} ({trend_value_mag:.3f} mag/day)",
                                            delta=f"{trend_value_mag:.3f}" if z_mag[0] > 0 else f"-{trend_value_mag:.3f}"
                                        )
                            except Exception as e:
                                st.error(f"Error in daily evolution analysis: {e}")
                        
                        # Tab 2: Weekly Patterns
                        with temp_tab2:
                            try:
                                # Translate days of the week
                                filtered_df['day_name'] = filtered_df['day_of_week'].map(days_translation)
                                
                                # Sort days of the week correctly
                                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                                ordered_days = [days_translation[day] for day in days_order]
                                
                                # Group by day of the week
                                dow_data = filtered_df.groupby('day_name').agg({
                                    'id': 'count',
                                    'mag': ['mean', 'max']
                                }).reset_index()
                                
                                # Rename columns
                                dow_data.columns = ['Day', 'Count', 'Average Magnitude', 'Maximum Magnitude']
                                
                                # Order days
                                dow_data['Day_ordered'] = pd.Categorical(dow_data['Day'], categories=ordered_days, ordered=True)
                                dow_data = dow_data.sort_values('Day_ordered')
                                
                                # Create chart
                                fig_dow = px.bar(
                                    dow_data,
                                    x='Day',
                                    y='Count',
                                    color='Average Magnitude',
                                    text='Count',
                                    title='Events Distribution by Day of the Week',
                                    color_continuous_scale='Viridis',
                                    labels={'Count': 'Number of Events', 'Average Magnitude': 'Average Magnitude'}
                                )
                                
                                fig_dow.update_traces(textposition='outside')
                                fig_dow.update_layout(height=400)
                                
                                st.plotly_chart(fig_dow, use_container_width=True)
                                
                                # Add an interpretation
                                st.markdown("""
                                ### Weekly pattern analysis
                                
                                This chart shows how seismic events are distributed throughout the week.
                                Significant patterns might indicate:
                                
                                - Possible influence of human activities (e.g., controlled explosions on weekdays)
                                - Trends that deserve additional investigation
                                - Note that in natural phenomena, weekly patterns are generally not expected
                                """)
                                
                                # Create a heatmap of activity by day of the week and week of the month
                                st.subheader("Heat Map: Activity by Week and Day")
                                
                                # Add a relative week number column within the period
                                filtered_df['week_num'] = filtered_df['time'].dt.isocalendar().week
                                min_week = filtered_df['week_num'].min()
                                filtered_df['rel_week'] = filtered_df['week_num'] - min_week + 1
                                
                                # Group by relative week and day of the week
                                heatmap_weekly = filtered_df.groupby(['rel_week', 'day_name']).size().reset_index(name='count')
                                
                                # Pivot to create the format for the heatmap
                                pivot_weekly = pd.pivot_table(
                                    heatmap_weekly, 
                                    values='count', 
                                    index='day_name', 
                                    columns='rel_week',
                                    fill_value=0
                                )
                                
                                # Reorder the days
                                if set(ordered_days).issubset(set(pivot_weekly.index)):
                                    pivot_weekly = pivot_weekly.reindex(ordered_days)
                                
                                # Create heatmap
                                fig_weekly_heat = px.imshow(
                                    pivot_weekly,
                                    labels=dict(x="Week", y="Day of the Week", color="Number of Events"),
                                    x=[f"Week {i}" for i in pivot_weekly.columns],
                                    y=pivot_weekly.index,
                                    color_continuous_scale="YlOrRd",
                                    title="Heat Map: Seismic Activity by Week and Day"
                                )
                                
                                st.plotly_chart(fig_weekly_heat, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error in weekly pattern analysis: {e}")
                        
                        # Tab 3: Hourly Patterns
                        with temp_tab3:
                            try:
                                st.subheader("Events Distribution by Hour of Day")
                                
                                # Group by hour
                                hourly_counts = filtered_df.groupby('hour').agg({
                                    'id': 'count',
                                    'mag': ['mean', 'max']
                                }).reset_index()
                                
                                # Rename columns
                                hourly_counts.columns = ['Hour', 'Count', 'Average Magnitude', 'Maximum Magnitude']
                                
                                # Create bar chart for distribution by hour
                                fig_hourly = px.bar(
                                    hourly_counts,
                                    x='Hour',
                                    y='Count',
                                    color='Average Magnitude',
                                    title="Distribution of seismic events by hour of day",
                                    labels={"Hour": "Hour of day (UTC)", "Count": "Number of events"},
                                    color_continuous_scale='Viridis',
                                    text='Count'
                                )
                                
                                fig_hourly.update_traces(textposition='outside')
                                fig_hourly.update_layout(height=400)
                                
                                st.plotly_chart(fig_hourly, use_container_width=True)
                                
                                # Heatmap by hour and day of the week
                                st.subheader("Heat Map: Activity by Hour and Day of the Week")
                                
                                # Ensure 'day_name' exists
                                if 'day_name' not in filtered_df.columns:
                                    filtered_df['day_name'] = filtered_df['day_of_week'].map(days_translation)
                                
                                # Group by hour and day of the week
                                heatmap_data = filtered_df.groupby(['day_name', 'hour']).size().reset_index(name='count')
                                
                                # Pivot to create the format for the heatmap
                                pivot_data = pd.pivot_table(
                                    heatmap_data, 
                                    values='count', 
                                    index='day_name', 
                                    columns='hour',
                                    fill_value=0
                                )
                                
                                # Reorder the days
                                ordered_days = [days_translation[day] for day in days_order]
                                if set(ordered_days).issubset(set(pivot_data.index)):
                                    pivot_data = pivot_data.reindex(ordered_days)
                                
                                # Create heatmap
                                fig_heatmap = px.imshow(
                                    pivot_data,
                                    labels=dict(x="Hour of Day (UTC)", y="Day of the Week", color="Number of Events"),
                                    x=[f"{h}:00" for h in range(24)],
                                    y=pivot_data.index,
                                    color_continuous_scale="YlOrRd",
                                    title="Heat Map: Seismic Activity by Hour and Day of the Week"
                                )
                                
                                st.plotly_chart(fig_heatmap, use_container_width=True)
                                
                                st.markdown("""
                                ### Heat map interpretation
                                
                                This heat map shows the distribution of seismic events by hour and day of the week.
                                
                                - Darker cells indicate times with higher seismic activity
                                - Horizontal patterns suggest hours of the day with higher activity
                                - Vertical patterns indicate days of the week with more events
                                - Isolated intense color cells may indicate special events or temporal clusters
                                """)
                            except Exception as e:
                                st.error(f"Error in hourly pattern analysis: {e}")
                adv_tab1, adv_tab2, adv_tab3 = st.tabs([
                    "Correlations", 
                    "Magnitude by Region", 
                    "Comparisons"
                ])
                
                # Tab 1: Correlations
                with adv_tab1:
                    try:
                        st.subheader("Correlation Matrix")
                        
                        # Select variables for correlation
                        corr_cols = ['mag', 'depth', 'rms', 'gap', 'horizontalError', 'depthError']
                        
                        # Filter columns that exist in the DataFrame
                        valid_cols = [col for col in corr_cols if col in filtered_df.columns]
                        
                        if len(valid_cols) > 1:
                            corr_df = filtered_df[valid_cols].dropna()
                            
                            if len(corr_df) > 1:  # Ensure there's enough data for correlation
                                corr_matrix = corr_df.corr()
                                
                                fig_corr = px.imshow(
                                    corr_matrix,
                                    text_auto=True,
                                    color_continuous_scale="RdBu_r",
                                    title="Correlation Matrix",
                                    aspect="auto"
                                )
                                
                                st.plotly_chart(fig_corr, use_container_width=True)
                                
                                st.markdown("""
                                **Correlation matrix interpretation:**
                                - Values close to 1 indicate strong positive correlation
                                - Values close to -1 indicate strong negative correlation
                                - Values close to 0 indicate little or no correlation
                                """)
                                
                                # Add detailed correlation analysis
                                st.subheader("Detailed Correlation Analysis")
                                
                                # Find significant correlations
                                significant_corr = []
                                for i in range(len(valid_cols)):
                                    for j in range(i+1, len(valid_cols)):
                                        corr_val = corr_matrix.iloc[i, j]
                                        if abs(corr_val) > 0.3:  # Threshold for significant correlation
                                            significant_corr.append({
                                                'Variables': f"{valid_cols[i]} vs {valid_cols[j]}",
                                                'Correlation': corr_val,
                                                'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate' if abs(corr_val) > 0.5 else 'Weak',
                                                'Type': 'Positive' if corr_val > 0 else 'Negative'
                                            })
                                
                                if significant_corr:
                                    significant_df = pd.DataFrame(significant_corr)
                                    significant_df = significant_df.sort_values('Correlation', key=abs, ascending=False)
                                    
                                    st.dataframe(significant_df, use_container_width=True)
                                    
                                    # Visualize the strongest correlation
                                    if len(significant_df) > 0:
                                        top_corr = significant_df.iloc[0]
                                        var1, var2 = top_corr['Variables'].split(' vs ')
                                        
                                        st.subheader(f"Correlation Visualization: {top_corr['Variables']}")
                                        
                                        fig_scatter_corr = px.scatter(
                                            filtered_df,
                                            x=var1,
                                            y=var2,
                                            color='magnitud_categoria',
                                            size=ensure_positive(filtered_df['mag']),  # Use guaranteed positive values
                                            hover_name='place',
                                            title=f"{top_corr['Type']} {top_corr['Strength']} Correlation (r={top_corr['Correlation']:.2f})",
                                            color_discrete_map=magnitude_colors
                                        )
                                        
                                        fig_scatter_corr.update_layout(height=500)
                                        
                                        st.plotly_chart(fig_scatter_corr, use_container_width=True)
                                else:
                                    st.info("No significant correlations found between the analyzed variables.")
                            else:
                                st.warning("Not enough data to calculate correlations.")
                        else:
                            st.warning("Not enough numeric columns to calculate correlations.")
                    except Exception as e:
                        st.error(f"Error in correlation analysis: {e}")
                
                # Tab 2: Magnitude by Region
                with adv_tab2:
                    try:
                        st.subheader("Magnitude Analysis by Region")
                        
                        # Extract main regions
                        filtered_df['region'] = filtered_df['place'].str.split(', ').str[-1]
                        region_stats = filtered_df.groupby('region').agg({
                            'id': 'count',
                            'mag': ['mean', 'max', 'min'],
                            'depth': 'mean'
                        }).reset_index()
                        
                        # Flatten multilevel columns
                        region_stats.columns = ['Region', 'Count', 'Average Magnitude', 'Maximum Magnitude', 'Minimum Magnitude', 'Average Depth']
                        
                        # Filter regions with enough events
                        min_events = st.slider("Minimum events per region", 1, 50, 5)
                        filtered_regions = region_stats[region_stats['Count'] >= min_events].sort_values('Average Magnitude', ascending=False)
                        
                        # Visualize
                        if not filtered_regions.empty:
                            fig_regions = px.bar(
                                filtered_regions.head(15),  # Top 15 regions
                                x='Region',
                                y='Average Magnitude',
                                error_y=filtered_regions.head(15)['Maximum Magnitude'] - filtered_regions.head(15)['Average Magnitude'],
                                color='Count',
                                hover_data=['Count', 'Maximum Magnitude', 'Average Depth'],
                                title='Average Magnitude by Region (Top 15)',
                                color_continuous_scale='Viridis'
                            )
                            
                            fig_regions.update_layout(height=500, xaxis_tickangle=-45)
                            st.plotly_chart(fig_regions, use_container_width=True)
                            
                            # Show detailed table
                            st.dataframe(
                                filtered_regions.sort_values('Count', ascending=False),
                                use_container_width=True
                            )
                        else:
                            st.warning(f"No regions with at least {min_events} events. Try reducing the minimum.")
                    except Exception as e:
                        st.error(f"Error in magnitude analysis by region: {e}")
                
                # Tab 3: Comparisons
                with adv_tab3:
                    try:
                        st.subheader("Comparative Analysis")
                        
                        # Available numeric columns
                        numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
                        numeric_cols = [col for col in numeric_cols if col not in ['cluster', 'rel_week', 'week_num']]
                        
                        # Select variables to compare
                        if len(numeric_cols) >= 2:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                x_variable = st.selectbox(
                                    "X Variable",
                                    options=numeric_cols,
                                    index=numeric_cols.index('mag') if 'mag' in numeric_cols else 0
                                )
                            
                            with col2:
                                y_variable = st.selectbox(
                                    "Y Variable",
                                    options=numeric_cols,
                                    index=numeric_cols.index('depth') if 'depth' in numeric_cols else min(1, len(numeric_cols)-1)
                                )
                            
                            # Create custom scatter plot
                            fig_custom = px.scatter(
                                filtered_df,
                                x=x_variable,
                                y=y_variable,
                                color='magnitud_categoria',
                                size=ensure_positive(filtered_df['mag']),  # Positive values
                                hover_name='place',
                                title=f"Relationship between {x_variable} and {y_variable}",
                                color_discrete_map=magnitude_colors,
                                trendline='ols'  # Add trend line
                            )
                            
                            fig_custom.update_layout(height=500)
                            st.plotly_chart(fig_custom, use_container_width=True)
                            
                            # Analysis by category
                            st.subheader("Statistics by Magnitude Category")
                            
                            # Group by magnitude category
                            cat_stats = filtered_df.groupby('magnitud_categoria').agg({
                                'id': 'count',
                                'mag': ['mean', 'std'],
                                'depth': ['mean', 'std'],
                                'rms': 'mean'
                            }).reset_index()
                            
                            # Flatten columns
                            cat_stats.columns = [
                                'Category', 'Count', 'Average Magnitude', 'Mag Std Dev', 
                                'Average Depth', 'Depth Std Dev', 'Average RMS'
                            ]
                            
                            # Order categories
                            cat_order = ['Minor (<2)', 'Light (2-4)', 'Moderate (4-6)', 'Strong (6+)']
                            cat_stats['Order'] = cat_stats['Category'].map({cat: i for i, cat in enumerate(cat_order)})
                            cat_stats = cat_stats.sort_values('Order').drop('Order', axis=1)
                            
                            # Visualize statistics
                            st.dataframe(cat_stats, use_container_width=True)
                            
                            # Comparative bar chart
                            fig_cats = go.Figure()
                            
                            # Add bars for count
                            fig_cats.add_trace(go.Bar(
                                x=cat_stats['Category'],
                                y=cat_stats['Count'],
                                name='Count',
                                marker_color='lightskyblue',
                                opacity=0.7
                            ))
                            
                            # Add line for average depth
                            fig_cats.add_trace(go.Scatter(
                                x=cat_stats['Category'],
                                y=cat_stats['Average Depth'],
                                name='Average Depth (km)',
                                mode='lines+markers',
                                marker=dict(color='darkred', size=8),
                                line=dict(width=2),
                                yaxis='y2'
                            ))
                            
                            # Configure axes and layout
                            fig_cats.update_layout(
                                title='Comparison of Count and Depth by Category',
                                xaxis=dict(title='Magnitude Category'),
                                yaxis=dict(title='Number of Events', side='left'),
                                yaxis2=dict(
                                    title='Average Depth (km)',
                                    side='right',
                                    overlaying='y'
                                ),
                                legend=dict(x=0.01, y=0.99),
                                barmode='group',
                                height=400
                            )
                            
                            st.plotly_chart(fig_cats, use_container_width=True)
                        else:
                            st.warning("Not enough numeric columns to perform comparative analysis.")
                    except Exception as e:
                        st.error(f"Error in comparative analysis: {e}")
            
            # Data table (expandable)
            with st.expander("View data in tabular format"):
                try:
                    # Available columns to display
                    display_cols = [col for col in ['time', 'place', 'mag', 'depth', 'type', 'magType', 'rms'] if col in filtered_df.columns]
                    
                    # Sort options
                    sort_col = st.selectbox(
                        "Sort by",
                        options=display_cols,
                        index=0,
                        key="sort_colums"
                    )
                    
                    sort_order = st.radio(
                        "Order",
                        options=['Descending', 'Ascending'],
                        index=0,
                        horizontal=True
                    )
                    
                    # Sort data
                    sorted_df = filtered_df.sort_values(
                        by=sort_col,
                        ascending=(sort_order == 'Ascending'),
                        key="sort_data"
                    )
                    
                    # Show table
                    st.dataframe(
                        sorted_df[display_cols],
                        use_container_width=True
                    )
                    
                    # Option to download filtered data
                    csv = sorted_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download filtered data (CSV)",
                        data=csv,
                        file_name="filtered_earthquake_data.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Error displaying data table: {e}")        

        # --- Advanced Analysis Tab ---
        with main_tabs[3]:
            adv_tab1, adv_tab2, adv_tab3 = st.tabs([
                "Correlations", 
                "Magnitude by Region", 
                "Comparisons",
            ])
            
            # Tab 1: Correlations
            with adv_tab1:
                try:
                    st.subheader("Correlation Matrix")
                    
                    # Select variables for correlation
                    corr_cols = ['mag', 'depth', 'rms', 'gap', 'horizontalError', 'depthError']
                    
                    # Filter columns that exist in the DataFrame
                    valid_cols = [col for col in corr_cols if col in filtered_df.columns]
                    
                    if len(valid_cols) > 1:
                        corr_df = filtered_df[valid_cols].dropna()
                        
                        if len(corr_df) > 1:  # Ensure there's enough data for correlation
                            corr_matrix = corr_df.corr()
                            
                            fig_corr = px.imshow(
                                corr_matrix,
                                text_auto=True,
                                color_continuous_scale="RdBu_r",
                                title="Correlation Matrix",
                                aspect="auto"
                            )
                            
                            st.plotly_chart(fig_corr, use_container_width=True)
                            
                            st.markdown("""
                            **Correlation matrix interpretation:**
                            - Values close to 1 indicate strong positive correlation
                            - Values close to -1 indicate strong negative correlation
                            - Values close to 0 indicate little or no correlation
                            """)
                            
                            # Add detailed correlation analysis
                            st.subheader("Detailed Correlation Analysis")
                            
                            # Find significant correlations
                            significant_corr = []
                            for i in range(len(valid_cols)):
                                for j in range(i+1, len(valid_cols)):
                                    corr_val = corr_matrix.iloc[i, j]
                                    if abs(corr_val) > 0.3:  # Threshold for significant correlation
                                        significant_corr.append({
                                            'Variables': f"{valid_cols[i]} vs {valid_cols[j]}",
                                            'Correlation': corr_val,
                                            'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate' if abs(corr_val) > 0.5 else 'Weak',
                                            'Type': 'Positive' if corr_val > 0 else 'Negative'
                                        })
                            
                            if significant_corr:
                                significant_df = pd.DataFrame(significant_corr)
                                significant_df = significant_df.sort_values('Correlation', key=abs, ascending=False)
                                
                                st.dataframe(significant_df, use_container_width=True)
                                
                                # Visualize the strongest correlation
                                if len(significant_df) > 0:
                                    top_corr = significant_df.iloc[0]
                                    var1, var2 = top_corr['Variables'].split(' vs ')
                                    
                                    st.subheader(f"Correlation Visualization: {top_corr['Variables']}")
                                    
                                    fig_scatter_corr = px.scatter(
                                        filtered_df,
                                        x=var1,
                                        y=var2,
                                        color='magnitud_categoria',
                                        size=ensure_positive(filtered_df['mag']),  # Use guaranteed positive values
                                        hover_name='place',
                                        title=f"{top_corr['Type']} {top_corr['Strength']} Correlation (r={top_corr['Correlation']:.2f})",
                                        color_discrete_map=magnitude_colors
                                    )
                                    
                                    fig_scatter_corr.update_layout(height=500)
                                    
                                    st.plotly_chart(fig_scatter_corr, use_container_width=True)
                            else:
                                st.info("No significant correlations found between the analyzed variables.")
                        else:
                            st.warning("Not enough data to calculate correlations.")
                    else:
                        st.warning("Not enough numeric columns to calculate correlations.")
                except Exception as e:
                    st.error(f"Error in correlation analysis: {e}")
            
            # Tab 2: Magnitude by Region
            with adv_tab2:
                try:
                    st.subheader("Magnitude Analysis by Region")
                    
                    # Extract main regions
                    filtered_df['region'] = filtered_df['place'].str.split(', ').str[-1]
                    region_stats = filtered_df.groupby('region').agg({
                        'id': 'count',
                        'mag': ['mean', 'max', 'min'],
                        'depth': 'mean'
                    }).reset_index()
                    
                    # Flatten multilevel columns
                    region_stats.columns = ['Region', 'Count', 'Average Magnitude', 'Maximum Magnitude', 'Minimum Magnitude', 'Average Depth']
                    
                    # Filter regions with enough events
                    min_events = st.slider("Minimum events per region", 1, 50, 5)
                    filtered_regions = region_stats[region_stats['Count'] >= min_events].sort_values('Average Magnitude', ascending=False)
                    
                    # Visualize
                    if not filtered_regions.empty:
                        fig_regions = px.bar(
                            filtered_regions.head(15),  # Top 15 regions
                            x='Region',
                            y='Average Magnitude',
                            error_y=filtered_regions.head(15)['Maximum Magnitude'] - filtered_regions.head(15)['Average Magnitude'],
                            color='Count',
                            hover_data=['Count', 'Maximum Magnitude', 'Average Depth'],
                            title='Average Magnitude by Region (Top 15)',
                            color_continuous_scale='Viridis'
                        )
                        
                        fig_regions.update_layout(height=500, xaxis_tickangle=-45)
                        st.plotly_chart(fig_regions, use_container_width=True)
                        
                        # Show detailed table
                        st.dataframe(
                            filtered_regions.sort_values('Count', ascending=False),
                            use_container_width=True
                        )
                    else:
                        st.warning(f"No regions with at least {min_events} events. Try reducing the minimum.")
                except Exception as e:
                    st.error(f"Error in magnitude analysis by region: {e}")
            
            # Tab 3: Comparisons
            with adv_tab3:

                try:
                    st.subheader("Comparative Analysis")
                    
                    # Available numeric columns
                    numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
                    numeric_cols = [col for col in numeric_cols if col not in ['cluster', 'rel_week', 'week_num']]
                    
                    # Select variables to compare
                    if len(numeric_cols) >= 2:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            x_variable = st.selectbox(
                                "X Variable",
                                options=numeric_cols,
                                index=numeric_cols.index('mag') if 'mag' in numeric_cols else 0
                            )
                        
                        with col2:
                            y_variable = st.selectbox(
                                "Y Variable",
                                options=numeric_cols,
                                index=numeric_cols.index('depth') if 'depth' in numeric_cols else min(1, len(numeric_cols)-1)
                            )
                        
                        # Create custom scatter plot
                        fig_custom = px.scatter(
                            filtered_df,
                            x=x_variable,
                            y=y_variable,
                            color='magnitud_categoria',
                            size=ensure_positive(filtered_df['mag']),  # Positive values
                            hover_name='place',
                            title=f"Relationship between {x_variable} and {y_variable}",
                            color_discrete_map=magnitude_colors,
                            trendline='ols'  # Add trend line
                        )
                        
                        fig_custom.update_layout(height=500)
                        st.plotly_chart(fig_custom, use_container_width=True)
                        
                        # Analysis by category
                        st.subheader("Statistics by Magnitude Category")
                        
                        # Group by magnitude category
                        cat_stats = filtered_df.groupby('magnitud_categoria').agg({
                            'id': 'count',
                            'mag': ['mean', 'std'],
                            'depth': ['mean', 'std'],
                            'rms': 'mean'
                        }).reset_index()
                        
                        # Flatten columns
                        cat_stats.columns = [
                            'Category', 'Count', 'Average Magnitude', 'Mag Std Dev', 
                            'Average Depth', 'Depth Std Dev', 'Average RMS'
                        ]
                        
                        # Order categories
                        cat_order = ['Minor (<2)', 'Light (2-4)', 'Moderate (4-6)', 'Strong (6+)']
                        cat_stats['Order'] = cat_stats['Category'].map({cat: i for i, cat in enumerate(cat_order)})
                        cat_stats = cat_stats.sort_values('Order').drop('Order', axis=1)
                        
                        # Visualize statistics
                        st.dataframe(cat_stats, use_container_width=True)
                        
                        # Comparative bar chart
                        fig_cats = go.Figure()
                        
                        # Add bars for count
                        fig_cats.add_trace(go.Bar(
                            x=cat_stats['Category'],
                            y=cat_stats['Count'],
                            name='Count',
                            marker_color='lightskyblue',
                            opacity=0.7
                        ))
                        
                        # Add line for average depth
                        fig_cats.add_trace(go.Scatter(
                            x=cat_stats['Category'],
                            y=cat_stats['Average Depth'],
                            name='Average Depth (km)',
                            mode='lines+markers',
                            marker=dict(color='darkred', size=8),
                            line=dict(width=2),
                            yaxis='y2'
                        ))
                        
                        # Configure axes and layout
                        fig_cats.update_layout(
                            title='Comparison of Count and Depth by Category',
                            xaxis=dict(title='Magnitude Category'),
                            yaxis=dict(title='Number of Events', side='left'),
                            yaxis2=dict(
                                title='Average Depth (km)',
                                side='right',
                                overlaying='y'
                            ),
                            legend=dict(x=0.01, y=0.99),
                            barmode='group',
                            height=400
                        )
                        
                        st.plotly_chart(fig_cats, use_container_width=True)
                    else:
                        st.warning("Not enough numeric columns to perform comparative analysis.")
                except Exception as e:
                    st.error(f"Error in comparative analysis: {e}")


                try:
                    st.subheader("Depth vs Magnitude Analysis")
                    
                    # Create a more effective visualization
                    heatmap_df = filtered_df.copy()
                    heatmap_df = heatmap_df[(heatmap_df['mag'] > 0) & (heatmap_df['depth'] > 0)]
                    
                    # Add visualization options
                    viz_type = st.radio(
                        "Select visualization type:",
                        ["Scatter plot", "3D distribution"],
                        horizontal=True
                    )
                    
                    if viz_type == "Scatter plot":
                        fig = px.scatter(
                            heatmap_df,
                            x="mag",
                            y="depth",
                            color="magnitud_categoria",
                            size=ensure_positive(heatmap_df['mag'], min_size=4),
                            color_discrete_map=magnitude_colors,
                            opacity=0.7,
                            hover_data=["place", "time"],
                            labels={"mag": "Magnitude", "depth": "Depth (km)"},
                            title="Relationship between Magnitude and Depth"
                        )
                        fig.update_yaxes(autorange="reversed")  # Depth increases downward
                    
                    elif viz_type == "3D distribution":
                        # Add region as a third dimension
                        fig = px.scatter_3d(
                            heatmap_df,
                            x="mag",
                            y="depth",
                            z="time",
                            color="magnitud_categoria",
                            size=ensure_positive(heatmap_df['mag'], min_size=3),
                            color_discrete_map=magnitude_colors,
                            opacity=0.7,
                            hover_data=["place"],
                            labels={"mag": "Magnitude", "depth": "Depth (km)", "time": "Date/Time"},
                            title="3D View: Magnitude, Depth and Time"
                        )
                    
                    # Make the plot larger for better visibility
                    fig.update_layout(height=600, width=800)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add statistical insight
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Statistical Summary")
                        depth_mag_corr = heatmap_df[['depth', 'mag']].corr().iloc[0,1]
                        st.metric(
                            "Correlation between Depth and Magnitude", 
                            f"{depth_mag_corr:.3f}",
                            delta=f"{depth_mag_corr:.3f}" if depth_mag_corr > 0 else f"{depth_mag_corr:.3f}"
                        )
                    
                    with col2:
                        st.subheader("Distribution by Depth Range")
                        # Create depth ranges from 0 to 200 km in steps of 20km
                        depth_ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), 
                                       (100, 120), (120, 140), (140, 160), (160, 180), (180, 200), (200, float('inf'))]
                        depth_labels = ['0-20km', '20-40km', '40-60km', '60-80km', '80-100km', 
                                      '100-120km', '120-140km', '140-160km', '160-180km', '180-200km', '>200km']
                        
                        # Categorize events by depth
                        heatmap_df['depth_category'] = pd.cut(
                            heatmap_df['depth'], 
                            bins=[r[0] for r in depth_ranges] + [float('inf')],
                            labels=depth_labels
                        )
                        
                        # Show distribution
                        depth_counts = heatmap_df['depth_category'].value_counts().reset_index()
                        depth_counts.columns = ['Depth Range', 'Count']
                        st.dataframe(depth_counts, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error in depth vs. magnitude analysis: {e}")

        with main_tabs[4]:
            st.markdown("## 🚨 Multi-Source Alert Center")
            st.info("Este módulo verifica múltiples fuentes de datos sobre alertas sísmicas recientes y notifica sobre eventos significativos.")
            
            # Función para reproducir sonido de alerta
            def play_alarm_sound():
                """
                Play an alarm sound using HTML5 audio with proper error handling
                to prevent WebSocket connection issues.
                """
                try:
                    # Path to the audio file - make sure it exists!
                    sound_file = "data/alarm.wav"  # Better to store in data folder
                    
                    # Create HTML with proper error handling
                    sound_html = f"""
                    <script>
                    // Use try-catch to handle audio errors properly
                    try {{
                        const audio = new Audio('{sound_file}');
                        audio.volume = 0.7;  // 70% volume
                        
                        // Handle errors properly
                        audio.onerror = function(e) {{
                            console.error('Audio error:', e);
                        }};
                        
                        // Only play if user has interacted with page (browser policy)
                        if (document.hasFocus()) {{
                            var playPromise = audio.play();
                            
                            // Handle promise rejection (common in some browsers)
                            if (playPromise !== undefined) {{
                                playPromise.catch(function(error) {{
                                    console.error('Play promise error:', error);
                                }});
                            }}
                        }}
                    }} catch (e) {{
                        console.error('Audio setup error:', e);
                    }}
                    </script>
                    
                    <div style="display:none">
                        <audio id="alert-sound">
                            <source src="{sound_file}" type="audio/wav">
                        </audio>
                    </div>
                    """
                    
                    # Render the HTML/JS safely
                    st.markdown(sound_html, unsafe_allow_html=True)
                except Exception as e:
                    # Log the error but don't crash if sound playing fails
                    print(f"Error playing alert sound: {e}")
                    # Avoid displaying errors to users for non-critical feature
                    pass
            
            # Crear columnas para mostrar diferentes fuentes de alerta
            alert_col1, alert_col2 = st.columns(2)
            
            # Primera fuente - USGS Significant Events
            with alert_col1:
                st.subheader("USGS Significant Events")
                
                try:
                    # Obtener eventos significativos de USGS
                    sig_url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_week.csv"
                    
                    # Mostrar que estamos cargando
                    with st.spinner("Verificando alertas USGS..."):
                        # Intentar cargar los datos
                        try:
                            sig_df = pd.read_csv(sig_url)
                            sig_df['time'] = pd.to_datetime(sig_df['time']).dt.tz_localize(None)
                            
                            # Filtrar eventos de las últimas 24 horas
                            last_24h = pd.Timestamp.now() - pd.Timedelta(hours=24)
                            recent_sig = sig_df[sig_df['time'] > last_24h]
                            
                            # Verificar si hay alertas recientes
                            if not recent_sig.empty:
                                # Reproducir sonido de alerta si hay eventos significativos recientes
                                play_alarm_sound()
                                
                                # Crear alerta con color basado en magnitud
                                for _, event in recent_sig.iterrows():
                                    mag = event['mag']
                                    place = event['place']
                                    event_time = event['time']
                                    
                                    # Color basado en magnitud
                                    if mag >= 7.0:
                                        st.error(f"⚠️ ALERTA CRÍTICA: Magnitud {mag:.1f} en {place} ({event_time})")
                                    elif mag >= 6.0:
                                        st.warning(f"⚠️ ALERTA ALTA: Magnitud {mag:.1f} en {place} ({event_time})")
                                    else:
                                        st.info(f"ℹ️ ALERTA: Magnitud {mag:.1f} en {place} ({event_time})")
                                
                                # Mostrar tabla de eventos significativos recientes
                                st.dataframe(
                                    recent_sig[['time', 'place', 'mag', 'depth', 'type']],
                                    use_container_width=True
                                )
                            else:
                                st.success("✓ No hay alertas significativas de USGS en las últimas 24 horas")
                        except Exception as e:
                            st.error(f"Error al cargar alertas de USGS: {e}")
                except Exception as e:
                    st.error(f"Error en fuente USGS: {e}")
            
            # Segunda fuente - EMSC (Centro Sismológico Euro-Mediterráneo)
            with alert_col2:
                st.subheader("EMSC Recent Events")
                
                try:
                    # URL de EMSC para eventos recientes
                    emsc_url = "https://www.emsc-csem.org/service/rss/rss.php?typ=emsc&magmin=4"
                    
                    # Mostrar estado de carga
                    st.info("Servicio de alertas EMSC no disponible en esta versión")
                    st.markdown("""
                    El Centro Sismológico Euro-Mediterráneo (EMSC) proporciona datos de terremotos 
                    que ocurren principalmente en la región europea y mediterránea.
                    
                    Para implementar completamente esta fuente, se requiere un parser XML para el feed RSS de EMSC.
                    """)
                except Exception as e:
                    st.error(f"Error en fuente EMSC: {e}")
            
            # Sección de información sobre alertas
            st.markdown("---")
            st.subheader("Sistema de Nivel de Alerta")
            
            # Crear una tabla de niveles de alerta
            alert_data = {
                "Nivel de Alerta": ["Baja", "Media", "Alta", "Crítica"],
                "Magnitud": ["< 5.0", "5.0 - 5.9", "6.0 - 6.9", "≥ 7.0"],
                "Notificación": ["Informativa", "Advertencia", "Alerta", "Emergencia"],
                "Sonido": ["No", "No", "Sí", "Sí"]
            }
            alert_df = pd.DataFrame(alert_data)
            
            # Aplicar colores a los niveles de alerta
            st.dataframe(alert_df, use_container_width=True)
            
            # Añadir información sobre sistemas de alerta temprana
            with st.expander("ℹ️ Sobre Sistemas de Alerta Sísmica"):
                st.markdown("""
                ### Sistemas de Alerta Temprana de Terremotos
                
                Los sistemas de alerta temprana detectan las primeras ondas sísmicas (P) que viajan más rápido pero causan menos daño, 
                proporcionando segundos o incluso minutos de advertencia antes de la llegada de las ondas S más destructivas.
                
                #### Principales sistemas en operación:
                
                - **ShakeAlert** (USA) - Operado por USGS
                - **Sistema de Alerta Sísmica Mexicano** (México)
                - **J-ALERT** (Japón)
                - **Sistema Nacional de Alerta de Terremotos de Taiwan**
                
                Las alertas tempranas permiten acciones como:
                - Detener trenes y elevadores
                - Cerrar válvulas de gas e infraestructura crítica
                - Permitir a las personas buscar refugio
                
                > **Nota**: Este panel muestra alertas históricos y no proporciona alertas en tiempo real para emergencias.
                """)

        with main_tabs[5]:
            st.title("Historical Earthquake Analysis (2005-2025)")
            st.markdown("""
            This analysis examines significant earthquakes (magnitude > 5.5) from 2005-2025, identifying patterns, 
            trends and potential predictive indicators for future seismic activity.
            """)
            
            # Load historical data
            @st.cache_data(ttl=3600)
            def load_historical_data():
                try:
                    # Try to load from the CSV file first
                    df_historic = pd.read_csv('data/2005-2025_mas_de_5_5_E_R.csv')
                    df_historic['time'] = pd.to_datetime(df_historic['time'])
                    return df_historic
                except Exception as e:
                    st.error(f"Error loading historical data: {e}")
                    return None
            
            with st.spinner("Loading historical earthquake data..."):
                historic_eq = load_historical_data()
            
            if historic_eq is not None:
                # Process and prepare data for analysis
                data = historic_eq.copy()
                
                # Display loading info
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"Loaded {len(data)} significant earthquakes (M>5.5) from 2005-2025")
                with col2:
                    st.warning("⚠️ Historical analysis is based on past events and patterns, not real-time data")
                
                # Verify and clean data
                data = data.dropna(subset=['mag', 'latitude', 'longitude'])
                duplicates = data.duplicated(subset=['time', 'latitude', 'longitude', 'mag']).sum()
                if duplicates > 0:
                    data = data.drop_duplicates(subset=['time', 'latitude', 'longitude', 'mag'])
                    st.warning(f"Removed {duplicates} duplicate entries from the dataset")
                
                # Filter major earthquakes
                major_quakes = data[data['mag'] >= 7.0].copy()
                
                # Add time-based features
                data['year'] = data['time'].dt.year
                data['month'] = data['time'].dt.month
                data['day'] = data['time'].dt.day
                data['hour'] = data['time'].dt.hour
                data['weekday'] = data['time'].dt.day_name()
                data['decade'] = (data['year'] // 10) * 10
                
                # Create magnitude categories with colors matching your app
                magnitude_ranges = [0, 2.5, 4.0, 5.5, 7.0, 10.0]
                magnitude_labels = ['Micro (<2.5)', 'Leve (2.5-4.0)', 'Moderado (4.0-5.5)', 'Fuerte (5.5-7.0)', 'Mayor (>7.0)']
                data['magnitude_category'] = pd.cut(data['mag'], bins=magnitude_ranges, labels=magnitude_labels)
                
                # Define tectonic regions function
                def categorize_region(lat, lon):
                    # Pacific Ring of Fire
                    if ((lon > 120 or lon < -120) and (lat > -60 and lat < 60)):
                        return "Pacific West" if lon > 120 else "Pacific East"
                    # Mediterranean-Caucasus
                    elif ((lat > 30 and lat < 45) and (lon > -10 and lon < 50)):
                        return "Mediterranean-Caucasus"
                    # Indonesia-Himalaya Belt
                    elif ((lat > -10 and lat < 45) and (lon > 70 and lon < 120)):
                        return "Indonesia-Himalaya"
                    # Other regions
                    else:
                        return "Other Regions"
                
                # Apply regional categorizations
                data['region'] = data.apply(lambda x: categorize_region(x['latitude'], x['longitude']), axis=1)
                major_quakes['region'] = major_quakes.apply(lambda x: categorize_region(x['latitude'], x['longitude']), axis=1)
                
                # Add tectonic region classification
                data['tectonic_region'] = data.apply(lambda x: 
                    'Ring of Fire' if ((x['longitude'] > 120 and x['longitude'] < 180) or 
                                (x['longitude'] < -120 and x['longitude'] > -180)) and 
                                (x['latitude'] > -60 and x['latitude'] < 60) else
                    'Alpine-Himalayan Belt' if ((x['latitude'] > 30 and x['latitude'] < 45) and 
                                            (x['longitude'] > 0 and x['longitude'] < 150)) or 
                                        ((x['latitude'] > 0 and x['latitude'] < 30) and 
                                            (x['longitude'] > 60 and x['longitude'] < 120)) else
                    'Mid-Atlantic Ridge' if (x['longitude'] > -45 and x['longitude'] < 0) and 
                                        (x['latitude'] > -60 and x['latitude'] < 80) else
                    'Other Tectonic Regions', axis=1)
                
                # Add rounded coordinates for hotspot analysis
                data['lat_rounded'] = round(data['latitude'], 1)
                data['lon_rounded'] = round(data['longitude'], 1)
                
                # Calculate time between major earthquakes in the same region
                if len(major_quakes) > 1:
                    major_quakes = major_quakes.sort_values(by=['region', 'time'])
                    major_quakes['years_since_last'] = major_quakes.groupby('region')['time'].diff().dt.total_seconds() / (365.25 * 24 * 60 * 60)
                
                # Create subtabs within the historical analysis tab
                hist_tabs = st.tabs([
                    "📊 Overview", 
                    "🗺️ Global Distribution", 
                    "📈 Time Patterns", 
                    "🌋 Major Events", 
                    "🔮 Recurrence Analysis"
                ])
                
                # Tab 1: Overview
                with hist_tabs[0]:
                    st.header("Historical Earthquake Overview")
                    
                    # Calculate key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            label="Total Earthquakes (>5.5)",
                            value=f"{len(data):,}",
                            delta=None
                        )
                        
                    with col2:
                        st.metric(
                            label="Average Magnitude",
                            value=f"{data['mag'].mean():.2f}",
                            delta=None
                        )
                        
                    with col3:
                        st.metric(
                            label="Maximum Magnitude",
                            value=f"{data['mag'].max():.1f}",
                            delta=None
                        )
                        
                    with col4:
                        most_active_year = data['year'].value_counts().idxmax()
                        st.metric(
                            label="Most Active Year",
                            value=f"{most_active_year}",
                            delta=f"{data[data['year']==most_active_year].shape[0]} events"
                        )
                    
                    # Magnitude distribution
                    st.subheader("Magnitude Distribution")
                    fig_mag_dist = px.histogram(
                        data, 
                        x="mag", 
                        nbins=30, 
                        color="magnitude_category",
                        color_discrete_sequence=px.colors.sequential.Plasma,
                        title="Distribution of Earthquake Magnitudes"
                    )
                    fig_mag_dist.add_vline(x=7.0, line_dash="dash", line_color="red", annotation_text="Major (7+)")
                    st.plotly_chart(fig_mag_dist, use_container_width=True)
                    
                    # Tectonic regions
                    st.subheader("Distribution by Tectonic Region")
                    region_counts = data['tectonic_region'].value_counts().reset_index()
                    region_counts.columns = ['Region', 'Count']
                    
                    fig_region = px.pie(
                        region_counts, 
                        values='Count', 
                        names='Region',
                        color_discrete_sequence=px.colors.sequential.Plasma,
                        hole=0.4
                    )
                    fig_region.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_region, use_container_width=True)
                    
                    # Yearly trend
                    st.subheader("Yearly Trend")
                    yearly_counts = data.groupby('year').size().reset_index(name='count')
                    
                    fig_yearly = px.line(
                        yearly_counts, 
                        x='year', 
                        y='count',
                        markers=True,
                        line_shape='spline',
                        title='Significant Earthquakes by Year (2005-2025)',
                        labels={'year': 'Year', 'count': 'Number of Earthquakes'}
                    )
                    
                    # Add trendline
                    x = yearly_counts['year']
                    y = yearly_counts['count']
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    fig_yearly.add_traces(
                        go.Scatter(
                            x=x,
                            y=p(x),
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='Trend'
                        )
                    )
                    
                    st.plotly_chart(fig_yearly, use_container_width=True)
                
                # Tab 2: Global Distribution
                with hist_tabs[1]:
                    st.header("Global Distribution of Significant Earthquakes")
                    
                    # Create a modern interactive map
                    st.markdown("### Earthquake Distribution Map (2005-2025)")
                    
                    fig_map = px.scatter_geo(
                        data,
                        lat="latitude",
                        lon="longitude",
                        color="mag",
                        size="mag",
                        hover_name="place",
                        hover_data=["time", "mag", "depth"],
                        title="Global Earthquake Distribution (M>5.5, 2005-2025)",
                        color_continuous_scale=px.colors.sequential.Plasma,
                        projection="natural earth",
                        size_max=20
                    )
                    
                    # Highlight the top 5 strongest earthquakes
                    top_earthquakes = data.nlargest(5, 'mag')
                    
                    for _, row in top_earthquakes.iterrows():
                        fig_map.add_trace(go.Scattergeo(
                            lat=[row['latitude']],
                            lon=[row['longitude']],
                            mode='markers+text',
                            marker=dict(
                                size=15,
                                symbol='triangle-up',
                                color='red',
                                line=dict(color='white', width=2)
                            ),
                            text=f"M{row['mag']:.1f}",
                            textposition="top center",
                            textfont=dict(color="white", size=12),
                            name=f"M{row['mag']:.1f} - {row['place']}",
                            hovertext=f"Magnitude: {row['mag']}<br>Date: {row['time']}<br>Place: {row['place']}",
                            hoverinfo="text"
                        ))
                    
                    # Update layout for dark theme
                    fig_map.update_layout(
                        height=600,
                        geo=dict(
                            showland=True,
                            landcolor="#d2b48c",
                            showocean=True,
                            oceancolor="#002244",
                            showcountries=True,
                            countrycolor="white",
                            showcoastlines=True,
                            coastlinecolor="white",
                            bgcolor="black"
                        ),
                        paper_bgcolor="black",
                        plot_bgcolor="black"
                    )
                    
                    st.plotly_chart(fig_map, use_container_width=True)
                    
                    # Heat map of activity
                    st.subheader("Earthquake Density Heatmap")
                    st.markdown("This visualization shows the concentration of seismic activity around the world")
                    
                    # Create a heat map using density_mapbox
                    fig_heat = px.density_mapbox(
                        data,
                        lat="latitude",
                        lon="longitude",
                        z="mag",  # Weight points by magnitude
                        radius=10,
                        center=dict(lat=0, lon=0),
                        zoom=0.5,
                        mapbox_style="carto-darkmatter",
                        opacity=0.7,
                        color_continuous_scale="Plasma"
                    )
                    
                    fig_heat.update_layout(height=500)
                    st.plotly_chart(fig_heat, use_container_width=True)
                    
                    # Cluster analysis
                    st.subheader("Regional Cluster Analysis")
                    
                    # Create a clustered view of earthquake hotspots
                    cluster_df = data[['latitude', 'longitude']].copy()
                    scaler = StandardScaler()
                    cluster_data = scaler.fit_transform(cluster_df)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        eps_distance = st.slider(
                            "Maximum distance between events (eps)",
                            min_value=0.05,
                            max_value=1.0,
                            value=0.2,
                            step=0.05,
                            key="hist_eps"
                        )
                    with col2:
                        min_samples = st.slider(
                            "Minimum samples per cluster",
                            min_value=2,
                            max_value=20,
                            value=5,
                            step=1,
                            key="hist_min_samples"
                        )
                    
                    with st.spinner("Performing cluster analysis..."):
                        dbscan = DBSCAN(eps=eps_distance, min_samples=min_samples)
                        data['cluster'] = dbscan.fit_predict(cluster_data)
                    
                    n_clusters = len(set(data['cluster'])) - (1 if -1 in data['cluster'] else 0)
                    n_noise = list(data['cluster']).count(-1)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Identified clusters", n_clusters)
                    col2.metric("Ungrouped events", n_noise)
                    
                    # Create cluster names based on regions
                    cluster_names = {}
                    for cluster_id in sorted(set(data['cluster'])):
                        if cluster_id == -1:
                            cluster_names[cluster_id] = "Ungrouped"
                        else:
                            # Get the most common place in this cluster
                            cluster_df = data[data['cluster'] == cluster_id]
                            most_common_place = cluster_df['place'].str.split(', ').str[-1].mode().iloc[0]
                            cluster_names[cluster_id] = f"Cluster {cluster_id}: {most_common_place}"
                    
                    data['cluster_name'] = data['cluster'].map(cluster_names)
                    
                    # Plot the clusters
                    fig_cluster = px.scatter_geo(
                        data,
                        lat="latitude",
                        lon="longitude",
                        color="cluster_name",
                        size="mag",
                        hover_name="place",
                        hover_data=["time", "mag", "depth"],
                        title=f"Earthquake Clusters (Found {n_clusters} clusters)",
                        projection="natural earth"
                    )
                    
                    fig_cluster.update_layout(height=600)
                    st.plotly_chart(fig_cluster, use_container_width=True)
                
                # Tab 3: Time Patterns
                with hist_tabs[2]:
                    st.header("Temporal Patterns in Earthquake Occurrence")
                    
                    # Monthly distribution
                    st.subheader("Monthly Distribution")
                    monthly_counts = data.groupby('month').size().reset_index(name='count')
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    monthly_counts['month_name'] = monthly_counts['month'].apply(lambda x: month_names[x-1])
                    
                    fig_monthly = px.bar(
                        monthly_counts,
                        x='month_name',
                        y='count',
                        color='count',
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title="Earthquakes by Month",
                        labels={'count': 'Number of Events', 'month_name': 'Month'}
                    )
                    st.plotly_chart(fig_monthly, use_container_width=True)
                    
                    # Hourly distribution
                    st.subheader("Distribution by Hour of Day")
                    hourly_counts = data.groupby('hour').size().reset_index(name='count')
                    
                    fig_hourly = px.line(
                        hourly_counts,
                        x='hour',
                        y='count',
                        markers=True,
                        title="Earthquakes by Hour of Day (UTC)",
                        labels={'count': 'Number of Events', 'hour': 'Hour of Day (UTC)'}
                    )
                    
                    # Fill area under the line
                    fig_hourly.update_traces(fill='tozeroy', fillcolor='rgba(128, 0, 128, 0.2)')
                    st.plotly_chart(fig_hourly, use_container_width=True)
                    
                    # Relation between depth and magnitude over time
                    st.subheader("Depth vs Magnitude Over Time")
                    
                    # Create 3D scatter plot
                    fig_3d = px.scatter_3d(
                        data.sort_values('time'),
                        x='time', 
                        y='depth', 
                        z='mag',
                        color='mag',
                        size='mag',
                        opacity=0.7,
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title="Depth vs Magnitude Over Time",
                        labels={
                            'time': 'Date',
                            'depth': 'Depth (km)',
                            'mag': 'Magnitude'
                        }
                    )
                    
                    fig_3d.update_layout(height=700)
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                    # Magnitude-depth relationship
                    st.subheader("Relationship Between Magnitude and Depth")
                    
                    fig_md = px.scatter(
                        data,
                        x='mag',
                        y='depth',
                        color='tectonic_region',
                        size='mag',
                        title="Magnitude vs Depth by Tectonic Region",
                        labels={'mag': 'Magnitude', 'depth': 'Depth (km)'}
                    )
                    
                    # Invert y-axis to show depth increasing downward
                    fig_md.update_yaxes(autorange="reversed")
                    
                    # Add trendline
                    fig_md.update_layout(height=600)
                    fig_md.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
                    st.plotly_chart(fig_md, use_container_width=True)
                
                # Tab 4: Major Events
                with hist_tabs[3]:
                    st.header("Analysis of Major Earthquakes (M≥7.0)")
                    
                    # Count and basic stats
                    major_count = len(major_quakes)
                    
                    st.metric(
                        label="Total Major Earthquakes (M≥7.0)",
                        value=f"{major_count}",
                        delta=f"{(major_count/len(data)*100):.1f}% of all events"
                    )
                    
                    # Map of major earthquakes
                    st.subheader("Global Distribution of Major Earthquakes")
                    
                    fig_major = px.scatter_geo(
                        major_quakes,
                        lat="latitude",
                        lon="longitude",
                        color="mag",
                        size="mag",
                        hover_name="place",
                        hover_data=["time", "mag", "depth"],
                        title="Major Earthquakes (M≥7.0, 2005-2025)",
                        color_continuous_scale="Viridis",
                        projection="natural earth",
                        size_max=25
                    )
                    
                    fig_major.update_layout(
                        height=600,
                        geo=dict(
                            showland=True,
                            landcolor="#d2b48c",
                            showocean=True,
                            oceancolor="#002244",
                            showcountries=True,
                            countrycolor="white",
                            showcoastlines=True,
                            coastlinecolor="white",
                            bgcolor="black"
                        ),
                        paper_bgcolor="black",
                        plot_bgcolor="black"
                    )
                    
                    st.plotly_chart(fig_major, use_container_width=True)
                    
                    # Distribution by region
                    st.subheader("Major Events by Region")
                    
                    region_major = major_quakes['region'].value_counts().reset_index()
                    region_major.columns = ['Region', 'Count']
                    
                    fig_region_major = px.bar(
                        region_major,
                        x='Region',
                        y='Count',
                        color='Count',
                        color_continuous_scale="Viridis",
                        title="Distribution of Major Earthquakes by Region"
                    )
                    
                    st.plotly_chart(fig_region_major, use_container_width=True)
                    
                    # Timeline of major events
                    st.subheader("Timeline of Major Earthquakes")
                    
                    fig_timeline = px.scatter(
                        major_quakes.sort_values('time'),
                        x='time',
                        y='mag',
                        color='region',
                        size='mag',
                        hover_name='place',
                        title="Major Earthquakes Timeline",
                        labels={'time': 'Date', 'mag': 'Magnitude'}
                    )
                    
                    # Add horizontal line at magnitude 8.0
                    fig_timeline.add_hline(y=8.0, line_dash="dash", line_color="red", annotation_text="M8.0+")
                    
                    fig_timeline.update_layout(height=500)
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Table of top 10 major events
                    st.subheader("Top 10 Most Powerful Earthquakes (2005-2025)")
                    
                    top10 = major_quakes.nlargest(10, 'mag').reset_index(drop=True)
                    top10['time'] = top10['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Add a rank column
                    top10.index = top10.index + 1
                    top10 = top10.rename_axis('Rank').reset_index()
                    
                    # Display the table
                    st.dataframe(
                        top10[['Rank', 'time', 'mag', 'place', 'depth']].rename(
                            columns={'time': 'Date', 'mag': 'Magnitude', 'place': 'Location', 'depth': 'Depth (km)'}
                        ),
                        use_container_width=True
                    )
                
                # Tab 5: Recurrence Analysis
                with hist_tabs[4]:
                    st.header("Recurrence Analysis and Forecasting")
                    
                    st.info("""
                    This analysis examines historical patterns of earthquake recurrence in different regions
                    to estimate probabilities of future seismic events. These are statistical estimates based
                    on historical data, not definitive predictions.
                    """)
                    
                    # Calculate recurrence intervals for regions with significant activity
                    st.subheader("Recurrence Analysis by Region")
                    
                    # Only analyze regions with multiple major events
                    region_counts = major_quakes['region'].value_counts()
                    regions_with_multiple = region_counts[region_counts >= 2].index.tolist()
                    
                    if not regions_with_multiple:
                        st.warning("Not enough major earthquakes in any single region for recurrence analysis")
                    else:
                        # Calculate recurrence statistics
                        recurrence_data = []
                        
                        for region in regions_with_multiple:
                            region_df = major_quakes[major_quakes['region'] == region].sort_values('time')
                            
                            # Calculate intervals between events
                            if len(region_df) >= 2:
                                time_diffs = region_df['time'].diff().dropna()
                                avg_interval_days = time_diffs.dt.total_seconds().mean() / (24*3600)
                                avg_interval_years = avg_interval_days / 365.25
                                
                                # Last event and estimate of next
                                last_event = region_df['time'].max()
                                next_estimate = last_event + pd.Timedelta(days=avg_interval_days)
                                current_time = pd.Timestamp.now()
                                if current_time.tz is not None:
                                    current_time = current_time.tz_localize(None)
                                if last_event.tz is not None:
                                    last_event = last_event.tz_localize(None)
                                time_since_last = (current_time - last_event).total_seconds() / (24*3600*365.25)
                                
                                # Calculate probability using Poisson distribution
                                lambda_param = 1 / avg_interval_years
                                prob_1yr = 1 - np.exp(-lambda_param * 1)
                                prob_5yr = 1 - np.exp(-lambda_param * 5)
                                
                                recurrence_data.append({
                                    'Region': region,
                                    'Events': len(region_df),
                                    'Avg Interval (years)': avg_interval_years,
                                    'Last Event': last_event,
                                    'Time Since Last (years)': time_since_last,
                                    'Next Estimate': next_estimate,
                                    'Probability (1 year)': prob_1yr,
                                    'Probability (5 years)': prob_5yr
                                })
                        
                        recurrence_df = pd.DataFrame(recurrence_data)
                        
                        # Format for display
                        display_df = recurrence_df.copy()
                        display_df['Last Event'] = display_df['Last Event'].dt.strftime('%Y-%m-%d')
                        display_df['Next Estimate'] = display_df['Next Estimate'].dt.strftime('%Y-%m-%d')
                        display_df['Avg Interval (years)'] = display_df['Avg Interval (years)'].round(2)
                        display_df['Time Since Last (years)'] = display_df['Time Since Last (years)'].round(2)
                        display_df['Probability (1 year)'] = (display_df['Probability (1 year)'] * 100).round(2).astype(str) + '%'
                        display_df['Probability (5 years)'] = (display_df['Probability (5 years)'] * 100).round(2).astype(str) + '%'
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Create probability visualization
                        st.subheader("Probability of Major Earthquake in Next 5 Years")
                        
                        fig_prob = px.bar(
                            recurrence_df,
                            x='Region',
                            y='Probability (5 years)',
                            color='Probability (5 years)',
                            color_continuous_scale='RdYlGn_r',  # Red for high probabilities
                            text=recurrence_df['Probability (5 years)'].apply(lambda x: f"{x*100:.1f}%"),
                            title="Probability of M7.0+ Earthquake in Next 5 Years by Region"
                        )
                        
                        fig_prob.update_traces(textposition='outside')
                        st.plotly_chart(fig_prob, use_container_width=True)
                        
                        # Create an interactive forecast map
                        st.subheader("Interactive Forecast Map")
                        
                        m = folium.Map(
                            location=[20, 0],
                            zoom_start=2,
                            tiles='CartoDB dark_matter'
                        )
                        
                        # Add circle markers for each region
                        for _, row in recurrence_df.iterrows():
                            # Get centroid coordinates for each region by filtering the data
                            region_data = data[data['region'] == row['Region']]
                            lat = region_data['latitude'].mean()
                            lon = region_data['longitude'].mean()
                            
                            # Determine color and size based on probability
                            prob = row['Probability (5 years)']
                            if prob > 0.75:
                                color = 'red'
                                radius = 300000  # in meters
                            elif prob > 0.5:
                                color = 'orange'
                                radius = 250000
                            elif prob > 0.25:
                                color = 'yellow'
                                radius = 200000
                            else:
                                color = 'green'
                                radius = 150000
                            
                            # Format popup content
                            popup_content = f"""
                            <div style='width:200px'>
                                <h4>{row['Region']}</h4>
                                <hr>
                                <b>Major Events:</b> {row['Events']}<br>
                                <b>Average Interval:</b> {row['Avg Interval (years)']:.2f} years<br>
                                <b>Last Event:</b> {row['Last Event'].strftime('%Y-%m-%d')}<br>
                                <b>Time Since Last:</b> {row['Time Since Last (years)']:.2f} years<br>
                                <hr>
                                <b>Probability (1yr):</b> {row['Probability (1 year)']*100:.1f}%<br>
                                <b>Probability (5yr):</b> {row['Probability (5 years)']*100:.1f}%<br>
                                <b>Next Estimated:</b> {row['Next Estimate'].strftime('%Y-%m-%d')}
                            </div>
                            """
                            
                            # Add circle to map
                            folium.Circle(
                                location=[lat, lon],
                                radius=radius,
                                color=color,
                                fill=True,
                                fill_color=color,
                                fill_opacity=0.4,
                                popup=folium.Popup(popup_content, max_width=250)
                            ).add_to(m)
                        
                        # Add legend to map
                        legend_html = """
                        <div style="position: fixed; 
                                    bottom: 50px; left: 50px; 
                                    border:2px solid grey; z-index:9999; font-size:14px;
                                    background-color: rgba(0, 0, 0, 0.7);
                                    color: white;
                                    padding: 10px;
                                    border-radius: 5px;">
                            <p><b>Probability of M7.0+ in Next 5 Years</b></p>
                            <p><i class="fa fa-circle" style="color:red"></i> Very High (>75%)</p>
                            <p><i class="fa fa-circle" style="color:orange"></i> High (50-75%)</p>
                            <p><i class="fa fa-circle" style="color:yellow"></i> Moderate (25-50%)</p>
                            <p><i class="fa fa-circle" style="color:green"></i> Low (<25%)</p>
                        </div>
                        """
                        
                        m.get_root().html.add_child(folium.Element(legend_html))
                        
                        # Display the map
                        folium_static(m, width=1000, height=600)
                        
                        # Disclaimer
                        st.warning("""
                        ⚠️ **Disclaimer:** These predictions are based solely on statistical analysis of historical data.
                        Earthquake forecasting has inherent limitations and uncertainties. This information should be used
                        for educational purposes only and not for making critical safety decisions.
                        """)
            
            else:
                st.error("Failed to load historical earthquake data. Please check the file path and format.")  

    # --- Data Table and Download ---
    with st.expander("View data in tabular format"):
        display_cols = [col for col in ['time', 'place', 'mag', 'depth', 'type', 'magType', 'rms'] if col in filtered_df.columns]
        sort_col = st.selectbox("Sort by", options=display_cols, index=0, key="sortcol")
        sort_order = st.radio("Order", options=['Descending', 'Ascending'], index=0, horizontal=True, key="sortorder")
        sorted_df = filtered_df.sort_values(by=sort_col, ascending=(sort_order == 'Ascending'))
        st.dataframe(sorted_df[display_cols], use_container_width=True)
        download_filtered_data(sorted_df, T)

# --- Sidebar About Section ---
st.sidebar.markdown("---")
st.sidebar.info(T["about"])



# Update Plotly color palette for magnitude categories
magnitude_colors = {
    'Minor (<2)': '#ffb347',      # yellowish orange
    'Light (2-4)': '#ffe066',     # light yellow
    'Moderate (4-6)': '#b22222',  # dark red
    'Strong (6+)': '#ff4444'      # bright red
}



# --- Footer ---
st.markdown("""
<footer>
    <p>© 2023 Earthquake Analysis and Prediction. Created by Alfonso Cifuentes Alonso.</p>
</footer>
""", unsafe_allow_html=True)