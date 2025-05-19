from __future__ import unicode_literals
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
import io
from sklearn.linear_model import LinearRegression
from dateutil import parser as date_parser
from tornado.websocket import websocket_connect
from tornado.ioloop import IOLoop
from tornado import gen
import logging
import json
import sys
import threading
import xml.etree.ElementTree as ET
import websocket

# --- WebSocket client ---
def on_message(ws, message):
    print("Received message:", message)

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws):
    print("WebSocket closed")

def on_open(ws):
    print("WebSocket opened")

def start_websocket():
    ws = websocket.WebSocketApp("wss://your-websocket-url",
                                  on_message=on_message,
                                  on_error=on_error,
                                  on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()

# Start WebSocket in a separate thread
threading.Thread(target=start_websocket).start()

# --- Define translations ---
T = {
    "dashboard_title": "🌋 Earthquake analysis and prediction",
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
    page_title="Earthquake analysis and prediction",
    page_icon="🌋",
    layout="wide"
)

# --- Page title ---
st.title(T["dashboard_title"])
st.markdown(T["dashboard_desc"])
df = None

# --- Custom CSS for styling ---
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
            
/* Cambia el color de fondo de st.info a verde */
.stAlert[data-testid="stAlertInfo"] {
    background: #2ecc40 !important;   /* Verde */
    color: #fff !important;
    border-left: 0.5rem solid #27ae60 !important;
}

/* Cambia el color de fondo de st.warning a amarillo */
.stAlert[data-testid="stAlertWarning"] {
    background: #ffe066 !important;   /* Amarillo */
    color: #333 !important;
    border-left: 0.5rem solid #ffb347 !important;
}
            
.stAlert[data-testid="stAlertSuccess"] {
    background: #2ecc40 !important;   /* Verde */
    color: #fff !important;
    border-left: 0.5rem solid #27ae60 !important;
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
        "Select date range",  # Added label
        value=st.session_state['date_range'],
        min_value=min_date,
        max_value=max_date,
        label_visibility="collapsed"  # Hide visually but keep for accessibility
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
        "Magnitude range",  # Add a descriptive label
        min_value=mag_min_val,
        max_value=mag_max_val,
        value=(mag_min_val, mag_max_val),
        step=0.1,
        key="mag_range_slider",
        label_visibility="collapsed"  # Hide label visually but keep for accessibility
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
        "Depth range",  # Add a descriptive label
        min_value=depth_min_val,
        max_value=depth_max_val,
        value=(depth_min_val, depth_max_val),
        step=0.5,
        key="depth_range_slider",
        label_visibility="collapsed"
    )
    
    # Event types filter
    st.sidebar.subheader(T["event_types"])
    all_types = df['type'].unique().tolist()
    
    if 'selected_types' not in st.session_state:
        st.session_state['selected_types'] = all_types
        
    selected_types = st.sidebar.multiselect(
        "Event types",
        options=all_types,
        default=st.session_state['selected_types']
    )
    
    # Region filter
    st.sidebar.subheader(T["region_filter"])
    regions = df['place'].str.split(', ').str[-1].unique().tolist()
    
    if 'selected_regions' not in st.session_state:
        st.session_state['selected_regions'] = []
        
    selected_regions = st.sidebar.multiselect(
        "Regions",
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
    # Apply filters from sidebar
    date_range, mag_range, depth_range, selected_types, selected_regions = sidebar_filters(df, T)
    
    # Filtering logic
    filtered_df = filter_data(df, date_range, mag_range, depth_range, selected_types, selected_regions)
    
    if len(filtered_df) == 0:
        st.warning("No data available with the selected filters. Please adjust the filters.")
    else:
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
        main_filtered_df = filtered_df.copy()
        main_tabs = st.tabs(["📊 General Summary", "🌐 Geographic Analysis", "⏱️ Temporal Analysis", "📈 Advanced Analysis", "🚨 Alert Center", "📚 Historical Analysis (2005-2025)"], )

        # --- General Summary Tab ---
        with main_tabs[0]:
            summary_df = main_filtered_df.copy()
            required_cols = ['mag', 'depth', 'place', 'id']
            missing_cols = [col for col in required_cols if col not in summary_df.columns]
            if missing_cols:
                st.error(f"Missing columns in the DataFrame: {', '.join(missing_cols)}")
            else:
                with st.container():
                    
                    col1, col2, col3, col4 = st.columns(4)

                    # Calculate metrics
                    total_events = len(summary_df)
                    avg_mag = summary_df['mag'].mean()
                    max_mag = summary_df['mag'].max()
                    avg_depth = summary_df['depth'].mean()

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
                    fig_mag = plot_magnitude_distribution(filtered_df)
                    # Personaliza el gráfico para hacerlo más atractivo
                    fig_mag.update_traces(marker_line_width=1.5, marker_line_color="#232526", showlegend=True)
                    fig_mag.update_layout(
                        plot_bgcolor="#232526",
                        paper_bgcolor="#232526",
                        font=dict(color="#ffb347", family="Montserrat"),
                        xaxis=dict(title="Magnitude", gridcolor="#444", zerolinecolor="#888"),
                        yaxis=dict(title="Frequency", gridcolor="#444", zerolinecolor="#888"),
                        bargap=0.15,
                        legend=dict(
                            bgcolor="rgba(35,37,38,0.7)",
                            bordercolor="#ffb347",
                            borderwidth=1,
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            title_text=" "  # Elimina el título de la leyenda
                        ),
                        title=" ",  # Elimina el título del gráfico
                        title_font=dict(size=22, color="#ffb347"),
                        hoverlabel=dict(bgcolor="#232526", font_size=14, font_family="Fira Mono")
                    )
                    # Añade anotación para el valor máximo, ocultando "undefined" si no hay datos
                    if not filtered_df['mag'].empty:
                        max_bin = filtered_df['mag'].round(1).value_counts().idxmax()
                        fig_mag.add_vline(
                            x=max_bin, line_dash="dash", line_color="#ffb347",
                            annotation_text=f"Peak: {max_bin}", annotation_position="top"
                        )
                    st.plotly_chart(fig_mag, use_container_width=True)
                with col_dist2:
                    st.subheader("Depth Distribution")
                    fig_depth = px.histogram(
                        filtered_df,
                        x="depth",
                        nbins=30,
                        color="magnitud_categoria",
                        color_discrete_map=magnitude_colors,
                        labels={"depth": "Depth (km)", "count": "Frequency", "magnitud_categoria": " "},
                    )
                    fig_depth.update_layout(
                        bargap=0.1,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            bgcolor="rgba(35,37,38,0.7)",
                            bordercolor="#ffb347",
                            borderwidth=1,
                            title_text=" "
                        ),
                        title=" ",
                        hoverlabel=dict(bgcolor="#232526", font_size=14, font_family="Fira Mono")
                    )
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
                    labels={"depth": "Depth (km)", "mag": "Magnitude", "magnitud_categoria": " "}
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
                    color_continuous_scale='Viridis',
                    labels={'Number of Events': 'Number<br>of events'}
                )
                fig_top.update_traces(textposition='outside', cliponaxis=False)
                fig_top.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    height=400,
                    margin=dict(l=120, r=200, t=40, b=40),  # Increased right margin
                    xaxis=dict(constrain='domain'),
                    autosize=True,
                    legend=dict(
                        x=0.05,
                        y=1,
                        xanchor='left',
                        yanchor='top'
                    )
                )
                st.plotly_chart(fig_top, use_container_width=True)

                # --- Data Table and Download ---
                with st.expander("View data in tabular format"):
                    display_cols = [col for col in ['time', 'place', 'mag', 'depth', 'type', 'magType', 'rms'] if col in filtered_df.columns]
                    sort_col = st.selectbox("Sort by", options=display_cols, index=0, key="sortcol")
                    sort_order = st.radio("Order", options=['Descending', 'Ascending'], index=0, horizontal=True, key="sortorder")
                    sorted_df = filtered_df.sort_values(by=sort_col, ascending=(sort_order == 'Ascending'))
                    st.dataframe(sorted_df[display_cols], use_container_width=True)
                    download_filtered_data(sorted_df, T)

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
                        "magnitud_categoria": False,  # Oculta la columna en el hover
                        "mag": ":.2f",
                        "depth": ":.2f km",
                        "time": True,
                        "type": True
                    },
                    color_discrete_map=magnitude_colors,
                    projection="natural earth"
                )
                # Oculta el título de la leyenda y el colorbar
                fig_map.update_layout(
                    legend_title_text="           ",  # Espacios para ocultar el título de la leyenda
                    margin={"r": 0, "t": 0, "l": 0, "b": 0},
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
                st.subheader("Significant Events (Magnitude ≥ 4.0)")
                significant_events = filtered_df[filtered_df['mag'] >= 4.0].sort_values(by='mag', ascending=False)
                if not significant_events.empty:
                    st.dataframe(
                        significant_events[['time', 'place', 'mag', 'depth', 'type']],
                        use_container_width=True
                    )
                else:
                    st.info("No events with magnitude ≥ 4.0 in the selected range.")

            with geo_tabs[1]:
                st.subheader("Seismic Activity Heat Map")
                st.markdown("This heat map shows areas with higher concentration of seismic activity. Brighter areas indicate higher density of events.")

                # Add padding/margin to the container div to prevent the map from being hidden
                with st.container():
                    st.markdown(
                        """
                        <div style="padding: 40px 2rem 40px 0;">
                        """,
                        unsafe_allow_html=True,
                    )
                    fig_heat = px.density_mapbox(
                        filtered_df,
                        lat="latitude",
                        lon="longitude",
                        z="mag",
                        radius=10,
                        center=dict(lat=filtered_df['latitude'].mean(), lon=filtered_df['longitude'].mean()),
                        zoom=1,
                        mapbox_style="carto-darkmatter",  # Use dark theme
                        opacity=0.8,
                        color_continuous_scale='Inferno'
                    )
                    fig_heat.update_layout(
                        margin={"r": 24, "t": 24, "l": 24, "b": 24},
                        height=600,
                        paper_bgcolor="black",
                        plot_bgcolor="black",
                        coloraxis_colorbar=dict(
                            x=0.85,  # Move colorbar to the left (default is 1.0)
                            xanchor='left',
                            len=0.7,
                            thickness=18,
                            bgcolor='rgba(35,37,38,0.7)',
                            bordercolor="#ffb347",
                            borderwidth=1,
                            outlinewidth=1,
                            tickcolor="#ffb347",
                            title_side='right'
                        )
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.subheader("Significant Events (Magnitude ≥ 4.0)")
                strong_events = filtered_df[filtered_df['mag'] >= 4.0].sort_values(by='mag', ascending=False)
                if not strong_events.empty:
                    st.dataframe(
                        strong_events[['time', 'place', 'mag', 'depth', 'type']],
                        use_container_width=True
                    )
                else:
                    st.info("No events with magnitude ≥ 4.0 in the selected range.")
         
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
                            "cluster_name": False,  # Hide cluster_name in hover
                            "mag": ":.2f",
                            "depth": ":.2f km",
                            "time": True
                        },
                        color_discrete_map=color_map,
                        projection="natural earth"
                    )
                    # Center the map on the globe
                    fig_cluster.update_geos(
                        center=dict(lat=0, lon=0),
                        projection_scale=1,
                        showcountries=True,
                        showcoastlines=True,
                        showland=True,
                        landcolor="#d2b48c",
                        oceancolor="#00356B",
                        bgcolor="black"
                    )
                    # Hide legend title and replace with spaces
                    fig_cluster.update_layout(
                        legend_title_text="            ",  # Spaces instead of "cluster_name"
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
                            bgcolor="black",
                            center=dict(lat=0, lon=0),
                            projection_scale=1
                        )
                    )
                    st.plotly_chart(fig_cluster, use_container_width=True)
                else:
                    st.warning("Not enough data to perform cluster analysis with current filters.")

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
                                    
                                    return df_volcano
                                except FileNotFoundError:
                                    # Si el archivo no existe, descarga un conjunto de datos de muestra
                                    url = "https://raw.githubusercontent.com/plotly/datasets/master/volcano_db.csv"
                                    df_volcano = pd.read_csv(url, encoding=encoding)
                                    
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
                        st.text("This section provides an overview of the global volcano dataset, including key statistics, type distributions, and country-level summaries.")
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
                            
                            # Add title before the chart
                            st.markdown("#### Distribution of Volcanoes by Activity Status")
                            
                            # Replace "Unknown" with "Dormant" in status categories
                            df_volcano['status_category'] = df_volcano['status_category'].replace('Unknown', 'Dormant')
                            
                            # Create a bar chart showing volcano status categories
                            status_counts = df_volcano['status_category'].value_counts()
                            fig_status = px.bar(
                                x=status_counts.index, 
                                y=status_counts.values,
                                labels={'x': 'Status', 'y': 'Count'},
                                color=status_counts.values,
                                color_continuous_scale=px.colors.sequential.Inferno
                            )
                            # Remove legend and add more margin space to prevent elements from being hidden
                            fig_status.update_layout(
                                showlegend=False,
                                margin=dict(l=50, r=175, t=30, b=50),  # Add generous margins on all sides
                                autosize=True,
                                height=400,  # Set a fixed height to ensure proper rendering
                                xaxis=dict(
                                    tickangle=0,
                                    tickmode='array',
                                    tickvals=list(range(len(status_counts))),
                                    ticktext=status_counts.index
                                )
                            )
                            # Ensure the chart container has padding
                            st.markdown('<div style="padding: 20px 0;">', unsafe_allow_html=True)
                            st.plotly_chart(fig_status, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                            # Add explanatory text below the chart
                            st.text("""
                            The chart above shows the distribution of volcanoes by their activity status. Active volcanoes have erupted in recent history or 
                            show signs of ongoing activity. Dormant volcanoes haven't erupted recently but may become active again in the future. 
                            Extinct volcanoes are considered to have no possibility of eruption. This classification helps scientists 
                            monitor volcanic hazards and prioritize observation efforts around the world.
                            """)
                        # Top countries by volcano count
                        if 'country' in df_volcano.columns:
                            st.subheader("Countries with Most Volcanoes")
                            country_counts = df_volcano['country'].value_counts().reset_index().head(15)
                            country_counts.columns = ['Country', 'Number of<br>Volcanoes']
                            
                            fig_countries = px.bar(
                                country_counts,
                                y='Country',
                                x='Number of<br>Volcanoes',
                                orientation='h',
                                color='Number of<br>Volcanoes',
                                
                                color_continuous_scale=px.colors.sequential.Inferno
                            )
                            fig_countries.update_layout(
                                yaxis={'categoryorder': 'total ascending'},
                                margin=dict(r=170)  # Added more right margin
                            )
                            st.plotly_chart(fig_countries, use_container_width=True)
                        
                        # Elevation distribution
                        if 'elevation' in df_volcano.columns:
                            st.subheader("Elevation Distribution")
                            
                            fig_elev = px.histogram(
                                df_volcano,
                                x='elevation',
                                nbins=50,
                                
                                color_discrete_sequence=['rgba(255, 102, 0, 0.8)']
                            )
                            fig_elev.update_layout(
                                xaxis_title='Elevation (meters)',
                                yaxis_title='Number of Volcanoes',
                                margin=dict(l=50, r=175, t=30, b=50)  # Added margins with 175px on right
                            )
                            st.plotly_chart(fig_elev, use_container_width=True)
                            
                            # Elevation by volcano type
                            if 'volcano_type_category' in df_volcano.columns:
                                fig_elev_type = px.box(
                                    df_volcano,
                                    x='volcano_type_category', 
                                    y='elevation',
                                  
                                    color='volcano_type_category',
                                    color_discrete_sequence=px.colors.sequential.Inferno
                                )
                                fig_elev_type.update_layout(
                                    xaxis_title='Volcano Type',
                                    yaxis_title='Elevation (meters)',
                                    showlegend=False,
                                    margin=dict(l=50, r=175, t=30, b=50)  # Added margins with 175px on right
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
                        # Replace 'Unknown' and 'undefined' with blank space for display
                        df_volcano_display = df_volcano.copy()
                        
                        if color_column in df_volcano_display.columns:
                            # Convierte todo a string y elimina espacios
                            df_volcano_display[color_column] = df_volcano_display[color_column].astype(str).str.strip()
                            # Filtra valores vacíos, NaN, "nan", "undefined" (en cualquier capitalización)
                            df_volcano_display = df_volcano_display[
                                df_volcano_display[color_column].notna() &
                                (df_volcano_display[color_column] != "") &
                                (df_volcano_display[color_column].str.lower() != "undefined") &
                                (df_volcano_display[color_column].str.lower() != "nan")
                            ]
                            
                            fig_volcano_map = px.scatter_geo(
                                df_volcano_display,
                                lat='latitude',
                                lon='longitude',
                                color=color_column,
                                hover_name='volcano_name',
                                hover_data={
                                    'country': True,
                                    'elevation': ':.0f m',
                                    'status_category': True,
                                    'latitude': False,
                                    'longitude': False
                                },
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

                            # Update map layout to match the earthquake map style and remove legend title
                            fig_volcano_map.update_layout(
                                legend_title_text="Volcano Type", 
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
                                    'last_known': True,
                                    'latitude': False,
                                    'longitude': False
                                },
                                labels={'last_known': 'Last Known'},
                     
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
                                   
                                    color=century_counts.values,
                                    color_continuous_scale=px.colors.sequential.Inferno
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
                                    color_continuous_scale=px.colors.sequential.Inferno,
                                    labels={'Number of Active Volcanoes': 'Number of<br>Active Volcanoes'}
                                )
                                fig_active.update_layout(
                                    yaxis={'categoryorder': 'total ascending'},
                                    coloraxis_colorbar=dict(
                                        title='Number of<br>Active Volcanoes',
                                        x=0.85,  # mueve la barra de color más a la izquierda
                                        xanchor='left',
                                        title_side='right'
                                       
                                    ),
                                    margin=dict(r=120)  # más margen derecho para que no se corte la leyenda
                                )
                                # Remove the colorbar label at the bottom (x-axis title)
                                fig_active.update_xaxes(title_text=None)
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
                                        color_continuous_scale='Inferno'
                                    )

                                    # Improve hover template
                                    fig_type_activity.update_traces(
                                        hovertemplate="Type: %{y}<br>Activity: %{x}<br>Percentage: %{z:.1f}%<extra></extra>"
                                    )

                                    # Add proper margins to ensure nothing is cut off
                                    fig_type_activity.update_layout(
                                        height=500,  # Increase height for better visualization
                                        margin=dict(l=50, r=150, t=50, b=50),  # Increased right margin to 150
                                        xaxis=dict(side='bottom'),  # Ensure x-axis labels are at the bottom
                                        yaxis=dict(side='left'),    # Ensure y-axis labels are on the left
                                    )

                                    # Adjust colorbar position to avoid overlap
                                    fig_type_activity.update_layout(
                                        coloraxis_colorbar=dict(
                                            len=0.8,  # Shorter colorbar
                                            thickness=20,  # Thicker colorbar for better visibility
                                            x=1.02,   # Position more to the left (was 1.05)
                                            y=0.5,    # Center vertically
                                            xanchor='left',  # Anchor to left side
                                            yanchor='middle'  # Anchor to middle vertically
                                        ),
                                    )

                                    st.plotly_chart(fig_type_activity, use_container_width=True)
                        # Educational content
                        with st.expander("📚 Learn About Different Volcano Types"):
                            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Volcano_types.svg/1200px-Volcano_types.svg.png", use_container_width=True)
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
                        in similar geographical regions due to shared underlying causes.
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
                        
                        # Update map layout
                        fig_combined.update_layout(
                            
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
                                min_value=500,
                                max_value=10000,
                                value=5000,
                                step=500
                            )

                            # Limit to 100 most recent earthquakes
                            max_quakes_to_analyze = 100
                            recent_earthquakes = filtered_df.sort_values('time', ascending=False).head(max_quakes_to_analyze)

                            # Limit to 50 volcanoes (e.g., by elevation or just first 50)
                            if 'elevation' in df_volcano.columns:
                                selected_volcanoes = df_volcano.sort_values('elevation', ascending=False).head(50)
                            else:
                                selected_volcanoes = df_volcano.head(50)

                            with st.spinner("Calculating proximity relationships..."):
                                proximity_results = []

                                for idx, quake in recent_earthquakes.iterrows():
                                    distances = []
                                    for _, volcano in selected_volcanoes.iterrows():
                                        dist = calculate_distance(
                                            quake['latitude'], quake['longitude'],
                                            volcano['latitude'], volcano['longitude']
                                        )
                                        distances.append((dist, volcano['volcano_name'], volcano['country']))
                                    nearest = min(distances, key=lambda x: x[0])
                                    proximity_results.append({
                                        'earthquake_time': quake['time'],
                                        'earthquake_mag': quake['mag'],
                                        'earthquake_place': quake['place'],
                                        'nearest_volcano': nearest[1],
                                        'volcano_country': nearest[2],
                                        'distance_km': nearest[0]
                                    })

                                proximity_df = pd.DataFrame(proximity_results)

                                near_volcano_count = sum(proximity_df['distance_km'] <= distance_threshold)
                                near_volcano_percentage = (near_volcano_count / len(proximity_df)) * 100

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
                                close_pairs = proximity_df[proximity_df['distance_km'] <= distance_threshold].sort_values('distance_km')
                                if len(close_pairs) > 0:
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

        # --- Temporal Analysis Tab ---
        with main_tabs[2]:
            # Usar copia local para esta pestaña
            temporal_df = main_filtered_df.copy()
            
            # Verificar columnas necesarias
            required_cols = ['time', 'day', 'hour', 'day_of_week', 'id', 'mag']
            missing_cols = [col for col in required_cols if col not in temporal_df.columns]
            
            if missing_cols:
                st.warning(f"Missing required columns for Temporal Analysis: {', '.join(missing_cols)}")
            elif len(temporal_df) == 0:
                st.warning("No data available for temporal analysis. Please adjust the filters.")
            else:
                # Crear columnas derivadas si no existen
                if 'day' not in temporal_df.columns and 'time' in temporal_df.columns:
                    temporal_df['day'] = temporal_df['time'].dt.date
                
                if 'hour' not in temporal_df.columns and 'time' in temporal_df.columns:
                    temporal_df['hour'] = temporal_df['time'].dt.hour
                    
                if 'day_of_week' not in temporal_df.columns and 'time' in temporal_df.columns:
                    temporal_df['day_of_week'] = temporal_df['time'].dt.day_name()
                
                    
                # Create tabs for different temporal analyses
                temp_tab1, temp_tab2, temp_tab3 = st.tabs([
                    "Daily Evolution", 
                    "Weekly Patterns",
                    "Hourly Patterns"
                ])
                
                # Tab 1: Daily Evolution
                with temp_tab1:
                    st.subheader("Daily Evolution of Seismic Activity")
                    try:
                        
                        
                        # Group by day
                    
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
                        # Usar copia local para este subtab
                        weekly_df = temporal_df.copy()
                        
                        # Translate days of the week
                        if 'day_of_week' in weekly_df.columns:
                            weekly_df['day_name'] = weekly_df['day_of_week'].map(days_translation)
                        elif 'time' in weekly_df.columns:
                            weekly_df['day_name'] = weekly_df['time'].dt.day_name().map(days_translation)
                        else:
                            st.warning("Missing required columns for weekly analysis")
                            pass
                        
                        # Sort days of the week correctly
                        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        ordered_days = [days_translation[day] for day in days_order]
                        
                        # Group by day of the week
                        dow_data = weekly_df.groupby('day_name').agg({
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
                           
                            color_continuous_scale='Viridis',
                            labels={'Count': 'Number of Events', 'Average Magnitude': 'Average Magnitude'}
                        )
                        
                        fig_dow.update_traces(textposition='outside')
                        fig_dow.update_layout(height=400)
                        
                        st.plotly_chart(fig_dow, use_container_width=True)
                        
                        # Add weekly heatmap
                        st.subheader("Heat Map: Activity by Week and Day")
                        
                        # Add a relative week number column within the period
                        if 'time' in weekly_df.columns:
                            weekly_df['week_num'] = weekly_df['time'].dt.isocalendar().week
                            min_week = weekly_df['week_num'].min()
                            weekly_df['rel_week'] = weekly_df['week_num'] - min_week + 1
                            
                            # Group by relative week and day of the week
                            heatmap_weekly = weekly_df.groupby(['rel_week', 'day_name']).size().reset_index(name='count')
                            
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
                               
                            )
                            
                            st.plotly_chart(fig_weekly_heat, use_container_width=True)
                        else:
                            st.warning("Cannot create weekly heatmap: missing time column")
                    except Exception as e:
                        st.error(f"Error in weekly pattern analysis: {e}")

                # Tab 3: Hourly Patterns
                with temp_tab3:
                    try:
                        # Usar copia local para este subtab
                        hourly_df = temporal_df.copy()
                        
                        if 'hour' not in hourly_df.columns and 'time' in hourly_df.columns:
                            hourly_df['hour'] = hourly_df['time'].dt.hour
                        
                        if 'hour' not in hourly_df.columns:
                            st.warning("Missing hour data for hourly analysis")
                        else:
                            st.subheader("Events Distribution by Hour of Day")
                            
                            # Group by hour
                            hourly_counts = hourly_df.groupby('hour').agg({
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
                                
                                labels={"Hour": "Hour of day (UTC)", "Count": "Number of events"},
                                color_continuous_scale='Viridis',
                                text='Count'
                            )
                            
                            fig_hourly.update_traces(textposition='outside')
                            fig_hourly.update_layout(height=400)
                            
                            st.plotly_chart(fig_hourly, use_container_width=True)
                            
                            # Heatmap by hour and day of the week
                            if 'day_of_week' in hourly_df.columns or 'time' in hourly_df.columns:
                                if 'day_name' not in hourly_df.columns:
                                    if 'day_of_week' in hourly_df.columns:
                                        hourly_df['day_name'] = hourly_df['day_of_week'].map(days_translation)
                                    else:
                                        hourly_df['day_name'] = hourly_df['time'].dt.day_name().map(days_translation)
                                
                                st.subheader("Heat Map: Activity by Hour and Day of the Week")
                                
                                # Group by hour and day of the week
                                heatmap_data = hourly_df.groupby(['day_name', 'hour']).size().reset_index(name='count')
                                
                                # Pivot to create the format for the heatmap
                                pivot_data = pd.pivot_table(
                                    heatmap_data, 
                                    values='count', 
                                    index='day_name', 
                                    columns='hour',
                                    fill_value=0
                                )
                                
                                # Reorder the days
                                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
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
                            else:
                                st.warning("Cannot create day-hour heatmap: missing day of week data")
                    except Exception as e:
                        st.error(f"Error in hourly pattern analysis: {e}")

        # --- Advanced Analysis Tab ---
        with main_tabs[3]:
            # Usar copia local para esta pestaña
            advanced_df = main_filtered_df.copy()
                    
            adv_tab1, adv_tab2, adv_tab3 = st.tabs([
                "Correlations", 
                "Magnitude by Region", 
                "Comparisons",
            ])
            
            # Tab 1: Correlations
            with adv_tab1:
                try:
                    # Usar copia local para este subtab
                    corr_df = advanced_df.copy()
                    
                    st.subheader("Correlation Matrix")
                    
                    # Select variables for correlation
                    corr_cols = ['mag', 'depth', 'rms', 'gap', 'horizontalError', 'depthError']
                    
                    # Filter columns that exist in the DataFrame
                    valid_cols = [col for col in corr_cols if col in corr_df.columns]

                    if len(valid_cols) > 1:
                        corr_data = corr_df[valid_cols].dropna()
                        
                        if len(corr_data) > 1:  # Ensure there's enough data for correlation
                            corr_matrix = corr_data.corr()
                            
                            fig_corr = px.imshow(
                                corr_matrix,
                                text_auto=True,
                                color_continuous_scale="RdBu_r",
                                
                                aspect="auto"
                            )
                            
                            st.plotly_chart(fig_corr, use_container_width=True)
                            
                            st.markdown("""
                            **Correlation matrix interpretation:**
                            - Values close to 1 indicate strong positive correlation
                            - Values close to -1 indicate strong negative correlation
                            - Values close to 0 indicate little or no correlation
                            """)
                        else:
                            st.warning("Not enough data to calculate correlations.")
                    else:
                        st.warning("Not enough numeric columns to calculate correlations.")
                except Exception as e:
                    st.error(f"Error in correlation analysis: {e}")

            # Tab 2: Magnitude by Region
            with adv_tab2:
                try:
                    # Usar copia local para este subtab
                    region_df = advanced_df.copy()
                    
                    st.subheader("Magnitude Analysis by Region")
                    
                    # Verificar columnas necesarias
                    if 'place' not in region_df.columns:
                        st.warning("Missing 'place' column for region analysis")
                    else:
                        # Extract main regions
                        if 'region' not in region_df.columns:
                            region_df['region'] = region_df['place'].str.split(', ').str[-1]
                        
                        region_stats = region_df.groupby('region').agg({
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
                              
                                color_continuous_scale=px.colors.sequential.Plasma
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
                    # Usar copia local para este subtab
                    compare_df = advanced_df.copy()
                    
                    st.subheader("Comparative Analysis")
                    
                    # Available numeric columns
                    numeric_cols = compare_df.select_dtypes(include=['number']).columns.tolist()
                    
                    # Filtrar columnas que no queremos usar para comparación
                    exclude_cols = ['cluster', 'rel_week', 'week_num', 'id']
                    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
                    
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
                            y_options = [col for col in numeric_cols if col != x_variable]
                            y_default = 'depth' if 'depth' in y_options else y_options[0]
                            y_variable = st.selectbox(
                                "Y Variable",
                                options=y_options,
                                index=y_options.index(y_default) if y_default in y_options else 0
                            )
                        
                        # Category column for color
                        if 'magnitud_categoria' not in compare_df.columns and 'mag' in compare_df.columns:
                            conditions = [
                                (compare_df['mag'] < 2.0),
                                (compare_df['mag'] >= 2.0) & (compare_df['mag'] < 4.0),
                                (compare_df['mag'] >= 4.0) & (compare_df['mag'] < 6.0),
                                (compare_df['mag'] >= 6.0)
                            ]
                            choices = ['Minor (<2)', 'Light (2-4)', 'Moderate (4-6)', 'Strong (6+)']
                            compare_df['magnitud_categoria'] = np.select(conditions, choices, default='Not classified')
                        
                        # Create scatter plot
                        if 'magnitud_categoria' in compare_df.columns:
                            # Con categoría de magnitud
                            fig_custom = px.scatter(
                                compare_df,
                                x=x_variable,
                                y=y_variable,
                                color='magnitud_categoria',
                                size=ensure_positive(compare_df['mag']) if 'mag' in compare_df.columns else None,
                                hover_name='place' if 'place' in compare_df.columns else None,
                                title=f"Relationship between {x_variable} and {y_variable}",
                                color_discrete_map=magnitude_colors if 'magnitude_colors' in globals() else None
                            )
                        else:
                            # Sin categoría de magnitud
                            fig_custom = px.scatter(
                                compare_df,
                                x=x_variable,
                                y=y_variable,
                                color=x_variable,
                                size='mag' if 'mag' in compare_df.columns else None,
                                hover_name='place' if 'place' in compare_df.columns else None,
                                title=f"Relationship between {x_variable} and {y_variable}"
                            )
                        
                        st.plotly_chart(fig_custom, use_container_width=True)
                        
                        # Añadir análisis de regresión si hay suficientes datos
                        if len(compare_df) >= 10:
                            st.subheader("Regression Analysis")
                            
                            # Calcular regresión
                            x = compare_df[x_variable].values.reshape(-1, 1)
                            y = compare_df[y_variable].values
                            

                            model = LinearRegression()
                            model.fit(x, y)
                            
                            # Mostrar resultados
                            r_squared = model.score(x, y)
                            slope = model.coef_[0]
                            intercept = model.intercept_
                            
                            st.write(f"R² (coefficient of determination): {r_squared:.4f}")
                            st.write(f"Equation: {y_variable} = {slope:.4f} × {x_variable} + {intercept:.4f}")
                            
                            # Añadir línea de tendencia al gráfico
                            trend_x = np.array([min(compare_df[x_variable]), max(compare_df[x_variable])])
                            trend_y = model.predict(trend_x.reshape(-1, 1))
                            
                            fig_custom.add_trace(
                                go.Scatter(
                                    x=trend_x, 
                                    y=trend_y,
                                    mode='lines',
                                    name='Trend Line',
                                    line=dict(color='red', width=2, dash='dash')
                                )
                            )
                            
                            st.plotly_chart(fig_custom, use_container_width=True)
                    else:
                        st.warning("Not enough numeric columns to perform comparative analysis.")
                except Exception as e:
                    st.error(f"Error in comparative analysis: {e}")
        
        # --- Alert Center Tab ---
        with main_tabs[4]:
            st.markdown("## 🚨 Multi-Source Alert Center")
            st.text("Este módulo verifica múltiples fuentes de datos sobre alertas sísmicas recientes y notifica sobre eventos significativos.")

            # -------------------------------------------------
            # USGS Significant Events
            # -------------------------------------------------
            st.markdown("### 📡 USGS Significant Events")
            st.markdown("Displaying significant earthquake events from the USGS feed")

            try:
                # Get significant events from USGS
                sig_url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_week.csv"

                # Show loading indicator
                with st.spinner("Verifying USGS alerts..."):
                    # Try to load data with timeout
                    try:
                        timeout_seconds = 10  # Set reasonable timeout
                        response = requests.get(sig_url, timeout=timeout_seconds)
                        response.raise_for_status()
                        sig_df = pd.read_csv(io.StringIO(response.text))
                        sig_df['time'] = pd.to_datetime(sig_df['time']).dt.tz_localize(None)

                        # Filter events from the last 24 hours
                        last_24h = pd.Timestamp.now() - pd.Timedelta(hours=24)
                        recent_sig = sig_df[sig_df['time'] > last_24h]

                        # Check if there are recent alerts
                        if not recent_sig.empty:
                            # Create alert with color and emoji based on magnitude
                            for _, event in recent_sig.iterrows():
                                mag = event['mag']
                                place = event['place']
                                event_time = event['time']

                                # Color and emoji based on magnitude
                                if (mag >= 7.0):
                                    st.error(f"🔴 ⚠️ CRITICAL ALERT: Magnitude {mag:.1f} in {place} ({event_time})")
                                elif (mag >= 6.0):
                                    st.warning(f"🟠 ⚠️ HIGH ALERT: Magnitude {mag:.1f} in {place} ({event_time})")
                                else:
                                    st.info(f"🔵 ℹ️ ALERT: Magnitude {mag:.1f} in {place} ({event_time})")

                            # Show table of recent significant events
                            st.dataframe(
                                recent_sig[['time', 'place', 'mag', 'depth', 'type']],
                                use_container_width=True,
                                height=250  # Fixed height to avoid excessive scrolling
                            )
                        else:
                            # Changed to success (green background) as requested
                            st.success("✓ No significant USGS alerts in the last 24 hours")
                    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                        st.warning(f"Could not connect to USGS API: {e}. Check your internet connection.")
                        # Provide fallback information
                        st.info("You can check alerts manually at: https://earthquake.usgs.gov/earthquakes/map/")
            except Exception as e:
                st.error(f"Error processing USGS data: {e}")
                # Provide a fallback
                st.info("Unable to load recent earthquake alerts. Try refreshing the page or check USGS website directly.")
            
            # -------------------------------------------------
            # Latest 10 USGS Alerts
            # -------------------------------------------------
            st.markdown("### 🔄 Latest 10 USGS Alerts")
            st.markdown("Shows the most recent earthquake events regardless of significance")
            
            # Function to get the latest 10 alerts from USGS
            @st.cache_data(ttl=300)  # Cache for 5 minutes
            def get_latest_usgs_alerts():
                try:
                    # USGS API URL for latest earthquakes (all, not just significant ones)
                    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson"
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Process the data
                    alerts = []
                    if 'features' in data:
                        for feature in data['features'][:10]:  # Take only the first 10
                            if 'properties' in feature:
                                props = feature['properties']
                                geometry = feature.get('geometry', {})
                                coordinates = geometry.get('coordinates', [0, 0, 0]) if geometry else [0, 0, 0]
                                
                                alerts.append({
                                    'time': props.get('time', ''),
                                    'place': props.get('place', ''),
                                    'magnitude': props.get('mag', 0),
                                    'type': props.get('type', ''),
                                    'status': props.get('status', ''),
                                    'depth': props.get('depth', 0) if 'depth' in props else coordinates[2],
                                    'latitude': coordinates[1] if len(coordinates) > 1 else 0,
                                    'longitude': coordinates[0] if len(coordinates) > 0 else 0
                                })
                    return alerts
                except Exception as e:
                    st.error(f"Error retrieving USGS alerts: {e}")
                    return []
                
            # Get and display the alerts
            with st.spinner("Retrieving latest USGS alerts..."):
                usgs_alerts = get_latest_usgs_alerts()
            
            if usgs_alerts:
                # Convert to DataFrame for display
                alerts_df = pd.DataFrame(usgs_alerts)
                
                # Convert timestamp
                if 'time' in alerts_df.columns:
                    try:
                        alerts_df['time'] = pd.to_datetime(alerts_df['time'], unit='ms')
                    except:
                        pass  # If already a string, leave it as is
                        
                    # Show critical alerts first
                    high_mag_alerts = [a for a in usgs_alerts if a.get('magnitude', 0) >= 6.0]
                    for alert in high_mag_alerts:
                        mag = alert.get('magnitude', 0)
                        place = alert.get('place', "Unknown location")
                        time = alert.get('time', "")
                        if isinstance(time, (int, float)):
                            time = pd.to_datetime(time, unit='ms')
                        if mag >= 7.0:
                            st.error(f"🔴 ⚠️ CRITICAL ALERT: Magnitude {mag:.1f} in {place} ({time}) [USGS]")
                        elif mag >= 6.0:
                            st.warning(f"🟠 ⚠️ HIGH ALERT: Magnitude {mag:.1f} in {place} ({time}) [USGS]")
                    
                    # Show complete table with fixed height
                    st.dataframe(
                        alerts_df.rename(columns={
                            'time': 'Time',
                            'place': 'Location',
                            'magnitude': 'Magnitude',
                            'depth': 'Depth (km)',
                            'type': 'Type'
                        }),
                        use_container_width=True,
                        height=400  # Fixed height
                    )
                    
                    # Add map to show locations of latest alerts
                    if 'latitude' in alerts_df.columns and 'longitude' in alerts_df.columns:
                        st.subheader("Latest USGS Alerts Map")
                        
                        fig_alerts = px.scatter_mapbox(
                            alerts_df,
                            lat="latitude",
                            lon="longitude",
                            color="magnitude",
                            size="magnitude",
                            size_max=15,
                            hover_name="place",
                            hover_data=["time", "type", "depth"],
                            color_continuous_scale=px.colors.sequential.Plasma,
                            zoom=1,
                            height=500,
                            mapbox_style="carto-darkmatter"
                        )
                        
                        fig_alerts.update_layout(
                            margin={"r":0,"t":0,"l":0,"b":0},
                            coloraxis_colorbar=dict(
                                title="Magnitude",
                                x=0.85,
                                xanchor='left'
                            )
                        )
                        
                        st.plotly_chart(fig_alerts, use_container_width=True)
                    else:
                        # If coordinates aren't available in the alerts_df, extract them from the place field or show a message
                        st.info("Map view not available - geographic coordinates not found in the data")
            else:
                st.info("No USGS alerts available or could not retrieve data")
            
            # Link to USGS website
            st.markdown("[Visit USGS Website for more information](https://earthquake.usgs.gov/earthquakes/map/)")

            # -------------------------------------------------
            # EMSC Latest Earthquakes Table
            # -------------------------------------------------
            st.markdown("### 🌎 EMSC Latest Earthquakes")
            st.markdown("Showing the most recent earthquakes from the European-Mediterranean Seismological Centre")
            
            @st.cache_data(ttl=300)  # Cache for 5 minutes
            def get_emsc_latest_earthquakes():
                try:
                    # EMSC provides a JSON feed for latest earthquakes
                    # This is the official API endpoint used by the EMSC website
                    url = "https://www.seismicportal.eu/fdsnws/event/1/query?format=json&limit=10"
                    response = requests.get(url, timeout=15)
                    response.raise_for_status()
                    data = response.json()
                    
                    earthquakes = []
                    if 'features' in data:
                        for feature in data['features']:
                            props = feature['properties']
                            geometry = feature['geometry']
                            
                            # Extract coordinates
                            coordinates = geometry.get('coordinates', [0, 0, 0])
                            
                            earthquakes.append({
                                'time': props.get('time', ''),
                                'magnitude': props.get('mag', 0),
                                'depth': props.get('depth', 0),
                                'region': props.get('flynn_region', props.get('region', '')),
                                'latitude': coordinates[1] if len(coordinates) > 1 else 0,
                                'longitude': coordinates[0] if len(coordinates) > 0 else 0,
                                'auth': props.get('auth', '')
                            })
                    return earthquakes
                except Exception as e:
                    st.error(f"Error retrieving EMSC data: {e}")
                    return []
            
            # Get and display EMSC data
            with st.spinner("Retrieving EMSC latest earthquakes..."):
                emsc_earthquakes = get_emsc_latest_earthquakes()
            
            if emsc_earthquakes:
                # Convert to DataFrame
                emsc_df = pd.DataFrame(emsc_earthquakes)
                
                # Process time column
                if 'time' in emsc_df.columns:
                    try:
                        # EMSC time format is ISO 8601
                        emsc_df['time'] = pd.to_datetime(emsc_df['time'])
                    except:
                        pass
                
                # Show high magnitude alerts first
                high_mag_emsc = [eq for eq in emsc_earthquakes if eq.get('magnitude', 0) >= 6.0]
                for eq in high_mag_emsc:
                    mag = eq.get('magnitude', 0)
                    region = eq.get('region', "Unknown location")
                    time = eq.get('time', "")
                    if mag >= 7.0:
                        st.error(f"🔴 ⚠️ CRITICAL ALERT: Magnitude {mag:.1f} in {region} ({time}) [EMSC]")
                    elif mag >= 6.0:
                        st.warning(f"🟠 ⚠️ HIGH ALERT: Magnitude {mag:.1f} in {region} ({time}) [EMSC]")
                
                # Display the data with fixed height
                st.dataframe(
                    emsc_df.rename(columns={
                        'time': 'Time',
                        'magnitude': 'Magnitude',
                        'depth': 'Depth (km)',
                        'region': 'Region',
                        'auth': 'Agency'
                    }),
                    use_container_width=True,
                    height=400  # Fixed height
                )
                
                # Draw a map of the latest EMSC earthquakes
                st.subheader("EMSC Latest Earthquakes Map")
                
                fig_emsc = px.scatter_mapbox(
                    emsc_df,
                    lat="latitude",
                    lon="longitude",
                    color="magnitude",
                    size="magnitude",
                    hover_name="region",
                    hover_data=["time", "depth", "auth"],
                    color_continuous_scale=px.colors.sequential.Plasma,
                    zoom=1,
                    height=500
                )
                
                fig_emsc.update_layout(
                    mapbox_style="carto-darkmatter",
                    margin={"r":0,"t":0,"l":0,"b":0}
                )
                
                st.plotly_chart(fig_emsc, use_container_width=True)
            else:
                st.info("No EMSC earthquake data available or could not retrieve data")
            
            # Link to EMSC website
            st.markdown("[Visit EMSC Website for more information](https://www.emsc-csem.org/)")

            # -------------------------------------------------
            # EMSC Real-Time Feed Section
            # -------------------------------------------------
            st.markdown("### 🌍 EMSC Real-Time Earthquake Feed")
            st.markdown("""
            This section connects to EMSC's real-time WebSocket feed showing earthquakes as they happen around the world. 
            Events appear in real-time without needing to refresh.
            """)

            # EMSC WebSocket configuration
            emsc_ws_status = st.empty()

            # Store events in session_state for persistence between refreshes
            if "emsc_events" not in st.session_state:
                st.session_state.emsc_events = []

            # Initialize controls to filter events
            magnitude_filter = st.slider(
                "Filter by minimum magnitude",
                min_value=1.0,
                max_value=8.0,
                value=4.5,
                step=0.5,
                key="emsc_mag_filter"
            )
            
            # Option to clear events
            if st.button("Clear Events"):
                st.session_state.emsc_events = []
                st.success("Event list cleared")

            # Class to handle WebSocket connection in a separate thread
            class EMSCWebSocketClient:
                def __init__(self):
                    self.echo_uri = 'wss://www.seismicportal.eu/standing_order/websocket'
                    self.connected = False
                    self.thread = None
                    self.events = []
                
                def start(self):
                    if self.thread is None or not self.thread.is_alive():
                        self.thread = threading.Thread(target=self._run_websocket, daemon=True)
                        self.thread.start()
                        
                def _process_message(self, message):
                    try:
                        data = json.loads(message)
                        if 'data' in data and 'properties' in data['data']:
                            info = data['data']['properties']
                            action = data['action'] if 'action' in data else 'unknown'
                        
                            # Create event dictionary
                            event = {
                                "time": info.get("time", ""),
                                "magnitude": float(info.get("mag", 0)),
                                "region": info.get("flynn_region", info.get("region", "")),
                                "depth": float(info.get("depth", 0)),
                                "lat": info.get("lat", 0),
                                "lon": info.get("lon", 0),
                                "action": action,
                                "id": info.get("unid", ""),
                                "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                                "source": info.get("auth", "EMSC")
                            }
                            
                            # Add to session state instead of local list
                            if "emsc_events" in st.session_state:
                                # Only add event if it doesn't already exist (check by ID)
                                ids = [e.get('id', '') for e in st.session_state.emsc_events]
                                if event['id'] not in ids:
                                    st.session_state.emsc_events.insert(0, event)
                                    # Limit to 100 events
                                    if len(st.session_state.emsc_events) > 100:
                                        st.session_state.emsc_events = st.session_state.emsc_events[:100]
                    except Exception as e:
                        print(f"Error processing WebSocket message: {e}")
                
                def _run_websocket(self):
                    import websocket
                    
                    def on_message(ws, message):
                        self._process_message(message)
                        
                    def on_error(ws, error):
                        print(f"WebSocket error: {error}")
                        self.connected = False
                        
                    def on_close(ws, close_status_code, close_msg):
                        print("WebSocket connection closed")
                        self.connected = False
                        
                    def on_open(ws):
                        print("WebSocket connection opened")
                        self.connected = True
                    
                    # Connect to WebSocket
                    ws = websocket.WebSocketApp(
                        self.echo_uri,
                        on_message=on_message,
                        on_error=on_error,
                        on_close=on_close,
                        on_open=on_open
                    )
                    
                    ws.run_forever()

            # Create and start WebSocket client
            if "ws_client" not in st.session_state:
                st.session_state.ws_client = EMSCWebSocketClient()
                st.session_state.ws_client.start()
                # Changed to success (green background) as requested
                emsc_ws_status.success("✓ Connected to EMSC real-time feed")
            
            # Get filtered events from session_state
            filtered_events = [
                event for event in st.session_state.emsc_events 
                if event.get('magnitude', 0) >= magnitude_filter
            ]
            
            # Display events
            if filtered_events:
                # Create a styled container for real-time events
                event_container = st.container()
                with event_container:
                    # First show high-magnitude alerts
                    high_mag_events = [e for e in filtered_events if e.get('magnitude', 0) >= 6.0]
                    for event in high_mag_events:
                        mag = event.get('magnitude', 0)
                        region = event.get('region', "Unknown location")
                        time = event.get('time', "")
                        if mag >= 7.0:
                            st.error(f"🔴 ⚠️ CRITICAL ALERT: Magnitude {mag:.1f} in {region} ({time}) [EMSC]")
                        elif mag >= 6.0:
                            st.warning(f"🟠 ⚠️ HIGH ALERT: Magnitude {mag:.1f} in {region} ({time}) [EMSC]")
                    
                    # Then show table of events with fixed height
                    events_df = pd.DataFrame(filtered_events)
                    if not events_df.empty:
                        display_cols = ["time", "magnitude", "region", "depth", "action", "source"]
                        display_cols = [col for col in display_cols if col in events_df.columns]
                        st.dataframe(
                            events_df[display_cols], 
                            use_container_width=True,
                            height=400  # Fixed height to prevent excessive scrolling
                        )
            else:
                # Changed to warning (yellow background) as requested
                st.warning("Waiting for earthquakes meeting the magnitude threshold...")

            # -------------------------------------------------
            # Alert System Information
            # -------------------------------------------------
            st.markdown("## Alert System Information")
            
            # Alert Levels
            st.markdown("### Alert Levels")
            alert_data = {
                "Level": ["Info", "Warning", "Alert", "Critical"],
                "Magnitude": ["< 5.0", "5.0 - 5.9", "6.0 - 6.9", "≥ 7.0"],
                "Color": ["Blue", "Yellow", "Orange", "Red"],
                "Icon": ["🔵", "🟡", "🟠", "🔴"]
            }
            st.dataframe(pd.DataFrame(alert_data), use_container_width=True)
            
            # Early Warning Systems
            st.markdown("### About Early Warning Systems")
            st.markdown("""
            Earthquake early warning systems detect the initial seismic waves (P waves) that travel faster 
            but cause less damage, providing a brief window of seconds to minutes before the more destructive 
            waves (S waves) arrive.
            
            Major systems currently in operation:
            - ShakeAlert (USA)
            - Sistema de Alerta Sísmica Mexicano (Mexico)
            - Earthquake Early Warning System (Japan)
            """)
            
            st.warning("""
            **Disclaimer:** This dashboard displays earthquake data with minimal delay, but it is NOT 
            an official early warning system. For official alerts, follow guidance from your local 
            geological survey or emergency management agency.
            """)

        # --- Historical Earthquake Analysis Tab ---
        with main_tabs[5]:
            st.title("Historical Earthquake Analysis (2005-2025)")
            
            # Add introduction text as requested
            st.markdown("""
            ## Introduction to Historical Analysis
            
            This section analyzes the 500 most powerful earthquakes recorded globally over the last two decades (2005-2025). 
            By examining these significant seismic events, we can identify patterns, recurring cycles, and geographic 
            distributions that contribute to our understanding of earthquake behavior. This historical perspective 
            provides valuable context for current seismic activity and supports more informed forecasting.
            
            The data is filtered to include only major earthquakes (primarily with magnitudes above 5.5) to focus on 
            events with significant impact. Through various visualizations and analyses, you can explore temporal trends, 
            geographic correlations, and recurrence intervals of powerful earthquakes across different tectonic regions.
            """)
            
            # Load historical data - dataset completamente independiente
            @st.cache_data(ttl=3600, show_spinner=False)
            def load_historical_data():
                try:
                    df_historic = pd.read_csv('data/2005-2025_mas_de_5_5_E_R.csv')
                    df_historic['time'] = pd.to_datetime(df_historic['time'])
                    return df_historic
                except Exception as e:
                    st.error(f"Error loading historical data: {e}")
                    return None
            
            # Procesar datos históricos sin afectar al dataframe principal
            @st.cache_data(ttl=3600, show_spinner=False)
            def process_historical_data(df_historic):
                if df_historic is None or df_historic.empty:
                    return None, None

                # Clonar los datos para evitar modificar el original
                data_historic = df_historic.copy()

                # Verificar y limpiar datos
                data_historic = data_historic.dropna(subset=['mag', 'latitude', 'longitude'])
                data_historic = data_historic.drop_duplicates(subset=['time', 'latitude', 'longitude', 'mag'])

                # Limitar a los 500 terremotos de mayor magnitud
                data_historic = data_historic.sort_values('mag', ascending=False).head(500).copy()

                # Añadir características basadas en el tiempo
                data_historic['year'] = data_historic['time'].dt.year
                data_historic['month'] = data_historic['time'].dt.month
                data_historic['day'] = data_historic['time'].dt.day
                data_historic['hour'] = data_historic['time'].dt.hour
                data_historic['weekday'] = data_historic['time'].dt.day_name()
                data_historic['decade'] = (data_historic['year'] // 10) * 10

                # Crear categorías de magnitud con colores que coincidan con su aplicación
                magnitude_ranges = [0, 2.5, 4.0, 5.5, 7.0, 10.0]
                magnitude_labels = ['Micro (<2.5)', 'Leve (2.5-4.0)', 'Moderado (4.0-5.5)', 'Fuerte (5.5-7.0)', 'Mayor (>7.0)']
                data_historic['magnitude_category'] = pd.cut(data_historic['mag'], bins=magnitude_ranges, labels=magnitude_labels)

                # Filtrar terremotos importantes
                major_quakes = data_historic[data_historic['mag'] >= 7.0].copy()

                return data_historic, major_quakes
            
            # Cargar y procesar datos
            with st.spinner("Loading historical earthquake data..."):
                df_historic = load_historical_data()
            
            if df_historic is not None:
                # Procesar datos sin afectar a filtered_df
                with st.spinner("Processing historical earthquake data..."):
                    data_historic, major_quakes = process_historical_data(df_historic)
                
                # A partir de aquí, trabajar exclusivamente con data_historic y major_quakes
                # sin hacer referencia a filtered_df en ningún momento
                
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
                data_historic['region'] = data_historic.apply(lambda x: categorize_region(x['latitude'], x['longitude']), axis=1)
                major_quakes['region'] = major_quakes.apply(lambda x: categorize_region(x['latitude'], x['longitude']), axis=1)
                
                # Add tectonic region classification
                data_historic['tectonic_region'] = data_historic.apply(lambda x: 
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
                data_historic['lat_rounded'] = round(data_historic['latitude'], 1)
                data_historic['lon_rounded'] = round(data_historic['longitude'], 1)

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
                        st.metric("Total Earthquakes (>5.5)",
                            value=f"{len(data_historic):,}",
                            delta=None
                        )
                    
                    with col2:
                        st.metric("Average Magnitude",
                            value=f"{data_historic['mag'].mean():.2f}",
                            delta=None
                        )
                    
                    with col3:
                        st.metric("Maximum Magnitude",
                            value=f"{data_historic['mag'].max():.1f}",
                            delta=None
                        )
                    
                    with col4:
                        most_active_year = data_historic['year'].value_counts().idxmax()
                        st.metric("Most Active Year",
                            value=f"{most_active_year}",
                            delta=f"{data_historic[data_historic['year']==most_active_year].shape[0]} events"
                        )
                    
                    # Magnitude distribution
                    st.subheader("Magnitude Distribution")
                    fig_mag_dist = px.histogram(
                        data_historic, 
                        x="mag", 
                        nbins=30, 
                        color="magnitude_category",
                        color_discrete_sequence=px.colors.sequential.Plasma
                    )
                    fig_mag_dist.add_vline(x=7.0, line_dash="dash", line_color="red", annotation_text="Major (7+)")
                    st.plotly_chart(fig_mag_dist, use_container_width=True)
                    
                    # Tectonic regions
                    st.subheader("Distribution by Tectonic Region")
                    region_counts = data_historic['tectonic_region'].value_counts().reset_index()
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
                    yearly_counts = data_historic.groupby('year').size().reset_index(name='count')
                    
                    fig_yearly = px.line(
                        yearly_counts, 
                        x='year', 
                        y='count',
                        markers=True,
                        line_shape='spline',
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
                        data_historic,
                        lat="latitude",
                        lon="longitude",
                        color="mag",
                        size="mag",
                        hover_name="place",
                        hover_data=["time", "mag", "depth"],
                        color_continuous_scale=px.colors.sequential.Plasma,
                        projection="natural earth",
                        size_max=20
                    )
                    
                    # Highlight the top 5 strongest earthquakes
                    top_earthquakes = data_historic.nlargest(5, 'mag')
                    
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
                    
                    # Update layout for dark theme and REMOVE LEGEND as requested
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
                            bgcolor="black",
                            projection_scale=1
                        ),
                        paper_bgcolor="black",
                        plot_bgcolor="black",
                        title_font=dict(size=20, color='white'),
                        showlegend=False  # Remove legend as requested
                    )
                    
                    st.plotly_chart(fig_map, use_container_width=True)
                    
                    # Heat map of activity
                    st.subheader("Earthquake Density Heatmap")
                    st.markdown("This visualization shows the concentration of seismic activity around the world")
                    
                    # Create a heat map using density_mapbox
                    fig_heat = px.density_mapbox(
                        data_historic,
                        lat="latitude",
                        lon="longitude",
                        z="mag",  # Weight points by magnitude
                        radius=10,
                        center=dict(lat=0, lon=0),
                        zoom=0.5,
                        mapbox_style="carto-darkmatter",
                        color_continuous_scale='Inferno'
                    )
                    
                    fig_heat.update_layout(height=500)
                    st.plotly_chart(fig_heat, use_container_width=True)
                    
                    # Cluster analysis
                    st.subheader("Regional Cluster Analysis")
                    
                    # Create a clustered view of earthquake hotspots
                    cluster_df = data_historic[['latitude', 'longitude']].copy()
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
                        data_historic['cluster'] = dbscan.fit_predict(cluster_data)
                    
                    n_clusters = len(set(data_historic['cluster'])) - (1 if -1 in data_historic['cluster'] else 0)
                    n_noise = list(data_historic['cluster']).count(-1)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Identified clusters", n_clusters)
                    col2.metric("Ungrouped events", n_noise)
                    
                    # Create cluster names based on regions
                    cluster_names = {}
                    for cluster_id in sorted(set(data_historic['cluster'])):
                        if cluster_id == -1:
                            cluster_names[cluster_id] = "Ungrouped"
                        else:
                        # Get the most common place in this cluster
                            cluster_df = data_historic[data_historic['cluster'] == cluster_id]
                            most_common_place = cluster_df['place'].str.split(', ').str[-1].mode().iloc[0]
                            cluster_names[cluster_id] = f"Cluster {cluster_id}: {most_common_place}"
                    
                    data_historic['cluster_name'] = data_historic['cluster'].map(cluster_names)
                    
                    # Plot the clusters
                    fig_cluster = px.scatter_geo(
                        data_historic,
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
                    monthly_counts = data_historic.groupby('month').size().reset_index(name='count')
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    monthly_counts['month_name'] = monthly_counts['month'].apply(lambda x: month_names[x-1])
                    
                    fig_monthly = px.bar(
                        monthly_counts,
                        x='month_name',
                        y='count',
                        color='count',
                        color_continuous_scale=px.colors.sequential.Plasma,
                        labels={'count': 'Number of Events', 'month_name': 'Month'}
                    )
                    st.plotly_chart(fig_monthly, use_container_width=True)
                    
                    # Hourly distribution
                    st.subheader("Distribution by Hour of Day")
                    hourly_counts = data_historic.groupby('hour').size().reset_index(name='count')
                    
                    fig_hourly = px.line(
                        hourly_counts,
                        x='hour',
                        y='count',
                        markers=True,
                        labels={'count': 'Number of Events', 'hour': 'Hour of Day (UTC)'}
                    )
                    
                    # Fill area under the line
                    fig_hourly.update_traces(fill='tozeroy', fillcolor='rgba(128, 0, 128, 0.2)')
                    st.plotly_chart(fig_hourly, use_container_width=True)
                    
                    # Relation between depth and magnitude over time
                    st.subheader("Depth vs Magnitude Over Time")
                    
                    # Create 3D scatter plot
                    fig_3d = px.scatter_3d(
                        data_historic.sort_values('time'),
                        x='time', 
                        y='depth', 
                        z='mag',
                        color='mag',
                        size='mag',
                        opacity=0.7,
                        color_continuous_scale=px.colors.sequential.Plasma,
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
                        data_historic,
                        x='mag',
                        y='depth',
                        color='tectonic_region',
                        size='mag',
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
                        delta=f"{(major_count/len(data_historic)*100):.1f}% of all events"
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
                
                # Tab 5: Recurrence Analysis - Modified to include multiple predictions
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
                        # Calculate recurrence statistics and generate multiple predictions
                        recurrence_data = []
                        all_predictions = []  # Store all predictions for display
                        
                        # Current time for reference
                        current_time = pd.Timestamp.now()
                        if current_time.tz is not None:
                            current_time = current_time.tz_localize(None)
                        
                        for region in regions_with_multiple:
                            region_df = major_quakes[major_quakes['region'] == region].sort_values('time')
                            
                            # Calculate intervals between events
                            if len(region_df) >= 2:
                                time_diffs = region_df['time'].diff().dropna()
                                avg_interval_days = time_diffs.dt.total_seconds().mean() / (24*3600)
                                avg_interval_years = avg_interval_days / 365.25
                                
                                # Calculate standard deviation for the interval
                                std_interval_days = time_diffs.dt.total_seconds().std() / (24*3600) if len(time_diffs) > 1 else avg_interval_days * 0.2
                                
                                # Last event and estimate of next
                                last_event = region_df['time'].max()
                                if last_event.tz is not None:
                                    last_event = last_event.tz_localize(None)
                                    
                                # Calculate time since last event
                                time_since_last = (current_time - last_event).total_seconds() / (24*3600*365.25)
                                
                                # Generate 10 predictions with varying intervals
                                predictions = []
                                for i in range(10):
                                    # Add some randomness to the interval for each prediction
                                    # Use normal distribution centered around the mean interval
                                    np.random.seed(i+42)  # Set seed for reproducibility but different for each i
                                    random_factor = np.random.normal(1.0, 0.3)  # Mean 1.0, std 0.3
                                    
                                    # Ensure reasonable factors (between 0.5 and 1.5 of the average)
                                    random_factor = max(0.5, min(1.5, random_factor))
                                    
                                    # Calculate prediction date
                                    prediction_interval = avg_interval_days * random_factor
                                    next_date = last_event + pd.Timedelta(days=prediction_interval)
                                    
                                    # Calculate magnitude - slight variations around the average magnitude
                                    avg_mag = region_df['mag'].mean()
                                    mag_std = region_df['mag'].std() if len(region_df) > 1 else 0.3
                                    predicted_mag = np.random.normal(avg_mag, mag_std * 0.5)
                                    predicted_mag = max(6.5, min(9.0, predicted_mag))  # Keep magnitude in reasonable range
                                    
                                    # Assign confidence level based on proximity to average interval
                                    if abs(random_factor - 1.0) < 0.1:
                                        confidence = "High"
                                    elif abs(random_factor - 1.0) < 0.25:
                                        confidence = "Medium"
                                    else:
                                        confidence = "Low"
                                        
                                    # Generate "time of day" for prediction (random but weighted toward typical patterns)
                                    # Using exactly summing probabilities
                                    hour_probs = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 
                                                         0.05, 0.05, 0.04, 0.04, 0.04, 0.04,
                                                         0.04, 0.04, 0.04, 0.04, 0.04, 0.05,
                                                         0.05, 0.05, 0.04, 0.04, 0.04, 0.04])
                                    # Normalize to ensure sum is exactly 1.0
                                    hour_probs = hour_probs / hour_probs.sum()
                                    hour = np.random.choice(range(24), p=hour_probs)
                                    minute = np.random.randint(0, 60)
                                    second = np.random.randint(0, 60)
                                    
                                    # Set time components
                                    next_date = next_date.replace(hour=hour, minute=minute, second=second)
                                    
                                    # Store prediction
                                    predictions.append({
                                        "region": region,
                                        "predicted_date": next_date,
                                        "predicted_magnitude": predicted_mag,
                                        "confidence": confidence,
                                        "days_from_now": (next_date - current_time).days,
                                        "interval_used": prediction_interval / 365.25,  # in years
                                        "time_since_last": time_since_last
                                    })
                                    
                                # Add all predictions to the master list
                                all_predictions.extend(predictions)
                                
                                # Calculate probability using Poisson distribution
                                lambda_param = 1 / avg_interval_years
                                prob_1yr = 1 - np.exp(-lambda_param * 1)
                                prob_5yr = 1 - np.exp(-lambda_param * 5)
                                
                                # Store main recurrence statistics
                                recurrence_data.append({
                                    'Region': region,
                                    'Events': len(region_df),
                                    'Avg Interval (years)': avg_interval_years,
                                    'Last Event': last_event,
                                    'Time Since Last (years)': time_since_last,
                                    'Next Estimate': last_event + pd.Timedelta(days=avg_interval_days),
                                    'Probability (1 year)': prob_1yr,
                                    'Probability (5 years)': prob_5yr
                                })
                        
                        # Convert to DataFrames
                        recurrence_df = pd.DataFrame(recurrence_data)
                        predictions_df = pd.DataFrame(all_predictions)
                        
                        # Format predictions for display
                        display_predictions = predictions_df.copy()
                        display_predictions['predicted_date'] = display_predictions['predicted_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        display_predictions['predicted_magnitude'] = display_predictions['predicted_magnitude'].round(2)
                        display_predictions['interval_used'] = display_predictions['interval_used'].round(2)
                        display_predictions['time_since_last'] = display_predictions['time_since_last'].round(2)
                        
                        # Display the predictions table
                        st.subheader("Future Earthquake Predictions (Next 10 Major Events)")
                        st.dataframe(
                            display_predictions[['region', 'predicted_date', 'predicted_magnitude', 'confidence', 'days_from_now']].rename(
                                columns={
                                    'region': 'Region',
                                    'predicted_date': 'Predicted Date',
                                    'predicted_magnitude': 'Magnitude',
                                    'confidence': 'Confidence',
                                    'days_from_now': 'Days From Now'
                                }
                            ),
                            use_container_width=True
                        )
                        
                        # Show a plot of predictions by region over time
                        st.subheader("Timeline of Predicted Major Earthquakes")
                        
                        # Sort predictions by date
                        sorted_predictions = predictions_df.sort_values('predicted_date')
                        
                        # Create a timeline of predictions
                        fig_pred_timeline = px.scatter(
                            sorted_predictions,
                            x='predicted_date',
                            y='predicted_magnitude',
                            color='region',
                            size='predicted_magnitude',
                            symbol='confidence',
                            symbol_map={'High': 'circle', 'Medium': 'square', 'Low': 'diamond'},
                            hover_name='region',
                            hover_data={
                                'predicted_date': True,
                                'predicted_magnitude': ':.2f',
                                'confidence': True,
                                'days_from_now': True
                            },
                            labels={
                                'predicted_date': 'Predicted Date',
                                'predicted_magnitude': 'Predicted Magnitude',
                                'region': 'Region',
                                'confidence': 'Confidence'
                            },
                            title="Timeline of Predicted Major Earthquakes"
                        )
                        
                        fig_pred_timeline.update_layout(height=600)
                        st.plotly_chart(fig_pred_timeline, use_container_width=True)
                        
                        # Create a bar chart for probability (5 years)
                        fig_prob = px.bar(
                            recurrence_df,
                            x='Region',
                            y='Probability (5 years)',
                            color='Probability (5 years)',
                            color_continuous_scale='Inferno',
                            labels={'Probability (5 years)': 'Probability (5 years)', 'Region': 'Region'},
                            text=recurrence_df['Probability (5 years)'].apply(lambda x: f"{x*100:.1f}%")
                        )
                        fig_prob.update_traces(textposition='outside')
                        st.plotly_chart(fig_prob, use_container_width=True)
                        
                        # Create an interactive forecast map showing all predictions
                        st.subheader("Interactive Earthquake Forecast Map")
                        
                        m = folium.Map(
                            location=[20, 0],
                            zoom_start=2,
                            tiles='CartoDB dark_matter'
                        )
                        
                        # Get coordinates for each region
                        region_coords = {}
                        for region in regions_with_multiple:
                            region_data = data_historic[data_historic['region'] == region]
                            region_coords[region] = {
                                'lat': region_data['latitude'].mean(),
                                'lon': region_data['longitude'].mean()
                            }
                        
                        # Add markers for each prediction
                        for i, pred in predictions_df.iterrows():
                            region = pred['region']
                            coords = region_coords.get(region, {'lat': 0, 'lon': 0})
                            
                            # Add small random offset to prevent complete overlap
                            lat_offset = np.random.uniform(-1, 1)
                            lon_offset = np.random.uniform(-1, 1)
                            
                            lat = coords['lat'] + lat_offset
                            lon = coords['lon'] + lon_offset
                            
                            # Determine color based on confidence
                            if pred['confidence'] == 'High':
                                color = 'red'
                                radius = 250000
                            elif pred['confidence'] == 'Medium':
                                color = 'orange'
                                radius = 200000
                            else:
                                color = 'yellow'
                                radius = 150000
                            
                            # Format popup content
                            popup_content = f"""
                            <div style='width:220px'>
                                <h4>{region}</h4>
                                <hr>
                                <b>Predicted Date:</b> {pred['predicted_date'].strftime('%Y-%m-%d %H:%M:%S')}<br>
                                <b>Magnitude:</b> {pred['predicted_magnitude']:.2f}<br>
                                <b>Confidence:</b> {pred['confidence']}<br>
                                <b>Days from now:</b> {pred['days_from_now']}<br>
                                <hr>
                                <b>Based on interval:</b> {pred['interval_used']:.2f} years<br>
                                <b>Time since last:</b> {pred['time_since_last']:.2f} years<br>
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
                        <p><b>Earthquake Prediction Confidence</b></p>
                        <p><i class="fa fa-circle" style="color:red"></i> High Confidence</p>
                        <p><i class="fa fa-circle" style="color:orange"></i> Medium Confidence</p>
                        <p><i class="fa fa-circle" style="color:yellow"></i> Low Confidence</p>
                        </div>
                        """
                        
                        m.get_root().html.add_child(folium.Element(legend_html))
                        
                        # Display the map
                        folium_static(m, width=1000, height=600)
                        
                        # Display detailed prediction table
                        st.subheader("Detailed Prediction Data")
                        
                        # Show all prediction details in an expanded view
                        detailed_pred = display_predictions[['region', 'predicted_date', 'predicted_magnitude', 
                                                        'confidence', 'days_from_now', 'interval_used', 
                                                        'time_since_last']].rename(
                            columns={
                                'region': 'Region',
                                'predicted_date': 'Predicted Date & Time',
                                'predicted_magnitude': 'Magnitude',
                                'confidence': 'Confidence',
                                'days_from_now': 'Days From Now',
                                'interval_used': 'Interval Used (years)',
                                'time_since_last': 'Time Since Last (years)'
                            }
                        )
                        
                        st.dataframe(detailed_pred, use_container_width=True)
                        
                        # Disclaimer
                        st.warning("""
                        ⚠️ **Disclaimer:** These predictions are based solely on statistical analysis of historical data.
                        Earthquake forecasting has inherent limitations and uncertainties. This information should be used
                        for educational purposes only and not for making critical safety decisions.
                        """)
                
            else:
                st.error("Failed to load historical earthquake data. Please check the file path and format.")


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