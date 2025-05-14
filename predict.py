import pandas as pd

# Cargar los datos de terremotos históricos con magnitud mayor a 5.5
import plotly.express as px

# Usar la API del USGS para cargar terremotos históricos significativos
url_historic = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2005-01-01&endtime=2025-05-08&minmagnitude=5.5"
historic_eq = pd.read_csv(url_historic)

# Convertir la columna de fecha a formato datetime
historic_eq['time'] = pd.to_datetime(historic_eq['time'])

# Filtrar terremotos con magnitud mayor a 5.5
historic_eq = historic_eq[historic_eq['mag'] >= 5.5]

# Procesar y preparar los datos para análisis
data = historic_eq.copy()

# Verificar valores nulos y limpiar datos
print(f"Valores faltantes en el conjunto de datos: {data['mag'].isna().sum()}")
data = data.dropna(subset=['mag', 'latitude', 'longitude'])

# Verificar entradas duplicadas
duplicados = data.duplicated(subset=['time', 'latitude', 'longitude', 'mag'])
print(f"Entradas duplicadas: {duplicados.sum()}")
data = data.drop_duplicates(subset=['time', 'latitude', 'longitude', 'mag'])

# Filtrar terremotos mayores (magnitud >= 7.0)
terremotos_grandes = data[data['mag'] >= 7.0].copy()
print(f"Número de terremotos mayores (magnitud >= 7.0): {len(terremotos_grandes)}")

# Añadir características basadas en tiempo
data['time'] = pd.to_datetime(data['time'])
data['year'] = data['time'].dt.year
data['month'] = data['time'].dt.month
data['day'] = data['time'].dt.day
data['hour'] = data['time'].dt.hour
data['weekday'] = data['time'].dt.day_name()
data['decade'] = (data['year'] // 10) * 10

# Crear categorías de magnitud
rangos_magnitud = [0, 2.5, 4.0, 5.5, 7.0, 10.0]
etiquetas_magnitud = ['Micro (<2.5)', 'Leve (2.5-4.0)', 'Moderado (4.0-5.5)', 'Fuerte (5.5-7.0)', 'Mayor (>7.0)']
data['magnitude_category'] = pd.cut(data['mag'], bins=rangos_magnitud, labels=etiquetas_magnitud)

# Aplicar estas características también a terremotos_grandes
terremotos_grandes['decade'] = (terremotos_grandes['time'].dt.year // 10) * 10

# Definir función de categorización regional
def categorizar_region(lat, lon):
    # Anillo de Fuego del Pacífico
    if ((lon > 120 or lon < -120) and (lat > -60 and lat < 60)):
        return "Pacífico Occidental" if lon > 120 else "Pacífico Oriental"
    # Mediterráneo-Cáucaso
    elif ((lat > 30 and lat < 45) and (lon > -10 and lon < 50)):
        return "Mediterráneo-Cáucaso"
    # Cinturón Indonesia-Himalaya
    elif ((lat > -10 and lat < 45) and (lon > 70 and lon < 120)):
        return "Indonesia-Himalaya"
    # Otras regiones
    else:
        return "Otras Regiones"

# Aplicar categorización regional
data['region'] = data.apply(lambda x: categorizar_region(x['latitude'], x['longitude']), axis=1)
terremotos_grandes['region'] = terremotos_grandes.apply(lambda x: categorizar_region(x['latitude'], x['longitude']), axis=1)

# Añadir clasificación de región tectónica
data['tectonic_region'] = data.apply(lambda x: 
    'Anillo de Fuego' if ((x['longitude'] > 120 and x['longitude'] < 180) or 
                       (x['longitude'] < -120 and x['longitude'] > -180)) and 
                      (x['latitude'] > -60 and x['latitude'] < 60) else
    'Cinturón Alpino-Himalayo' if ((x['latitude'] > 30 and x['latitude'] < 45) and 
                                (x['longitude'] > 0 and x['longitude'] < 150)) or 
                               ((x['latitude'] > 0 and x['latitude'] < 30) and 
                                (x['longitude'] > 60 and x['longitude'] < 120)) else
    'Dorsal Medio-Atlántica' if (x['longitude'] > -45 and x['longitude'] < 0) and 
                            (x['latitude'] > -60 and x['latitude'] < 80) else
    'Otras Regiones Tectónicas', axis=1)

# Calcular tiempo entre terremotos mayores en la misma región
terremotos_grandes = terremotos_grandes.sort_values(by=['region', 'time'])
terremotos_grandes['years_since_last'] = terremotos_grandes.groupby('region')['time'].diff().dt.total_seconds() / (365.25 * 24 * 60 * 60)

# Añadir coordenadas redondeadas para análisis de puntos calientes
data['lat_rounded'] = round(data['latitude'], 1)
data['lon_rounded'] = round(data['longitude'], 1)

# Calcular estadísticas básicas
total_terremotos = len(data)
magnitud_promedio = data['mag'].mean()
magnitud_maxima = data['mag'].max()
terremotos_por_año = data['year'].value_counts().sort_index()
terremotos_por_region = data['tectonic_region'].value_counts()
terremotos_por_magnitud = data['magnitude_category'].value_counts().sort_index()

# Mostrar estadísticas básicas
print(f"Total terremotos significativos (>5.5): {total_terremotos}")
print(f"Magnitud promedio: {magnitud_promedio:.2f}")
print(f"Magnitud máxima: {magnitud_maxima:.1f}")
print(f"Año con más terremotos significativos: {terremotos_por_año.idxmax()} ({terremotos_por_año.max()} eventos)")

# Crear dataframe de estadísticas regionales
estadisticas_region = data.groupby('tectonic_region').agg({
    'mag': ['count', 'mean', 'max'],
    'depth': ['mean', 'min', 'max']
}).round(2)

# Preparar datos para visualización
datos_para_viz = data.sort_values('mag', ascending=False)

# Crear un mapa interactivo de terremotos históricos significativos
fig = px.scatter_geo(historic_eq,
                   lat='latitude',
                   lon='longitude',
                   color='mag',
                   size='mag',
                   hover_name='place',
                   hover_data=['time', 'depth'],
                   title='Terremotos Significativos (Magnitud > 5.5) entre 2005-2025',
                   color_continuous_scale=px.colors.sequential.Plasma,
                   projection='natural earth',
                   size_max=20)

# Personalizar el mapa con mayor altura y tema oscuro
fig.update_layout(
    height=1200,
    coloraxis_colorbar=dict(
        title='Magnitud',  # Título del colorbar
        title_side='right',  # Asegura que el título esté al lado del colorbar
        title_font=dict(size=14),  # Ajusta el tamaño de la fuente del título
        tickfont=dict(size=12),  # Ajusta el tamaño de la fuente de las etiquetas
        len=0.8,  # Ajusta la longitud del colorbar
        y=0.5,  # Centra el colorbar verticalmente
        yanchor='middle'  # Asegura el anclaje vertical
    ),
    legend_title_text='Magnitud',
    geo=dict(
        showland=True,
        landcolor='rgb(50, 50, 50)',
        showcountries=True,
        countrycolor='rgb(120, 120, 120)',
        showocean=True,
        oceancolor='rgb(30, 30, 60)',
        showcoastlines=True,
        coastlinecolor='rgb(170, 170, 170)',
        projection_type='orthographic',
        bgcolor='rgb(10, 10, 20)'
    ),
    margin=dict(l=20, r=20, t=50, b=20),
    template="plotly_dark"
)

# Identificar los 5 terremotos más fuertes
top_earthquakes = historic_eq.nlargest(5, 'mag')

# Añadir marcadores de bandera para los 5 terremotos más fuertes
for _, row in top_earthquakes.iterrows():
    fig.add_trace(go.Scattergeo(
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
        name=f"M{row['mag']:.1f} - {row['place'][:20]}...",
        hovertext=f"Magnitud: {row['mag']}<br>Fecha: {row['time']}<br>Lugar: {row['place']}",
        hoverinfo="text"
    ))

# Mostrar el mapa
fig.show()

# Análisis estadístico de terremotos históricos
total_terremotos = len(historic_eq)
magnitud_promedio = historic_eq['mag'].mean()
magnitud_maxima = historic_eq['mag'].max()
año_con_más_terremotos = historic_eq['time'].dt.year.value_counts().idxmax()

print(f"Total de terremotos significativos (>5.5): {total_terremotos}")
print(f"Magnitud promedio: {magnitud_promedio:.2f}")
print(f"Magnitud máxima registrada: {magnitud_maxima}")
print(f"Año con más terremotos significativos: {año_con_más_terremotos}")

# Extraer componentes de fecha para análisis
data['year'] = data['time'].dt.year
data['month'] = data['time'].dt.month
data['day'] = data['time'].dt.day
data['hour'] = data['time'].dt.hour

# Tendencia temporal de terremotos por año
plt.figure(figsize=(12, 6))
yearly_counts = data['year'].value_counts().sort_index()
sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker='o', color='blue')
plt.title('Tendencia de Terremotos por Año (2005-2025)', fontsize=16)
plt.xlabel('Año', fontsize=12)
plt.ylabel('Número de Terremotos', fontsize=12)
plt.grid(alpha=0.3)
plt.show()


# Distribución de magnitudes
plt.figure(figsize=(12, 6))
sns.histplot(data['mag'], bins=30, kde=True, color='orange')
plt.title('Distribución de Magnitudes de Terremotos', fontsize=16)
plt.xlabel('Magnitud', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.axvline(x=5.5, color='red', linestyle='--', label='Umbral de análisis (5.5)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Crear un gráfico de dispersión interactivo con Plotly Express
import plotly.express as px

# Crear una escala de color basada en la profundidad
fig = px.scatter(data, 
                x='mag', 
                y='depth',
                color='depth',
                size='mag',
                hover_name='place',
                hover_data=['time', 'mag', 'depth'],
                color_continuous_scale='Viridis',
                title='Relación entre Magnitud y Profundidad de Terremotos',
                labels={'mag': 'Magnitud', 'depth': 'Profundidad (km)'},
                height=700)

# Personalizar el diseño
fig.update_layout(
    template='plotly_dark',
    xaxis_title='Magnitud',
    yaxis_title='Profundidad (km)',
    coloraxis_colorbar=dict(title='Profundidad (km)'),
    hovermode='closest',
    legend_title_text='Profundidad',
    font=dict(family="Arial", size=12),
)

# Agregar anotaciones para diferentes categorías de terremotos
fig.add_annotation(
    x=7.0, y=10,
    text="Terremotos mayores",
    showarrow=True,
    arrowhead=1
)

fig.add_annotation(
    x=6.0, y=100,
    text="Terremotos profundos",
    showarrow=True,
    arrowhead=1
)

# Invertir el eje y para mostrar la profundidad aumentando hacia abajo
fig.update_yaxes(autorange="reversed")

# Mostrar el gráfico
fig.show()

# Distribución de terremotos por hora del día
plt.figure(figsize=(12, 6))
hourly_counts = data['hour'].value_counts().sort_index()
sns.barplot(x=hourly_counts.index, y=hourly_counts.values, color='purple')
plt.title('Distribución de Terremotos por Hora del Día', fontsize=16)
plt.xlabel('Hora', fontsize=12)
plt.ylabel('Número de Terremotos', fontsize=12)
plt.ylim(300, 500)  # Limitar la escala del eje y entre 300 y 500
plt.grid(alpha=0.3)

# Añadir anotación para indicar la escala limitada
plt.annotate(
    'Escala limitada entre 300 y 500\npara apreciar mejor las diferencias',
    xy=(12, 450), xytext=(15, 480),
    arrowprops=dict(facecolor='white', arrowstyle='->'),
    fontsize=10, color='white'
)

plt.show()

import seaborn as sns
import numpy as np
import pandas as pd
from datetime import timedelta

import matplotlib.pyplot as plt
import plotly.express as px

# Filtrar terremotos mayores a 7 en los últimos 20 años
mayores7 = terremotos_grandes[terremotos_grandes['mag'] >= 7.0].copy()

# Estadística por región
region_counts = mayores7['region'].value_counts()
region_top = region_counts.idxmax()
region_top_count = region_counts.max()

# Calcular intervalos de recurrencia por región
mayores7 = mayores7.sort_values(['region', 'time'])
mayores7['years_since_last'] = mayores7.groupby('region')['time'].diff().dt.total_seconds() / (365.25 * 24 * 60 * 60)

recurrencia = mayores7.groupby('region')['years_since_last'].mean().dropna()

# Predicción simple: próxima fecha estimada por región (última fecha + media de recurrencia)
predicciones = []
for region, group in mayores7.groupby('region'):
    if group['years_since_last'].notnull().any():
        media = group['years_since_last'].mean()
        ultima = group['time'].max()
        proxima = ultima + pd.Timedelta(days=media*365.25)
        predicciones.append({'region': region, 'media_años': media, 'ultima': ultima, 'proxima_estim': proxima})
predicciones_df = pd.DataFrame(predicciones)

# Gráfico de barras: frecuencia de terremotos >7 por región
plt.figure(figsize=(10,6))
sns.barplot(x=region_counts.index, y=region_counts.values, palette='Reds')
plt.title('Frecuencia de Terremotos >7 por Región')
plt.ylabel('Cantidad')
plt.xlabel('Región')
plt.tight_layout()
plt.show()

# Gráfico de intervalos de recurrencia
plt.figure(figsize=(10,6))
recurrencia.sort_values().plot(kind='bar', color='orange')
plt.title('Intervalo Promedio entre Terremotos >7 por Región (años)')
plt.ylabel('Años')
plt.xlabel('Región')
plt.tight_layout()
plt.show()

# Mapa de los terremotos >7 históricos
fig = px.scatter_geo(
    mayores7,
    lat='latitude',
    lon='longitude',
    color='region',
    size='mag',
    hover_name='place',
    hover_data=['time', 'mag', 'depth'],
    title='Terremotos Históricos >7 (2005-2025)',
    projection='natural earth',
    size_max=20
)


# Asignar x a hue y legend=False
plt.figure(figsize=(10,6))
sns.barplot(x=region_counts.index, y=region_counts.values, hue=region_counts.index, palette='Reds', legend=False)
plt.title('Frecuencia de Terremotos >7 por Región')
plt.ylabel('Cantidad')
plt.xlabel('Región')
plt.tight_layout()
plt.show()
fig.update_layout(height=600, template='plotly_dark')
fig.show()

# Tabla de predicciones
print("Predicción de próxima ocurrencia de terremoto >7 por región:")
display(predicciones_df[['region', 'media_años', 'ultima', 'proxima_estim']])

# Probabilidad simple: Poisson (λ = eventos/año por región)
años = 20
probs = []
for region, count in region_counts.items():
    lam = count / años
    # Probabilidad de al menos 1 evento en el próximo año
    p = 1 - np.exp(-lam)
    probs.append({'region': region, 'prob_1año': p, 'eventos/año': lam})
probs_df = pd.DataFrame(probs)
print("Probabilidad de al menos un terremoto >7 en el próximo año por región:")
display(probs_df)

import folium
from folium.plugins import MarkerCluster
from datetime import datetime

# Crear un mapa centrado en el mundo con tema oscuro
mapa_probabilidades = folium.Map(location=[0, 0], zoom_start=2, tiles='CartoDB dark_matter')

# Título para el mapa
title_html = '''
<div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); 
            z-index: 1000; background-color: rgba(0,0,0,0.7); color: white; 
            padding: 10px; border-radius: 5px; font-size: 16px; text-align: center;">
    <h3>Pronóstico de Terremotos Mayores (>7.0)</h3>
    <p>Predicción basada en análisis histórico de recurrencia</p>
</div>
'''
mapa_probabilidades.get_root().html.add_child(folium.Element(title_html))

# Crear un marcador para cada región con información de probabilidad y fecha estimada
for _, row in predicciones_df.iterrows():
    region = row['region']
    proxima_fecha = row['proxima_estim']
    ultima_fecha = row['ultima']
    media_años = row['media_años']
    
    # Encontrar la probabilidad correspondiente para esta región
    probabilidad = probs_df.loc[probs_df['region'] == region, 'prob_1año'].values[0] * 100  # Convertir a porcentaje
    eventos_año = probs_df.loc[probs_df['region'] == region, 'eventos/año'].values[0]

    # Coordenadas aproximadas para cada región
    coordenadas = {
        'Indonesia-Himalaya': [-2.5, 100],
        'Mediterráneo-Cáucaso': [40, 30],
        'Otras Regiones': [0, 0],
        'Pacífico Occidental': [15, 140],
        'Pacífico Oriental': [-10, -100]
    }

    # Obtener las coordenadas de la región
    lat, lon = coordenadas.get(region, [0, 0])

    # Determinar el color y tamaño del marcador según la probabilidad
    if probabilidad > 80:
        color = 'red'
        radio = 20
    elif probabilidad > 50:
        color = 'orange'
        radio = 15
    elif probabilidad > 20:
        color = 'yellow'
        radio = 12
    else:
        color = 'green'
        radio = 10
    
    # Formatear las fechas para mejor visualización
    proxima_fecha_str = proxima_fecha.strftime('%d/%m/%Y') if pd.notnull(proxima_fecha) else 'N/A'
    ultima_fecha_str = ultima_fecha.strftime('%d/%m/%Y') if pd.notnull(ultima_fecha) else 'N/A'
    
    # Calcular días restantes hasta la próxima fecha estimada
    # Calculate days remaining until the next estimated date
    if pd.notnull(proxima_fecha):
        # Make datetime.now() timezone-aware to match proxima_fecha
        now = datetime.now().replace(tzinfo=proxima_fecha.tzinfo)
        dias_restantes = (proxima_fecha - now).days
    else:
        dias_restantes = 'N/A'
    
    # Añadir un marcador al mapa
    folium.CircleMarker(
        location=[lat, lon],
        radius=radio,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(
            f"<div style='width: 300px'>"
            f"<h4 style='text-align: center; margin-bottom: 10px;'>{region}</h4>"
            f"<hr style='margin: 5px 0;'>"
            f"<b>Probabilidad (1 año):</b> {probabilidad:.1f}%<br>"
            f"<b>Eventos esperados/año:</b> {eventos_año:.2f}<br>"
            f"<b>Intervalo promedio:</b> {media_años:.2f} años<br>"
            f"<hr style='margin: 5px 0;'>"
            f"<b>Último terremoto >7:</b> {ultima_fecha_str}<br>"
            f"<b>Próxima fecha estimada:</b> {proxima_fecha_str}<br>"
            f"<b>Días restantes:</b> {dias_restantes if isinstance(dias_restantes, int) else 'N/A'} días"
            f"</div>",
            max_width=350
        )
    ).add_to(mapa_probabilidades)
    
    # Añadir etiquetas directamente en el mapa para fecha estimada
    folium.map.Marker(
        [lat+5, lon], 
        icon=folium.DivIcon(
            icon_size=(150,36),
            icon_anchor=(75,0),
            html=f'<div style="font-size: 10pt; color: white; background-color: rgba(0,0,0,0.5); padding: 5px; border-radius: 5px;">{proxima_fecha_str}</div>'
        )
    ).add_to(mapa_probabilidades)

# Añadir una leyenda personalizada al mapa
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: rgba(0,0,0,0.7); 
color: white; padding: 10px; border: 1px solid gray; border-radius: 5px;">
<h4>Probabilidad de Terremoto >7.0</h4>
<p><i style="background: red; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></i> Alta (>80%)</p>
<p><i style="background: orange; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></i> Media (50-80%)</p>
<p><i style="background: yellow; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></i> Moderada (20-50%)</p>
<p><i style="background: green; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></i> Baja (<20%)</p>
</div>
'''
mapa_probabilidades.get_root().html.add_child(folium.Element(legend_html))

# Mostrar el mapa
mapa_probabilidades