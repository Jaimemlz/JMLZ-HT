import streamlit as st
import plotly.express as px
import plotly.io as pio
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import numpy as np

# Configurar tema claro por defecto para Plotly
pio.templates.default = "plotly_white"

# Funciones para calcular ranking de riesgo-beneficio
def es_ea_valida(nombre_ea):
    """
    Verifica si un nombre de EA es válido (contiene letras).
    Descarta strings vacíos, solo números, o valores como 0.70, -0.02, etc.
    """
    if not nombre_ea or nombre_ea.strip() == "":
        return False
    
    # Verificar si contiene al menos una letra
    return any(c.isalpha() for c in nombre_ea)

def calcular_ratio_riesgo_beneficio(df_ea):
    """
    Calcula el ratio riesgo-beneficio para una EA específica.
    Ratio = (Beneficio promedio de operaciones ganadoras) / (Pérdida promedio de operaciones perdedoras)
    """
    operaciones_ganadoras = df_ea[df_ea['Beneficio'] > 0]
    operaciones_perdedoras = df_ea[df_ea['Beneficio'] < 0]
    
    if len(operaciones_ganadoras) == 0:
        return 0  # Si no hay operaciones ganadoras, ratio es 0
    
    if len(operaciones_perdedoras) == 0:
        return float('inf')  # Si no hay operaciones perdedoras, ratio es infinito
    
    beneficio_promedio = operaciones_ganadoras['Beneficio'].mean()
    perdida_promedio = abs(operaciones_perdedoras['Beneficio'].mean())
    
    if perdida_promedio == 0:
        return float('inf')  # Evitar división por cero
    
    return beneficio_promedio / perdida_promedio

def calcular_estadisticas_ea(df_ea):
    """
    Calcula estadísticas detalladas para una EA específica.
    """
    total_ops = len(df_ea)
    ops_ganadoras = len(df_ea[df_ea['Beneficio'] > 0])
    ops_perdedoras = len(df_ea[df_ea['Beneficio'] < 0])
    
    win_rate = (ops_ganadoras / total_ops * 100) if total_ops > 0 else 0
    
    if ops_ganadoras > 0:
        beneficio_promedio = df_ea[df_ea['Beneficio'] > 0]['Beneficio'].mean()
    else:
        beneficio_promedio = 0
    
    if ops_perdedoras > 0:
        perdida_promedio = abs(df_ea[df_ea['Beneficio'] < 0]['Beneficio'].mean())
    else:
        perdida_promedio = 0
    
    ratio_riesgo_beneficio = calcular_ratio_riesgo_beneficio(df_ea)
    
    beneficio_total = df_ea['Beneficio'].sum()
    
    return {
        'EA': df_ea['EA'].iloc[0],
        'Símbolo': df_ea['Símbolo'].iloc[0],
        'Total_Ops': total_ops,
        'Ops_Ganadoras': ops_ganadoras,
        'Ops_Perdedoras': ops_perdedoras,
        'Win_Rate': win_rate,
        'Beneficio_Promedio': beneficio_promedio,
        'Perdida_Promedio': perdida_promedio,
        'Ratio_Riesgo_Beneficio': ratio_riesgo_beneficio,
        'Beneficio_Total': beneficio_total
    }

def crear_ranking_ea(df):
    """
    Crea un ranking de todas las EAs basado en un score de rentabilidad que combina
    ratio riesgo-beneficio y win rate.
    """
    if df.empty:
        return pd.DataFrame()
    
    # Agrupar por EA y Símbolo
    grupos = df.groupby(['EA', 'Símbolo'])
    
    ranking_data = []
    for (ea, simbolo), grupo in grupos:
        stats = calcular_estadisticas_ea(grupo)
        ranking_data.append(stats)
    
    ranking_df = pd.DataFrame(ranking_data)
    
    # Calcular score de rentabilidad que combina ratio R/B y win rate
    def calcular_score_rentabilidad(row):
        win_rate = row['Win_Rate'] / 100  # Convertir a decimal
        ratio_rb = row['Ratio_Riesgo_Beneficio']
        
        # Si no hay operaciones ganadoras, score es 0
        if win_rate == 0:
            return 0
        
        # Si no hay operaciones perdedoras (ratio infinito), usar un valor alto
        if ratio_rb == float('inf'):
            ratio_rb = 100  # Valor alto pero finito
        
        # Score = Win Rate * Ratio Riesgo-Beneficio
        # Esto premia tanto la frecuencia de ganancias como la eficiencia
        score = win_rate * ratio_rb
        return score
    
    ranking_df['Score_Rentabilidad'] = ranking_df.apply(calcular_score_rentabilidad, axis=1)
    
    # Ordenar por score de rentabilidad (descendente)
    ranking_df = ranking_df.sort_values('Score_Rentabilidad', ascending=False)
    
    # Agregar posición en el ranking
    ranking_df['Posicion'] = range(1, len(ranking_df) + 1)
    
    # Formatear columnas para presentación
    ranking_df['Win_Rate_Formateado'] = ranking_df['Win_Rate'].apply(lambda x: f"{x:.1f}%")
    ranking_df['Beneficio_Promedio_Formateado'] = ranking_df['Beneficio_Promedio'].apply(lambda x: f"${x:.2f}")
    ranking_df['Perdida_Promedio_Formateado'] = ranking_df['Perdida_Promedio'].apply(lambda x: f"${x:.2f}")
    ranking_df['Ratio_Formateado'] = ranking_df['Ratio_Riesgo_Beneficio'].apply(
        lambda x: f"{x:.2f}" if x != float('inf') else "∞"
    )
    ranking_df['Beneficio_Total_Formateado'] = ranking_df['Beneficio_Total'].apply(lambda x: f"${x:.2f}")
    ranking_df['Score_Formateado'] = ranking_df['Score_Rentabilidad'].apply(lambda x: f"{x:.2f}")
    
    return ranking_df

# Configurar tema claro para Streamlit
import streamlit.components.v1 as components

# Configuración de la página
st.set_page_config(
    page_title="Análisis de EA", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Estilos para las tarjetas y componentes
st.markdown("""
<style>
    header[data-testid="stHeader"] {
        display: none !important;
    }

    header[data-testid="stHeader"] {display: none !important;}

    /* Ocultar el div del borde de los tabs */
    div[data-baseweb="tab-border"] {
        display: none !important;
    }
    
    /* Corregir margin-bottom negativo en párrafos dentro de divs personalizados */
    .st-emotion-cache-r44huj {
        margin-bottom: 0 !important;
    }
    
    /* Configuración específica para tablas con tema claro - MÁS ESPECÍFICO */
    [data-testid="stDataFrame"] {
        background-color: white !important;
        color: #495057 !important;
    }
    
    [data-testid="stDataFrame"] table {
        background-color: white !important;
        color: #495057 !important;
    }
    
    [data-testid="stDataFrame"] thead th {
        background-color: #f8f9fa !important;
        color: #495057 !important;
        border: 1px solid #e9ecef !important;
    }
    
    [data-testid="stDataFrame"] tbody td {
        background-color: white !important;
        color: #495057 !important;
        border: 1px solid #e9ecef !important;
    }
    
    [data-testid="stDataFrame"] tbody tr:nth-child(even) {
        background-color: #f8f9fa !important;
    }
    
    [data-testid="stDataFrame"] tbody tr:nth-child(even) td {
        background-color: #f8f9fa !important;
    }
    
    /* CSS MÁS ESPECÍFICO PARA FORZAR TEMA CLARO */
    div[data-testid="stDataFrame"] {
        background-color: white !important;
        color: #495057 !important;
    }
    
    div[data-testid="stDataFrame"] table {
        background-color: white !important;
        color: #495057 !important;
    }
    
    div[data-testid="stDataFrame"] thead th {
        background-color: #f8f9fa !important;
        color: #495057 !important;
        border: 1px solid #e9ecef !important;
    }
    
    div[data-testid="stDataFrame"] tbody td {
        background-color: white !important;
        color: #495057 !important;
        border: 1px solid #e9ecef !important;
    }
    
    div[data-testid="stDataFrame"] tbody tr {
        background-color: white !important;
    }
    
    div[data-testid="stDataFrame"] tbody tr:nth-child(even) {
        background-color: #f8f9fa !important;
    }
    
    div[data-testid="stDataFrame"] tbody tr:nth-child(even) td {
        background-color: #f8f9fa !important;
    }
    
    /* Forzar todos los elementos de tabla */
    div[data-testid="stDataFrame"] * {
        color: #495057 !important;
    }
    
    /* CSS ULTRA ESPECÍFICO PARA FORZAR TEMA CLARO */
    .stDataFrame div[data-testid="stDataFrame"] {
        background-color: white !important;
        color: #495057 !important;
    }
    
    .stDataFrame div[data-testid="stDataFrame"] table {
        background-color: white !important;
        color: #495057 !important;
    }
    
    .stDataFrame div[data-testid="stDataFrame"] thead th {
        background-color: #f8f9fa !important;
        color: #495057 !important;
        border: 1px solid #e9ecef !important;
    }
    
    .stDataFrame div[data-testid="stDataFrame"] tbody td {
        background-color: white !important;
        color: #495057 !important;
        border: 1px solid #e9ecef !important;
    }
    
    .stDataFrame div[data-testid="stDataFrame"] tbody tr {
        background-color: white !important;
    }
    
    .stDataFrame div[data-testid="stDataFrame"] tbody tr:nth-child(even) {
        background-color: #f8f9fa !important;
    }
    
    .stDataFrame div[data-testid="stDataFrame"] tbody tr:nth-child(even) td {
        background-color: #f8f9fa !important;
    }
    
    /* Forzar todos los elementos de tabla */
    .stDataFrame div[data-testid="stDataFrame"] * {
        color: #495057 !important;
    }
        /* Box sizing global */
    *, *::before, *::after {
        box-sizing: border-box !important;
    }
    
    /* Fondo principal */
    .stApp {
        background-color: #f8f9fa;
    }
    
    div[data-testid="stMainBlockContainer"] {
        padding-top: 0 !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 10rem !important; /* mantiene el bottom original si quieres */
    }
    
    /* Ocultar elementos por defecto de Streamlit */
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
        max-width: 100%;
    }
    
    /* Barra de navegación superior */
    .nav-container {
        background-color: #e9ecef;
        padding: 1rem 2rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid #dee2e6;
    }
    
    .nav-tabs {
        display: flex;
        gap: 1rem;
        align-items: center;
    }
    
    .nav-tab {
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.2s;
        cursor: pointer;
        border: none;
        background: none;
        color: #6c757d;
    }
    
    .nav-tab.active {
        background-color: #6c757d;
        color: white;
    }
    
    .nav-tab:hover {
        background-color: #f8f9fa;
        color: #495057;
    }
    
    .nav-tab.active:hover {
        background-color: #5a6268;
        color: white;
    }
    
    /* Contenedor principal */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    /* Tarjetas */
    .card {
        background-color: white;
        border-radius: 0.75rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        padding: 1.25rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid #e9ecef;
        box-sizing: border-box;
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #495057;
    }
    
    /* Botones personalizados */
    .btn-primary {
        background-color: #6c757d;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .btn-primary:hover {
        background-color: #5a6268;
    }
    
    /* Mensajes de estado */
    .status-success {
        color: #6c757d;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .status-error {
        color: #6c757d;
        font-weight: 500;
    }
    
    /* Estilos para elementos de Streamlit */
    .stSelectbox > div > div {
        background-color: white;
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
    }
    
    .stDataFrame {
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
    }
    
    .stExpander {
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
    }
    
    .stExpander > div {
        background-color: white;
    }

    .stExpander [data-testid="stExpanderDetails"] {
        background-color: white;
    }
    
    /* Animaciones para páginas en construcción */
    .loading-dots {
        display: inline-flex;
        gap: 0.3rem;
    }
    
    .loading-dots span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #6c757d;
        animation: loading-dots 1.4s infinite ease-in-out both;
    }
    
    .loading-dots span:nth-child(1) {
        animation-delay: -0.32s;
    }
    
    .loading-dots span:nth-child(2) {
        animation-delay: -0.16s;
    }
    
    @keyframes loading-dots {
        0%, 80%, 100% {
            transform: scale(0);
            opacity: 0.5;
        }
        40% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    .loading-spinner {
        width: 20px;
        height: 20px;
        border: 2px solid #e9ecef;
        border-top: 2px solid #6c757d;
        border-radius: 50%;
        animation: loading-spinner 1s linear infinite;
    }
    
    @keyframes loading-spinner {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }
    
    /* Efecto de pulso para los iconos */
    .construction-icon {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }

    .stMain{
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        background-color: #f9f9f9 !important;
    }
    
    /* Asegurar que el contenido dentro de stMain se centre horizontalmente */
    .stMain > div {
        width: 100% !important;
        max-width: 1200px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
    
    /* Centrar el contenedor principal horizontalmente */
    .main-container {
        width: 100% !important;
        max-width: 1200px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
    
    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden; display: none; !important;}
    footer {visibility: hidden; display: none; !important;}
    header {visibility: hidden; display: none; !important;}
    
    /* Estilo para el contenido de tabs */

    .tab-content {
        display: none;
    }
    
    .tab-content.active {
        display: block;
    }

    /* Estilos para tabs de Streamlit */
    .stTabs {
        position: relative !important;
    }
    
    /* Estilos para tabs de Streamlit */
    .stTabs [data-baseweb="tab-list"] {
        margin-bottom: 1rem;
        display: flex;
        gap: 0.5rem;
        padding: 0.5rem;
        overflow: visible;
        align-items: center;
    }
    
    /* Estilos para tabs anidados (sub-tabs) - mismo estilo que tabs principales */
    .stTabs .stTabs [data-baseweb="tab-list"] {
        margin-bottom: 1rem;
        display: flex;
        gap: 0.5rem;
        padding: 0rem;
        overflow: visible;
        align-items: center;
    }
    
    /* Tabs anidados - mismo estilo que tabs principales */
    .stTabs .stTabs [data-baseweb="tab"] {
        display: flex;
        align-items: center;
        gap: 0.5rem; /* espacio entre icono y texto */
        background-color: #f1f3f5 !important;
        color: #495057 !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem 1.25rem !important;
        border: 1px solid #dee2e6 !important;
        box-shadow: none !important;
        transition: background-color 0.3s ease, 
                    color 0.3s ease, 
                    box-shadow 0.3s ease, 
                    transform 0.2s ease;
        font-weight: 500;
    }
    
    /* Tab anidado activo - mismo estilo que tabs principales */
    .stTabs .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #212529 !important;
        box-shadow: 0 3px 8px rgba(0,0,0,0.08) !important;
        border: 1px solid #dee2e6 !important;
        transform: translateY(-2px); /* levanta un poco el activo */
        z-index: 5;
    }
    
    /* Hover en tabs anidados inactivos - mismo estilo que tabs principales */
    .stTabs .stTabs [data-baseweb="tab"]:not([aria-selected="true"]):hover {
        background-color: #e9ecef !important;
        box-shadow: inset 0 0 6px rgba(0,0,0,0.1);
    }
    
    /* Contenedor de panel para tabs anidados - mismo estilo que tabs principales */
    .stTabs .stTabs [data-baseweb="tab-panel"] {
        background-color: #f8f9fa !important;
        border-radius: 0.5rem !important;
        padding: 1rem !important;
    }
    
    /* Transiciones para tabs anidados */
    .stTabs .stTabs [role="tab"] {
        transition: background-color 0.7s ease, 
                color 0.7s ease, 
                padding 1s ease, 
                box-shadow 0.5s ease;
    }
    
    
    /* Contenedor principal de tabs */
    .stTabs {
        overflow: visible !important;
    }
    
    /* Base de todos los tabs */
    .stTabs [data-baseweb="tab"] {
        display: flex;
        align-items: center;
        gap: 0.5rem; /* espacio entre icono y texto */
        background-color: #f1f3f5 !important;
        color: #495057 !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem 1.25rem !important;
        border: 1px solid #dee2e6 !important;
        box-shadow: none !important;
        transition: background-color 0.3s ease, 
                    color 0.3s ease, 
                    box-shadow 0.3s ease, 
                    transform 0.2s ease;
        font-weight: 500;
    }

    /* Tab activo */
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #212529 !important;
        box-shadow: 0 3px 8px rgba(0,0,0,0.08) !important;
        border: 1px solid #dee2e6 !important;
        transform: translateY(-2px); /* levanta un poco el activo */
        z-index: 5;
    }

    /* Hover en tabs inactivos */
    .stTabs [data-baseweb="tab"]:not([aria-selected="true"]):hover {
        background-color: #e9ecef !important;
        box-shadow: inset 0 0 6px rgba(0,0,0,0.1);
    }

    /* Contenedor de panel */
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #f8f9fa !important;
        border-radius: 0.5rem !important;
        padding: 1rem !important;
    }
      
    .stTabs [role="tab"] {
        transition: background-color 0.7s ease, 
                color 0.7s ease, 
                padding 1s ease, 
                box-shadow 0.5s ease;
    }

    
    /* Contenedor del contenido del tab */
    .stTabs > div > div > div > div {
        background-color: #f8f9fa !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Estilos adicionales para el contenido de tabs */
    .stTabs [role="tabpanel"] {
        padding: 1rem !important;
        background-color: #f8f9fa !important;
        border: 1px solid #e9ecef !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Asegurar que el fondo se aplique a todo el contenido */
    .stTabs .stMarkdown {
        background-color: transparent !important;
    }
    
    .stTabs .stDataFrame {
        background-color: white !important;
    }
    
    /* Títulos y texto */
    h1, h2, h3, h4, h5, h6 {
        color: #495057 !important;
    }
    
    p, div, span {
        color: #495057;
    }
    
    /* Estilo para el file uploader */
    .stFileUploader > div {
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        background-color: white !important;
        border: 2px dashed #e9ecef !important;
        border-radius: 0.75rem !important;
        color: #495057 !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"]:hover {
        background-color: #f8f9fa !important;
        border-color: #6c757d !important;
    }
    
    .stFileUploader button {
        background-color: #6c757d !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
    }

    .stFileUploaderFile button {
        width: 20px !important;
        height: 20px !important;
    }
    
    .stFileUploader button:hover {
        background-color: #5a6268 !important;
    }
    
    /* Estilos para gráficos de Plotly - forzar tema claro */
    .stPlotlyChart {
        border-radius: 0.75rem !important;
        border: 1px solid #e9ecef !important;
        box-sizing: border-box !important;
    }
    
    /* Forzar colores de texto y líneas */
    .stPlotlyChart text {
        fill: #495057 !important;
    }
    
    .stPlotlyChart .xtick text,
    .stPlotlyChart .ytick text {
        fill: #495057 !important;
    }
    
    .stPlotlyChart .xaxis-title,
    .stPlotlyChart .yaxis-title {
        fill: #495057 !important;
    }
    
    /* Forzar líneas de cuadrícula claras */
    .stPlotlyChart .gridlayer .xgrid,
    .stPlotlyChart .gridlayer .ygrid {
        stroke: #e9ecef !important;
    }
    
    /* Forzar bordes de ejes */
    .stPlotlyChart .xaxis .domain,
    .stPlotlyChart .yaxis .domain {
        stroke: #e9ecef !important;
    }
    
    /* Forzar tema claro en el contenedor SVG específico */
    .stPlotlyChart .user-select-none {
    }
    
    .stPlotlyChart .svg-container {
    }
    
    /* Forzar todos los elementos de Plotly a tema claro */
    .stPlotlyChart * {
    }
    
    .stPlotlyChart .plot-container * {
    }
    
    /* Estilos para el selectbox */
    .stSelectbox > div > div {
        border: 1px solid #e9ecef !important;
        border-radius: 0.75rem !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #6c757d !important;
    }
    
    /* Estilos para expanders */
    .streamlit-expanderHeader {
        border: 1px solid #e9ecef !important;
        border-radius: 0.75rem !important;
        color: #495057 !important;
    }
    
    .streamlit-expanderContent {
        border: 1px solid #e9ecef !important;
        border-top: none !important;
        border-radius: 0 0 0.75rem 0.75rem !important;
    }
    
    /* Estilos para tablas de Streamlit - Tema claro forzado */
    .stDataFrame {
        background-color: white !important;
        color: #495057 !important;
    }
    
    .stDataFrame table {
        background-color: white !important;
        color: #495057 !important;
        border-collapse: collapse !important;
    }
    
    .stDataFrame thead {
        background-color: #f8f9fa !important;
    }
    
    .stDataFrame thead th {
        background-color: #f8f9fa !important;
        color: #495057 !important;
        border: 1px solid #e9ecef !important;
        font-weight: 600 !important;
        padding: 0.75rem !important;
    }
    
    .stDataFrame tbody {
        background-color: white !important;
    }
    
    .stDataFrame tbody td {
        background-color: white !important;
        color: #495057 !important;
        border: 1px solid #e9ecef !important;
        padding: 0.75rem !important;
    }
    
    .stDataFrame tbody tr {
        background-color: white !important;
    }
    
    .stDataFrame tbody tr:nth-child(even) {
        background-color: #f8f9fa !important;
    }
    
    .stDataFrame tbody tr:nth-child(even) td {
        background-color: #f8f9fa !important;
    }
    
    .stDataFrame tbody tr:hover {
        background-color: #e9ecef !important;
    }
    
    .stDataFrame tbody tr:hover td {
        background-color: #e9ecef !important;
    }
    
    /* Estilos adicionales para el contenedor de la tabla */
    .stDataFrame > div {
        background-color: white !important;
        border: 1px solid #e9ecef !important;
        border-radius: 0.75rem !important;
        overflow: hidden !important;
    }
    
    /* Forzar tema claro en todos los elementos de tabla */
    .stDataFrame * {
        color: #495057 !important;
    }
    
    /* Estilos específicos para el scrollbar de las tablas */
    .stDataFrame::-webkit-scrollbar {
        background-color: #f8f9fa !important;
    }
    
    .stDataFrame::-webkit-scrollbar-thumb {
        background-color: #e9ecef !important;
        border-radius: 0.5rem !important;
    }
    
    .stDataFrame::-webkit-scrollbar-thumb:hover {
        background-color: #dee2e6 !important;
    }
    
    /* Forzar tema claro en todos los elementos de tabla de Streamlit */
    .stDataFrame div[data-testid="stDataFrame"] {
        background-color: white !important;
        color: #495057 !important;
    }
    
    /* Estilos para elementos específicos de tabla que pueden usar tema oscuro */
    .stDataFrame .dataframe {
        background-color: white !important;
        color: #495057 !important;
    }
    
    .stDataFrame .dataframe thead th {
        background-color: #f8f9fa !important;
        color: #495057 !important;
        border: 1px solid #e9ecef !important;
    }
    
    .stDataFrame .dataframe tbody td {
        background-color: white !important;
        color: #495057 !important;
        border: 1px solid #e9ecef !important;
    }
    
    .stDataFrame .dataframe tbody tr:nth-child(even) {
        background-color: #f8f9fa !important;
    }
    
    .stDataFrame .dataframe tbody tr:nth-child(even) td {
        background-color: #f8f9fa !important;
    }
    
    /* Forzar colores de texto en todos los elementos de tabla */
    .stDataFrame .dataframe * {
        color: #495057 !important;
    }
    
    /* Estilos para el contenedor de la tabla */
    .stDataFrame .dataframe-container {
        background-color: white !important;
        border: 1px solid #e9ecef !important;
        border-radius: 0.75rem !important;
    }
    
    /* Estilos para tarjetas de promociones completas */
    .promo-card-full {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e9ecef;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        display: flex;
        min-height: 200px;
    }
    
    .promo-card-full:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .promo-image-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 250px;
        position: relative;
    }
    
    .promo-main-logo {
        height: 80px;
        max-width: 200px;
        object-fit: contain;
        margin-bottom: 1rem;
    }
    
    .promo-badge {
        background: linear-gradient(45deg, #ff6b6b, #ff8e8e);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        position: absolute;
        top: 1rem;
        right: 1rem;
    }
    
    .promo-content-section {
        flex: 1;
        padding: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .promo-info {
        flex: 1;
        margin-right: 2rem;
    }
    
    .promo-info h3 {
        color: #212529;
        margin: 0 0 1rem 0;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .promo-description {
        color: #495057;
        font-size: 1rem;
        line-height: 1.6;
        margin: 0 0 1.5rem 0;
    }
    
    .promo-discount {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
        margin: 0 0 1rem 0;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3);
        display: inline-block;
    }
    
    .promo-code {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 1rem 0;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
    }
    
    .promo-code code {
        background-color: #e9ecef;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-family: 'Courier New', monospace;
        font-weight: 600;
        color: #495057;
        flex: 1;
        font-size: 1.1rem;
    }
    
    .copy-btn {
        background-color: #6c757d;
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 0.25rem;
        cursor: pointer;
        transition: background-color 0.2s;
        font-size: 1.1rem;
    }
    
    .copy-btn:hover {
        background-color: #5a6268;
    }
    
    .promo-actions {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1.5rem;
        min-width: 200px;
    }
    
    .countdown-timer {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #ff6b6b, #ff8e8e);
        border-radius: 0.75rem;
        color: white;
        box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3);
    }
    
    .timer-label {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .timer-display {
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'Courier New', monospace;
    }
    
    .timer-display span {
        background-color: rgba(255, 255, 255, 0.2);
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        margin: 0 0.25rem;
    }
    
    .promo-link {
        text-decoration: none;
        display: block;
    }
    
    .promo-button {
        background: linear-gradient(45deg, #007bff, #0056b3);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 0.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        min-width: 180px;
    }
    
    .promo-button:hover {
        background: linear-gradient(45deg, #0056b3, #004085);
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0, 123, 255, 0.3);
    }
    
    /* Animación de copiado */
    .copy-success {
        background-color: #28a745 !important;
        animation: copyPulse 0.6s ease;
    }
    
    @keyframes copyPulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
</style>

<script>
function copyCode(elementId) {
    const codeElement = document.getElementById(elementId);
    const code = codeElement.textContent;
    
    navigator.clipboard.writeText(code).then(function() {
        const button = codeElement.nextElementSibling;
        const originalText = button.innerHTML;
        button.innerHTML = '✅';
        button.classList.add('copy-success');
        
        setTimeout(function() {
            button.innerHTML = originalText;
            button.classList.remove('copy-success');
        }, 2000);
    }).catch(function(err) {
        console.error('Error al copiar: ', err);
        alert('Código copiado: ' + code);
    });
}
</script>
""", unsafe_allow_html=True)
# Estado de la sesión para manejar tabs
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'analisis'

# Usar tabs de Streamlit para el contenido
tab1, tab2, tab3 = st.tabs(["📊 Análisis", "% Descuentos", "⚙️ Configuración"])

# Contenedor principal
st.markdown('<div class="main-container">', unsafe_allow_html=True)


# Panel central con contenido dinámico según el tab
with tab1:
    # Crear tabs anidados dentro del tab 1
    sub_tab1, sub_tab2 = st.tabs(["📈 Análisis de Portafolio", "🔍 Análisis de Prop Firm"])
    
    with sub_tab1:
        # Tarjeta de carga de archivo
        st.markdown("""
        <div class="card">
            <div class="card-title">Analiza tu portafolio</div>
        """, unsafe_allow_html=True)

        archivo = st.file_uploader("Elegir archivo", type=["htm", "html"], label_visibility="collapsed")

        if archivo:
            try:
                soup = BeautifulSoup(archivo, 'html.parser')
                rows = soup.find_all('tr', align='right')

                datos = []
                eas_filtradas = 0
                total_operaciones = 0

                for i in range(0, len(rows), 2):
                    try:
                        fila_op = rows[i].find_all('td')
                        fila_ea = rows[i+1].find_all('td')

                        tipo = fila_op[2].text.strip().lower()
                        size = float(fila_op[3].text)
                        symbol = fila_op[4].text.strip().lower()
                        open_time = datetime.strptime(fila_op[1].text.strip(), "%Y.%m.%d %H:%M:%S")
                        close_time = datetime.strptime(fila_op[8].text.strip(), "%Y.%m.%d %H:%M:%S")
                        profit = float(fila_op[13].text)
                        ea_raw = fila_ea[-1].text.strip()

                        if "cancelled" in ea_raw.lower():
                            continue

                        ea_name = ea_raw.split('[')[0]
                        total_operaciones += 1

                        # Filtrar EAs válidas (que contengan letras)
                        if not es_ea_valida(ea_name):
                            eas_filtradas += 1
                            continue

                        datos.append({
                            "EA": ea_name,
                            "Símbolo": symbol,
                            "Tipo": tipo,
                            "Beneficio": profit,
                            "Open": open_time,
                            "Close": close_time,
                            "Duración": (close_time - open_time).total_seconds() / 60  # en minutos
                        })
                    except Exception:
                        continue

                if datos:
                    df = pd.DataFrame(datos)
                    
                    # Mostrar información sobre el filtrado
                    if eas_filtradas > 0:
                        st.info(f"ℹ️ Se eliminaron {eas_filtradas} operaciones que no pertenecen a ninguna EA de un total de {total_operaciones} operaciones. Se procesaron {len(datos)} operaciones válidas.")

                    resumen = df.groupby(["EA", "Símbolo"]).agg(
                        Ops=('Beneficio', 'count'),
                        Win_pct=('Beneficio', lambda x: 100 * (x > 0).sum() / len(x)),
                        Profit_medio=('Beneficio', 'mean'),
                        Max_Loss=('Beneficio', 'min'),
                        Duracion_media_min=('Duración', 'mean'),
                        Beneficio_total=('Beneficio', 'sum')
                    ).reset_index()

                    # Redondear numéricos primero
                    resumen = resumen.round({
                        "Win_pct": 2,
                        "Profit_medio": 2,
                        "Max_Loss": 2,
                        "Duracion_media_min": 1,
                        "Beneficio_total": 2
                    })

                    resumen["Beneficio_total_raw"] = resumen["Beneficio_total"]

                    # 💡 Formatear columnas para presentación legible
                    def formatear_duracion(minutos):
                        horas = int(minutos) // 60
                        mins = int(minutos) % 60
                        return f"{horas}h {mins}m"

                    resumen["Win_pct"] = resumen["Win_pct"].astype(str) + " %"
                    resumen["Profit_medio"] = resumen["Profit_medio"].apply(lambda x: f"${x:.2f}")
                    resumen["Max_Loss"] = resumen["Max_Loss"].apply(lambda x: f"${x:.2f}")
                    resumen["Beneficio_total"] = resumen["Beneficio_total"].apply(lambda x: f"${x:.2f}")
                    resumen["Duracion_media"] = resumen["Duracion_media_min"].apply(formatear_duracion)
                    resumen = resumen.drop(columns=["Duracion_media_min"])  # Quitamos la versión cruda

                    # Ordenar y preparar resumen
                    resumen = resumen.sort_values(by="Beneficio_total_raw", ascending=False)

                    # Selector de EA para filtrar
                    ea_opciones = ["Todas"] + sorted(resumen["EA"].unique())
                    ea_seleccionada = st.selectbox("🧠 Selecciona una EA para filtrar", ea_opciones)

                    # Filtrar datos según selección
                    if ea_seleccionada != "Todas":
                        resumen_filtrado = resumen[resumen["EA"] == ea_seleccionada]
                        df_filtrado = df[df["EA"] == ea_seleccionada]
                    else:
                        resumen_filtrado = resumen
                        df_filtrado = df

                    # Crear gráfico de beneficio acumulado
                    df_filtrado['Fecha'] = df_filtrado['Close'].dt.date
                    beneficios_diarios = df_filtrado.groupby(['EA', 'Fecha'])['Beneficio'].sum().reset_index()
                    beneficios_diarios['Beneficio_acumulado'] = beneficios_diarios.groupby('EA')['Beneficio'].cumsum()

                    if len(beneficios_diarios) > 0:
                        fig = px.line(
                            beneficios_diarios,
                            x="Fecha",
                            y="Beneficio_acumulado",
                            color="EA",
                            markers=True,
                            labels={"Beneficio_acumulado": "Beneficio acumulado", "Fecha": "Fecha"}
                        )

                        # Añadir tooltip personalizado
                        fig.update_traces(
                            hovertemplate=
                            "<b>Fecha:</b> %{x|%d-%m-%Y}<br>" +
                            "<b>Beneficio acumulado:</b> $%{y:.2f}<extra></extra>"
                        )

                        fig.update_layout(
                            height=400, 
                            margin=dict(l=20, r=20, t=40, b=20),
                            template="plotly_white",
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(color='#495057'),
                            xaxis=dict(
                                gridcolor='#e9ecef',
                                linecolor='#e9ecef',
                                tickcolor='#495057',
                                tickfont=dict(color='#495057')
                            ),
                            yaxis=dict(
                                gridcolor='#e9ecef',
                                linecolor='#e9ecef',
                                tickcolor='#495057',
                                tickfont=dict(color='#495057')
                            )
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No hay datos suficientes para mostrar el gráfico")
                
                    # Cerrar tarjeta de análisis
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Panel de Ranking por Score de Rentabilidad
                    st.markdown("""
                    <div class="card">
                        <div class="card-title">🏆 Ranking por Score de Rentabilidad</div>
                    """, unsafe_allow_html=True)
                
                    # Crear ranking
                    ranking_df = crear_ranking_ea(df_filtrado)
                    
                    if not ranking_df.empty:
                        # Crear tabla de ranking con columnas formateadas
                        ranking_mostrar = ranking_df[[
                            'Posicion', 'EA', 'Símbolo', 'Total_Ops', 'Win_Rate_Formateado',
                            'Beneficio_Promedio_Formateado', 'Perdida_Promedio_Formateado',
                            'Ratio_Formateado', 'Score_Formateado', 'Beneficio_Total_Formateado'
                        ]].copy()
                        
                        # Renombrar columnas para mejor presentación
                        ranking_mostrar.columns = [
                            'Posición', 'EA', 'Símbolo', 'Total Ops', 'Win Rate',
                            'Beneficio Promedio', 'Pérdida Promedio', 'Ratio R/B', 'Score Rentabilidad', 'Beneficio Total'
                        ]
                    
                        # Configurar columnas para tema claro
                        column_config = {
                            "Posición": st.column_config.NumberColumn(
                                "Posición",
                                help="Posición en el ranking",
                                format="%d"
                            ),
                            "EA": st.column_config.TextColumn(
                                "EA",
                                help="Nombre del Expert Advisor"
                            ),
                            "Símbolo": st.column_config.TextColumn(
                                "Símbolo",
                                help="Símbolo de trading"
                            ),
                            "Total Ops": st.column_config.NumberColumn(
                                "Total Ops",
                                help="Total de operaciones",
                                format="%d"
                            ),
                            "Win Rate": st.column_config.TextColumn(
                                "Win Rate",
                                help="Porcentaje de operaciones ganadoras"
                            ),
                            "Beneficio Promedio": st.column_config.TextColumn(
                                "Beneficio Promedio",
                                help="Beneficio promedio por operación ganadora"
                            ),
                            "Pérdida Promedio": st.column_config.TextColumn(
                                "Pérdida Promedio",
                                help="Pérdida promedio por operación perdedora"
                            ),
                            "Ratio R/B": st.column_config.TextColumn(
                                "Ratio R/B",
                                help="Ratio Riesgo-Beneficio"
                            ),
                            "Score Rentabilidad": st.column_config.TextColumn(
                                "Score Rentabilidad",
                                help="Score que combina Win Rate y Ratio R/B (Win Rate × Ratio R/B)"
                            ),
                            "Beneficio Total": st.column_config.TextColumn(
                                "Beneficio Total",
                                help="Beneficio total acumulado"
                            )
                        }
                        
                        # Mostrar tabla de ranking con configuración de tema claro
                        st.dataframe(
                            ranking_mostrar, 
                            use_container_width=True,
                            column_config=column_config,
                            hide_index=True
                        )
                        
                        # Explicación del ranking
                        st.markdown("""
                        <div style="margin-top: -1rem; padding: 1rem; background-color: #f8f9fa; border-left: 4px solid #6c757d;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #495057;">📊 Cómo se calcula el Score de Rentabilidad:</h4>
                            <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">
                                <strong>Score de Rentabilidad = Win Rate × Ratio Riesgo-Beneficio</strong><br>
                                Este score combina la frecuencia de ganancias (Win Rate) con la eficiencia (Ratio R/B).<br>
                                <em>Ejemplo:</em> EA con 60% Win Rate y Ratio 2:1 = Score 1.2, mientras que EA con 20% Win Rate y Ratio 4:1 = Score 0.8
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No hay datos suficientes para crear el ranking")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Tarjeta de tabla resumen
                    st.markdown("""
                    <div class="card">
                        <div class="card-title">💰 Ranking por beneficio total</div>
                    """, unsafe_allow_html=True)
                    
                    # Crear una copia para mostrar sin la columna raw
                    resumen_mostrar = resumen_filtrado.drop(columns=['Beneficio_total_raw'])
                    
                    # Configurar columnas para la tabla comparativa
                    column_config_resumen = {
                        "EA": st.column_config.TextColumn(
                            "EA",
                            help="Nombre del Expert Advisor"
                        ),
                        "Símbolo": st.column_config.TextColumn(
                            "Símbolo",
                            help="Símbolo de trading"
                        ),
                        "Ops": st.column_config.NumberColumn(
                            "Ops",
                            help="Número de operaciones",
                            format="%d"
                        ),
                        "Win_pct": st.column_config.TextColumn(
                            "Win %",
                            help="Porcentaje de operaciones ganadoras"
                        ),
                        "Profit_medio": st.column_config.TextColumn(
                            "Profit Medio",
                            help="Beneficio promedio por operación"
                        ),
                        "Max_Loss": st.column_config.TextColumn(
                            "Max Loss",
                            help="Pérdida máxima registrada"
                        ),
                        "Duracion_media": st.column_config.TextColumn(
                            "Duración Media",
                            help="Duración promedio de las operaciones"
                        ),
                        "Beneficio_total": st.column_config.TextColumn(
                            "Beneficio Total",
                            help="Beneficio total acumulado"
                        )
                    }
                    
                    st.dataframe(
                        resumen_mostrar, 
                        use_container_width=True,
                        column_config=column_config_resumen,
                        hide_index=True
                    )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Tarjeta de estadísticas por EA
                    st.markdown("""
                    <div class="card">
                        <div class="card-title">📊 Estadísticas por EA</div>
                    """, unsafe_allow_html=True)
                    
                    # Crear análisis estadístico detallado
                    estadisticas_detalladas = []
                    for ea in df_filtrado['EA'].unique():
                        df_ea = df_filtrado[df_filtrado['EA'] == ea]
                        
                        # Calcular estadísticas por EA
                        total_ops = len(df_ea)
                        ops_ganadoras = len(df_ea[df_ea['Beneficio'] > 0])
                        ops_perdedoras = len(df_ea[df_ea['Beneficio'] < 0])
                        ops_cero = len(df_ea[df_ea['Beneficio'] == 0])
                        
                        win_rate = (ops_ganadoras / total_ops * 100) if total_ops > 0 else 0
                        beneficio_total = df_ea['Beneficio'].sum()
                        beneficio_promedio = df_ea['Beneficio'].mean()
                        beneficio_max = df_ea['Beneficio'].max()
                        perdida_max = df_ea['Beneficio'].min()
                        
                        # Duración promedio
                        duracion_promedio = df_ea['Duración'].mean()
                        
                        estadisticas_detalladas.append({
                            'EA': ea,
                            'Total_Operaciones': total_ops,
                            'Operaciones_Ganadoras': ops_ganadoras,
                            'Operaciones_Perdedoras': ops_perdedoras,
                            'Operaciones_Cero': ops_cero,
                            'Win_Rate_%': round(win_rate, 2),
                            'Beneficio_Total': round(beneficio_total, 2),
                            'Beneficio_Promedio': round(beneficio_promedio, 2),
                            'Beneficio_Maximo': round(beneficio_max, 2),
                            'Perdida_Maxima': round(perdida_max, 2),
                            'Duracion_Promedio_Min': round(duracion_promedio, 1)
                        })
                    
                    if estadisticas_detalladas:
                        df_estadisticas = pd.DataFrame(estadisticas_detalladas)
                        df_estadisticas = df_estadisticas.sort_values('Beneficio_Total', ascending=False)
                        
                        # Configurar columnas para la tabla de estadísticas
                        column_config_stats = {
                            "EA": st.column_config.TextColumn("EA", help="Nombre del Expert Advisor"),
                            "Total_Operaciones": st.column_config.NumberColumn("Total Ops", help="Total de operaciones", format="%d"),
                            "Operaciones_Ganadoras": st.column_config.NumberColumn("Ops Ganadoras", help="Operaciones ganadoras", format="%d"),
                            "Operaciones_Perdedoras": st.column_config.NumberColumn("Ops Perdedoras", help="Operaciones perdedoras", format="%d"),
                            "Operaciones_Cero": st.column_config.NumberColumn("Ops Cero", help="Operaciones con beneficio cero", format="%d"),
                            "Win_Rate_%": st.column_config.NumberColumn("Win Rate %", help="Porcentaje de operaciones ganadoras", format="%.2f"),
                            "Beneficio_Total": st.column_config.NumberColumn("Beneficio Total", help="Beneficio total acumulado", format="$%.2f"),
                            "Beneficio_Promedio": st.column_config.NumberColumn("Beneficio Promedio", help="Beneficio promedio por operación", format="$%.2f"),
                            "Beneficio_Maximo": st.column_config.NumberColumn("Beneficio Máximo", help="Mayor beneficio individual", format="$%.2f"),
                            "Perdida_Maxima": st.column_config.NumberColumn("Pérdida Máxima", help="Mayor pérdida individual", format="$%.2f"),
                            "Duracion_Promedio_Min": st.column_config.NumberColumn("Duración Promedio (min)", help="Duración promedio en minutos", format="%.1f")
                        }
                        
                        st.dataframe(
                            df_estadisticas, 
                            use_container_width=True,
                            column_config=column_config_stats,
                            hide_index=True
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Tarjeta de operaciones detalladas por estrategia
                    st.markdown("""
                    <div class="card">
                        <div class="card-title">📋 Operaciones por Estrategia</div>
                    """, unsafe_allow_html=True)
                    
                    # Mostrar operaciones individuales por EA y símbolo
                    grupos_ordenados = df_filtrado.groupby(["EA", "Símbolo"]).agg(Beneficio_total=('Beneficio', 'sum')).reset_index()
                    grupos_ordenados = grupos_ordenados.sort_values(by="Beneficio_total", ascending=False)

                    for _, row in grupos_ordenados.iterrows():
                        ea = row["EA"]
                        symbol = row["Símbolo"]
                        grupo = df_filtrado[(df_filtrado["EA"] == ea) & (df_filtrado["Símbolo"] == symbol)]
                        with st.expander(f"📌 {ea} - {symbol} ({len(grupo)} operaciones)"):
                            st.dataframe(grupo.sort_values(by="Open"), use_container_width=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                else:
                    st.markdown("""
                    <div class="status-error">
                        ❌ No se encontraron operaciones válidas en el archivo.
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            except Exception as e:
                st.markdown(f"""
                <div class="status-error">
                    ❌ Error al procesar el archivo<br>
                    Verifica que sea un archivo HTML válido de MT4.<br>
                    Error: {str(e)}
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="status-error">
                ⚠️ Selecciona un archivo para comenzar el análisis
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with sub_tab2:
        # Tarjeta de análisis de Prop Firm
        st.markdown("""
        <div class="card">
            <div class="card-title">🔍 Análisis de Prop Firm</div>
        """, unsafe_allow_html=True)

        # Contenedor para subir archivo CSV
        archivo_csv = st.file_uploader("Elegir archivo CSV", type=["csv"], label_visibility="collapsed", key="prop_firm_csv")

        if archivo_csv:
            try:
                # Leer el archivo CSV
                df_csv = pd.read_csv(archivo_csv, sep=';')
                
                # Mostrar información básica del archivo
                st.success(f"✅ Archivo cargado exitosamente: {len(df_csv)} operaciones encontradas")
                
                # Selectores de configuración básica
                col1, col2 = st.columns(2)
                
                with col1:
                    riesgo_por_operacion = st.selectbox(
                        "💰 Riesgo por operación",
                        options=[1, 2, 3, 4, 5],
                        format_func=lambda x: f"{x}%",
                        help="Porcentaje de la cuenta que se arriesga por operación"
                    )
                
                with col2:
                    tamaño_cuenta = st.selectbox(
                        "💵 Tamaño de cuenta de fondeo",
                        options=[10000, 25000, 50000, 100000, 200000],
                        format_func=lambda x: f"${x:,}",
                        help="Tamaño de la cuenta de fondeo"
                    )
                
                # Segunda fila de selectores para objetivos y límites
                st.markdown("---")
                st.markdown("### 🎯 Objetivos de Ganancia y Límites de Drawdown")
                
                col3, col4, col5, col6 = st.columns(4)
                
                with col3:
                    porcentaje_fase1 = st.selectbox(
                        "🎯 Ganancia objetivo Fase 1",
                        options=[5, 8, 10, 12, 15],
                        index=2,
                        format_func=lambda x: f"{x}%",
                        help="Porcentaje de ganancia objetivo para completar Fase 1"
                    )
                
                with col4:
                    porcentaje_fase2 = st.selectbox(
                        "🎯 Ganancia objetivo Fase 2",
                        options=[3, 5, 7, 10, 12],
                        index=1,
                        format_func=lambda x: f"{x}%",
                        help="Porcentaje de ganancia objetivo para completar Fase 2"
                    )
                
                with col5:
                    drawdown_maximo_diario = st.selectbox(
                        "📉 Drawdown máximo diario",
                        options=[3, 4, 5, 6, 7, 8, 9, 10],
                        index=1,
                        format_func=lambda x: f"{x}%",
                        help="Drawdown máximo permitido en un solo día. Si pierdes más, suspendes el examen"
                    )
                
                with col6:
                    drawdown_maximo_total = st.selectbox(
                        "📊 Drawdown máximo total",
                        options=[5, 6, 7, 8, 9, 10, 12, 15],
                        index=1,
                        format_func=lambda x: f"{x}%",
                        help="Drawdown máximo total permitido durante toda la evaluación. No se puede superar en ninguna fase"
                    )
                
                # Calcular análisis de prop firm
                if st.button("🚀 Analizar Prop Firm", type="primary"):
                    with st.spinner("Analizando datos..."):
                        # Procesar datos del CSV
                        df_csv['Profit/Loss'] = df_csv['Profit/Loss'].astype(float)
                        df_csv['Balance'] = df_csv['Balance'].astype(float)
                        
                        # Calcular métricas básicas
                        total_operaciones = len(df_csv)
                        operaciones_ganadoras = len(df_csv[df_csv['Profit/Loss'] > 0])
                        operaciones_perdedoras = len(df_csv[df_csv['Profit/Loss'] < 0])
                        win_rate = (operaciones_ganadoras / total_operaciones * 100) if total_operaciones > 0 else 0
                        
                        # Calcular pérdida máxima y ganancia máxima
                        perdida_maxima = df_csv['Profit/Loss'].min()
                        ganancia_maxima = df_csv['Profit/Loss'].max()
                        
                        # Calcular drawdown máximo
                        df_csv_sorted = df_csv.sort_values('Close time')
                        df_csv_sorted['Balance_acumulado'] = df_csv_sorted['Balance'].cummax()
                        df_csv_sorted['Drawdown'] = (df_csv_sorted['Balance_acumulado'] - df_csv_sorted['Balance']) / df_csv_sorted['Balance_acumulado'] * 100
                        drawdown_maximo = df_csv_sorted['Drawdown'].max()
                        
                        # Los límites de drawdown son globales para toda la evaluación
                        # No hay límites específicos por fase, solo los drawdowns globales
                        
                        # Calcular riesgo por operación en dinero
                        riesgo_por_op_dinero = tamaño_cuenta * (riesgo_por_operacion / 100)
                        
                        # Análisis de probabilidades basado en drawdown
                        # Calcular drawdown máximo histórico
                        drawdown_maximo_historico = df_csv_sorted['Drawdown'].max()
                        
                        # Calcular probabilidad de exceder drawdown diario
                        # Simular pérdida diaria máxima basada en el riesgo por operación
                        perdida_maxima_diaria_simulada = abs(df_csv['Profit/Loss'].min()) * (riesgo_por_operacion / 2.0)  # Escalado por riesgo
                        drawdown_diario_simulado = (perdida_maxima_diaria_simulada / tamaño_cuenta) * 100
                        
                        # Calcular probabilidades aproximadas
                        prob_exceder_drawdown_diario = 100 if drawdown_diario_simulado > drawdown_maximo_diario else 0
                        prob_exceder_drawdown_total = 100 if drawdown_maximo_historico > drawdown_maximo_total else 0
                        
                        # Probabilidades de pasar (aproximación)
                        prob_pasar_fase1 = max(0, 100 - prob_exceder_drawdown_diario - prob_exceder_drawdown_total)
                        prob_pasar_fase2 = max(0, 100 - prob_exceder_drawdown_diario - prob_exceder_drawdown_total)
                        
                        # ANÁLISIS DE TIEMPO PARA PROP FIRMS
                        # Convertir fechas a datetime si no lo están
                        df_csv['Open time'] = pd.to_datetime(df_csv['Open time'])
                        df_csv['Close time'] = pd.to_datetime(df_csv['Close time'])
                        
                        # Calcular operaciones por día
                        df_csv['Fecha'] = df_csv['Close time'].dt.date
                        operaciones_por_dia = df_csv.groupby('Fecha').agg({
                            'Profit/Loss': ['count', 'sum', 'min', 'max'],
                            'Balance': 'last'
                        }).reset_index()
                        
                        # Aplanar columnas
                        operaciones_por_dia.columns = ['Fecha', 'Ops_Dia', 'Profit_Dia', 'Min_Profit_Dia', 'Max_Profit_Dia', 'Balance_Final']
                        operaciones_por_dia['Fecha'] = pd.to_datetime(operaciones_por_dia['Fecha'])
                        
                        # Calcular drawdown diario
                        operaciones_por_dia['Balance_Max'] = operaciones_por_dia['Balance_Final'].cummax()
                        operaciones_por_dia['Drawdown_Dia'] = (operaciones_por_dia['Balance_Max'] - operaciones_por_dia['Balance_Final']) / operaciones_por_dia['Balance_Max'] * 100
                        
                        # Simular escenarios de prop firm con fases secuenciales REALISTAS (sin límite de días)
                        def simular_prop_firm(df_diario, riesgo_por_op, drawdown_diario, drawdown_total, dias_simulacion=30):
                            resultados = []
                            
                            # ESCALAR OPERACIONES SEGÚN EL RIESGO SELECCIONADO
                            riesgo_historico_promedio = 2.0  # 2% promedio histórico
                            factor_escalado = riesgo_por_op / riesgo_historico_promedio
                            
                            # Escalar todo el dataset una vez
                            df_diario['Profit_Dia_Escalado'] = df_diario['Profit_Dia'] * factor_escalado
                            df_diario['Min_Profit_Dia_Escalado'] = df_diario['Min_Profit_Dia'] * factor_escalado
                            df_diario['Max_Profit_Dia_Escalado'] = df_diario['Max_Profit_Dia'] * factor_escalado
                            
                            # Usar todo el dataset disponible, no limitar a 30 días
                            for inicio in range(len(df_diario) - 10):  # Mínimo 10 días para tener datos suficientes
                                # Tomar desde el inicio hasta el final del dataset
                                periodo = df_diario.iloc[inicio:].copy()
                                
                                # SIMULACIÓN SECUENCIAL DE FASES CON CONTADOR DE TRADES
                                balance_inicial = tamaño_cuenta
                                balance_acumulado = balance_inicial
                                balance_max = balance_inicial
                                drawdown_max = 0
                                
                                # Objetivos de cada fase usando los selectores
                                objetivo_fase1 = balance_inicial * (porcentaje_fase1 / 100)  # Ganancia objetivo Fase 1
                                objetivo_fase2 = balance_inicial * (porcentaje_fase2 / 100)  # Ganancia objetivo Fase 2
                                
                                # Variables de seguimiento
                                fase_actual = 1
                                dias_fase1 = 0
                                dias_fase2 = 0
                                dias_totales = 0
                                trades_totales = 0  # Contador de trades que NO se resetea
                                trades_fase1 = 0
                                trades_fase2 = 0
                                resultado = "En progreso"
                                profit_total = 0
                                
                                for i, (_, row) in enumerate(periodo.iterrows()):
                                    dias_totales += 1
                                    trades_del_dia = row['Ops_Dia']
                                    trades_totales += trades_del_dia
                                    
                                    balance_acumulado += row['Profit_Dia_Escalado']
                                    profit_total = balance_acumulado - balance_inicial
                                    balance_max = max(balance_max, balance_acumulado)
                                    
                                    # Calcular drawdown actual
                                    drawdown_actual = (balance_max - balance_acumulado) / balance_max * 100
                                    drawdown_max = max(drawdown_max, drawdown_actual)
                                    
                                    # Calcular drawdown diario (pérdida del día vs balance inicial del día)
                                    balance_inicio_dia = balance_acumulado - row['Profit_Dia_Escalado']
                                    drawdown_diario_actual = abs(row['Min_Profit_Dia_Escalado']) / balance_inicio_dia * 100 if balance_inicio_dia > 0 else 0
                                    
                                    # Verificar límites de drawdown globales (aplican a ambas fases)
                                    if (drawdown_max >= drawdown_total or 
                                        drawdown_diario_actual >= drawdown_diario):
                                        # Se quema por drawdown (aplica a cualquier fase)
                                        if fase_actual == 1:
                                            resultado = "Quemado Fase 1"
                                            dias_fase1 = dias_totales
                                            trades_fase1 = trades_totales
                                        else:
                                            resultado = "Quemado Fase 2"
                                            dias_fase2 = dias_totales - dias_fase1
                                            trades_fase2 = trades_totales - trades_fase1
                                        break
                                    
                                    # Verificar objetivos de ganancia por fase
                                    if fase_actual == 1:
                                        # FASE 1: Verificar si pasa
                                        if profit_total >= objetivo_fase1:
                                            # Pasó Fase 1, ahora empieza Fase 2
                                            fase_actual = 2
                                            dias_fase1 = dias_totales
                                            trades_fase1 = trades_totales
                                            # Resetear para Fase 2
                                            balance_inicial_fase2 = balance_acumulado
                                            objetivo_fase2_absoluto = balance_inicial_fase2 + (balance_inicial * porcentaje_fase2 / 100)
                                    
                                    elif fase_actual == 2:
                                        # FASE 2: Verificar si pasa
                                        if profit_total >= objetivo_fase2_absoluto:
                                            resultado = "Pasó Fase 2"
                                            dias_fase2 = dias_totales - dias_fase1
                                            trades_fase2 = trades_totales - trades_fase1
                                            break
                                
                                # Si no se completó ninguna fase (llegó al final del dataset)
                                if resultado == "En progreso":
                                    if fase_actual == 1:
                                        resultado = "No completó Fase 1"
                                        dias_fase1 = dias_totales
                                        trades_fase1 = trades_totales
                                    elif fase_actual == 2:
                                        resultado = "No completó Fase 2"
                                        dias_fase2 = dias_totales - dias_fase1
                                        trades_fase2 = trades_totales - trades_fase1
                                
                                # Calcular métricas finales
                                total_ops = periodo['Ops_Dia'].sum()
                                
                                resultados.append({
                                    'Inicio': periodo['Fecha'].iloc[0],
                                    'Fin': periodo['Fecha'].iloc[-1],
                                    'Dias_Fase1': dias_fase1,
                                    'Dias_Fase2': dias_fase2,
                                    'Dias_Totales': dias_totales,
                                    'Trades_Totales': trades_totales,
                                    'Trades_Fase1': trades_fase1,
                                    'Trades_Fase2': trades_fase2,
                                    'Total_Ops': total_ops,
                                    'Profit_Total': profit_total,
                                    'Drawdown_Max': drawdown_max,
                                    'Fase_Alcanzada': fase_actual,
                                    'Resultado': resultado
                                })
                                
                                # Limitar el número de simulaciones para evitar sobrecarga
                                if len(resultados) >= 100:  # Máximo 100 simulaciones
                                    break
                            
                            return pd.DataFrame(resultados)
                        
                        # Ejecutar simulaciones (ahora sin límite de días)
                        simulaciones = simular_prop_firm(operaciones_por_dia, riesgo_por_operacion, drawdown_maximo_diario, drawdown_maximo_total)
                        
                        # Calcular estadísticas de tiempo y trades
                        def calcular_estadisticas_tiempo(simulaciones):
                            # Filtrar por resultados específicos
                            pasaron_fase1 = simulaciones[simulaciones['Resultado'] == 'Pasó Fase 1']
                            pasaron_fase2 = simulaciones[simulaciones['Resultado'] == 'Pasó Fase 2']
                            quemaron_fase1 = simulaciones[simulaciones['Resultado'] == 'Quemado Fase 1']
                            quemaron_fase2 = simulaciones[simulaciones['Resultado'] == 'Quemado Fase 2']
                            
                            # También contar los que pasaron Fase 1 pero no Fase 2
                            pasaron_solo_fase1 = simulaciones[simulaciones['Fase_Alcanzada'] >= 1]
                            
                            stats = {
                                'Total_Simulaciones': len(simulaciones),
                                'Pasaron_Fase1': len(pasaron_solo_fase1),
                                'Pasaron_Fase2': len(pasaron_fase2),
                                'Quemaron_Fase1': len(quemaron_fase1),
                                'Quemaron_Fase2': len(quemaron_fase2),
                                'Prob_Pasar_Fase1': len(pasaron_solo_fase1) / len(simulaciones) * 100 if len(simulaciones) > 0 else 0,
                                'Prob_Pasar_Fase2': len(pasaron_fase2) / len(simulaciones) * 100 if len(simulaciones) > 0 else 0,
                                'Prob_Quemar_Fase1': len(quemaron_fase1) / len(simulaciones) * 100 if len(simulaciones) > 0 else 0,
                                'Prob_Quemar_Fase2': len(quemaron_fase2) / len(simulaciones) * 100 if len(simulaciones) > 0 else 0,
                            }
                            
                            # Calcular días promedio para Fase 1 (todos los que llegaron a Fase 1)
                            if len(pasaron_solo_fase1) > 0:
                                stats['Dias_Promedio_Pasar_Fase1'] = pasaron_solo_fase1['Dias_Fase1'].mean()
                                stats['Trades_Promedio_Pasar_Fase1'] = pasaron_solo_fase1['Trades_Fase1'].mean()
                            else:
                                stats['Dias_Promedio_Pasar_Fase1'] = None
                                stats['Trades_Promedio_Pasar_Fase1'] = None
                                
                            # Calcular días promedio para Fase 2 (solo los que pasaron Fase 2)
                            if len(pasaron_fase2) > 0:
                                stats['Dias_Promedio_Pasar_Fase2'] = pasaron_fase2['Dias_Fase2'].mean()
                                stats['Trades_Promedio_Pasar_Fase2'] = pasaron_fase2['Trades_Fase2'].mean()
                            else:
                                stats['Dias_Promedio_Pasar_Fase2'] = None
                                stats['Trades_Promedio_Pasar_Fase2'] = None
                                
                            # Calcular días promedio para quemar Fase 1
                            if len(quemaron_fase1) > 0:
                                stats['Dias_Promedio_Quemar_Fase1'] = quemaron_fase1['Dias_Fase1'].mean()
                                stats['Trades_Promedio_Quemar_Fase1'] = quemaron_fase1['Trades_Fase1'].mean()
                            else:
                                stats['Dias_Promedio_Quemar_Fase1'] = None
                                stats['Trades_Promedio_Quemar_Fase1'] = None
                                
                            # Calcular días promedio para quemar Fase 2
                            if len(quemaron_fase2) > 0:
                                stats['Dias_Promedio_Quemar_Fase2'] = quemaron_fase2['Dias_Fase2'].mean()
                                stats['Trades_Promedio_Quemar_Fase2'] = quemaron_fase2['Trades_Fase2'].mean()
                            else:
                                stats['Dias_Promedio_Quemar_Fase2'] = None
                                stats['Trades_Promedio_Quemar_Fase2'] = None
                            
                            return stats
                        
                        stats = calcular_estadisticas_tiempo(simulaciones)
                        
                        # Debug: Mostrar información de las simulaciones
                        st.markdown(f"""
                        <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #2196f3;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #1565c0;">🔍 Información de Simulación</h4>
                            <p style="margin: 0; color: #1565c0; font-size: 0.9rem;">
                                <strong>Total de simulaciones:</strong> {len(simulaciones)}<br>
                                <strong>Días de datos disponibles:</strong> {len(operaciones_por_dia)}<br>
                                <strong>Período de datos:</strong> {operaciones_por_dia['Fecha'].min().strftime('%Y-%m-%d')} a {operaciones_por_dia['Fecha'].max().strftime('%Y-%m-%d')}<br>
                                <strong>Simulación realista:</strong> Sin límite de días por fase
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Mostrar resultados
                        st.markdown("""
                        <div class="card">
                            <div class="card-title">⏱️ Análisis de Tiempo para Prop Firms</div>
                        """, unsafe_allow_html=True)
                        
                        # Métricas principales de tiempo y trades
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            dias_promedio_pasar = stats['Dias_Promedio_Pasar_Fase1']
                            trades_promedio_pasar = stats['Trades_Promedio_Pasar_Fase1']
                            if dias_promedio_pasar:
                                st.metric(
                                    label="Fase 1 - Pasar",
                                    value=f"{dias_promedio_pasar:.0f} días",
                                    delta=f"{trades_promedio_pasar:.0f} trades" if trades_promedio_pasar else "N/A trades"
                                )
                            else:
                                st.metric(
                                    label="Fase 1 - Pasar",
                                    value="No disponible",
                                    delta=f"{stats['Prob_Pasar_Fase1']:.1f}% probabilidad"
                                )
                        
                        with col2:
                            dias_promedio_quemar = stats['Dias_Promedio_Quemar_Fase1']
                            trades_promedio_quemar = stats['Trades_Promedio_Quemar_Fase1']
                            if dias_promedio_quemar:
                                st.metric(
                                    label="Fase 1 - Quemar",
                                    value=f"{dias_promedio_quemar:.0f} días",
                                    delta=f"{trades_promedio_quemar:.0f} trades" if trades_promedio_quemar else "N/A trades"
                                )
                            else:
                                st.metric(
                                    label="Fase 1 - Quemar",
                                    value="No disponible",
                                    delta=f"{stats['Prob_Quemar_Fase1']:.1f}% probabilidad"
                                )
                        
                        with col3:
                            dias_promedio_pasar2 = stats['Dias_Promedio_Pasar_Fase2']
                            trades_promedio_pasar2 = stats['Trades_Promedio_Pasar_Fase2']
                            if dias_promedio_pasar2:
                                st.metric(
                                    label="Fase 2 - Pasar",
                                    value=f"{dias_promedio_pasar2:.0f} días",
                                    delta=f"{trades_promedio_pasar2:.0f} trades" if trades_promedio_pasar2 else "N/A trades"
                                )
                            else:
                                st.metric(
                                    label="Fase 2 - Pasar",
                                    value="No disponible",
                                    delta=f"{stats['Prob_Pasar_Fase2']:.1f}% probabilidad"
                                )
                        
                        with col4:
                            dias_promedio_quemar2 = stats['Dias_Promedio_Quemar_Fase2']
                            trades_promedio_quemar2 = stats['Trades_Promedio_Quemar_Fase2']
                            if dias_promedio_quemar2:
                                st.metric(
                                    label="Fase 2 - Quemar",
                                    value=f"{dias_promedio_quemar2:.0f} días",
                                    delta=f"{trades_promedio_quemar2:.0f} trades" if trades_promedio_quemar2 else "N/A trades"
                                )
                            else:
                                st.metric(
                                    label="Fase 2 - Quemar",
                                    value="No disponible",
                                    delta=f"{stats['Prob_Quemar_Fase2']:.1f}% probabilidad"
                                )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Mostrar información del escalado
                        factor_escalado = riesgo_por_operacion / 2.0  # 2% es el riesgo histórico promedio
                        st.markdown(f"""
                        <div style="background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #ffc107;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #856404;">⚖️ Escalado de Riesgo</h4>
                            <p style="margin: 0; color: #856404; font-size: 0.9rem;">
                                <strong>Riesgo seleccionado:</strong> {riesgo_por_operacion}%<br>
                                <strong>Riesgo histórico promedio:</strong> 2%<br>
                                <strong>Factor de escalado:</strong> {factor_escalado:.1f}x<br>
                                <em>Las pérdidas y ganancias se multiplican por {factor_escalado:.1f} para simular tu riesgo real</em>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Resumen de probabilidades
                        st.markdown("""
                        <div class="card">
                            <div class="card-title">📊 Resumen de Probabilidades</div>
                        """, unsafe_allow_html=True)
                        
                        # Crear tabla de resumen
                        resumen_data = {
                            'Métrica': ['Pasar Fase 1', 'Quemar Fase 1', 'Pasar Fase 2', 'Quemar Fase 2'],
                            'Probabilidad': [
                                f"{stats['Prob_Pasar_Fase1']:.1f}%",
                                f"{stats['Prob_Quemar_Fase1']:.1f}%",
                                f"{stats['Prob_Pasar_Fase2']:.1f}%",
                                f"{stats['Prob_Quemar_Fase2']:.1f}%"
                            ],
                            'Días Promedio': [
                                f"{stats['Dias_Promedio_Pasar_Fase1']:.0f}" if stats['Dias_Promedio_Pasar_Fase1'] else "N/A",
                                f"{stats['Dias_Promedio_Quemar_Fase1']:.0f}" if stats['Dias_Promedio_Quemar_Fase1'] else "N/A",
                                f"{stats['Dias_Promedio_Pasar_Fase2']:.0f}" if stats['Dias_Promedio_Pasar_Fase2'] else "N/A",
                                f"{stats['Dias_Promedio_Quemar_Fase2']:.0f}" if stats['Dias_Promedio_Quemar_Fase2'] else "N/A"
                            ],
                            'Trades Promedio': [
                                f"{stats['Trades_Promedio_Pasar_Fase1']:.0f}" if stats['Trades_Promedio_Pasar_Fase1'] else "N/A",
                                f"{stats['Trades_Promedio_Quemar_Fase1']:.0f}" if stats['Trades_Promedio_Quemar_Fase1'] else "N/A",
                                f"{stats['Trades_Promedio_Pasar_Fase2']:.0f}" if stats['Trades_Promedio_Pasar_Fase2'] else "N/A",
                                f"{stats['Trades_Promedio_Quemar_Fase2']:.0f}" if stats['Trades_Promedio_Quemar_Fase2'] else "N/A"
                            ]
                        }
                        
                        df_resumen = pd.DataFrame(resumen_data)
                        st.dataframe(df_resumen, use_container_width=True, hide_index=True)
                        
                        # Análisis de sensibilidad al riesgo
                        st.markdown("""
                        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #495057;">🎯 Sensibilidad al Riesgo</h4>
                            <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">
                                Comparación de cómo cambian los resultados según el riesgo por operación:
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Crear análisis de sensibilidad
                        riesgos_test = [1, 2, 3, 4, 5]
                        sensibilidad_data = []
                        
                        for riesgo_test in riesgos_test:
                            sim_test_30d = simular_prop_firm(operaciones_por_dia, riesgo_test, drawdown_maximo_diario, drawdown_maximo_total, 30)
                            stats_test = calcular_estadisticas_tiempo(sim_test_30d)
                            
                            sensibilidad_data.append({
                                'Riesgo %': f"{riesgo_test}%",
                                'Prob. Pasar F1': f"{stats_test['Prob_Pasar_Fase1']:.1f}%",
                                'Prob. Quemar F1': f"{stats_test['Prob_Quemar_Fase1']:.1f}%",
                                'Días Prom. Pasar F1': f"{stats_test['Dias_Promedio_Pasar_Fase1']:.0f}" if stats_test['Dias_Promedio_Pasar_Fase1'] else "N/A",
                                'Trades Prom. Pasar F1': f"{stats_test['Trades_Promedio_Pasar_Fase1']:.0f}" if stats_test['Trades_Promedio_Pasar_Fase1'] else "N/A",
                                'Días Prom. Quemar F1': f"{stats_test['Dias_Promedio_Quemar_Fase1']:.0f}" if stats_test['Dias_Promedio_Quemar_Fase1'] else "N/A",
                                'Trades Prom. Quemar F1': f"{stats_test['Trades_Promedio_Quemar_Fase1']:.0f}" if stats_test['Trades_Promedio_Quemar_Fase1'] else "N/A"
                            })
                        
                        df_sensibilidad = pd.DataFrame(sensibilidad_data)
                        st.dataframe(df_sensibilidad, use_container_width=True, hide_index=True)
                        
                        # Mostrar ejemplos de simulaciones para debug
                        st.markdown("""
                        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #495057;">🔍 Ejemplos de Simulaciones (Primeras 10)</h4>
                            <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">
                                Muestra las primeras 10 simulaciones para verificar el cálculo de días:
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Mostrar ejemplos de simulaciones
                        ejemplos = simulaciones.head(10)[['Inicio', 'Dias_Fase1', 'Dias_Fase2', 'Trades_Fase1', 'Trades_Fase2', 'Trades_Totales', 'Resultado', 'Fase_Alcanzada', 'Profit_Total', 'Drawdown_Max']].copy()
                        ejemplos['Inicio'] = ejemplos['Inicio'].dt.strftime('%Y-%m-%d')
                        ejemplos['Profit_Total'] = ejemplos['Profit_Total'].round(2)
                        ejemplos['Drawdown_Max'] = ejemplos['Drawdown_Max'].round(2)
                        ejemplos.columns = ['Fecha Inicio', 'Días F1', 'Días F2', 'Trades F1', 'Trades F2', 'Trades Total', 'Resultado', 'Fase Alcanzada', 'Profit Total', 'Drawdown Max']
                        
                        st.dataframe(ejemplos, use_container_width=True, hide_index=True)
                        
                        # Análisis de operaciones por día
                        st.markdown("""
                        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #495057;">📊 Estadísticas de Trading Diario</h4>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                        """, unsafe_allow_html=True)
                        
                        ops_por_dia_promedio = operaciones_por_dia['Ops_Dia'].mean()
                        profit_por_dia_promedio = operaciones_por_dia['Profit_Dia'].mean()
                        dias_con_ops = len(operaciones_por_dia[operaciones_por_dia['Ops_Dia'] > 0])
                        total_dias = len(operaciones_por_dia)
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 0.5rem;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #6c757d;">{ops_por_dia_promedio:.1f}</div>
                            <div style="font-size: 0.9rem; color: #6c757d;">Ops por día promedio</div>
                        </div>
                        <div style="text-align: center; padding: 0.5rem;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #6c757d;">${profit_por_dia_promedio:.2f}</div>
                            <div style="font-size: 0.9rem; color: #6c757d;">Profit por día promedio</div>
                        </div>
                        <div style="text-align: center; padding: 0.5rem;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #6c757d;">{dias_con_ops}/{total_dias}</div>
                            <div style="font-size: 0.9rem; color: #6c757d;">Días con operaciones</div>
                        </div>
                        <div style="text-align: center; padding: 0.5rem;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #6c757d;">{(dias_con_ops/total_dias*100):.1f}%</div>
                            <div style="font-size: 0.9rem; color: #6c757d;">Frecuencia de trading</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Análisis de Prop Firm
                        st.markdown("""
                        <div class="card">
                            <div class="card-title">🏢 Análisis de Prop Firm</div>
                        """, unsafe_allow_html=True)
                        
                        # Configuración seleccionada
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #495057;">⚙️ Configuración Seleccionada</h4>
                            <p style="margin: 0; color: #6c757d;">
                                <strong>Tamaño de cuenta:</strong> ${tamaño_cuenta:,}<br>
                                <strong>Riesgo por operación:</strong> {riesgo_por_operacion}% (${riesgo_por_op_dinero:.2f})<br>
                                <strong>Drawdown máximo diario:</strong> {drawdown_maximo_diario}% (si pierdes más en 1 día, suspendes)<br>
                                <strong>Drawdown máximo total:</strong> {drawdown_maximo_total}% (límite global para toda la evaluación)<br>
                                <strong>Fase 1:</strong> Ganancia objetivo {porcentaje_fase1}% (${tamaño_cuenta * porcentaje_fase1 / 100:.2f})<br>
                                <strong>Fase 2:</strong> Ganancia objetivo {porcentaje_fase2}% (${tamaño_cuenta * porcentaje_fase2 / 100:.2f})
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Probabilidades de pasar
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Fase 1
                            color_fase1 = "green" if prob_pasar_fase1 >= 70 else "orange" if prob_pasar_fase1 >= 50 else "red"
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; text-align: center; border-left: 4px solid {color_fase1};">
                                <h3 style="margin: 0 0 0.5rem 0; color: {color_fase1};">Fase 1</h3>
                                <div style="font-size: 2rem; font-weight: bold; color: {color_fase1}; margin-bottom: 0.5rem;">
                                    {prob_pasar_fase1:.1f}%
                                </div>
                                <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">
                                    Probabilidad de pasar<br>
                                    <small>Objetivo: {porcentaje_fase1}%</small>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Fase 2
                            color_fase2 = "green" if prob_pasar_fase2 >= 70 else "orange" if prob_pasar_fase2 >= 50 else "red"
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; text-align: center; border-left: 4px solid {color_fase2};">
                                <h3 style="margin: 0 0 0.5rem 0; color: {color_fase2};">Fase 2</h3>
                                <div style="font-size: 2rem; font-weight: bold; color: {color_fase2}; margin-bottom: 0.5rem;">
                                    {prob_pasar_fase2:.1f}%
                                </div>
                                <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">
                                    Probabilidad de pasar<br>
                                    <small>Objetivo: {porcentaje_fase2}%</small>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Recomendaciones basadas en tiempo
                        st.markdown("""
                        <div style="background-color: #e8f5e8; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #28a745;">💡 Recomendaciones Prácticas</h4>
                        """, unsafe_allow_html=True)
                        
                        # Calcular recomendaciones basadas en tiempo
                        mejor_prob_pasar = stats['Prob_Pasar_Fase1']
                        mejor_prob_quemar = stats['Prob_Quemar_Fase1']
                        
                        if mejor_prob_pasar >= 70:
                            st.markdown("""
                            <p style="margin: 0; color: #28a745;">
                                ✅ <strong>Estrategia Sólida:</strong> Con estos parámetros tienes alta probabilidad de pasar las pruebas. 
                                Tu estrategia es adecuada para prop firms.
                            </p>
                            """, unsafe_allow_html=True)
                        elif mejor_prob_pasar >= 50:
                            st.markdown("""
                            <p style="margin: 0; color: #ffc107;">
                                ⚠️ <strong>Estrategia Moderada:</strong> Probabilidad media de éxito. 
                                Considera reducir el riesgo por operación para mejorar tus chances.
                            </p>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <p style="margin: 0; color: #dc3545;">
                                ❌ <strong>Estrategia de Alto Riesgo:</strong> Baja probabilidad de pasar. 
                                Recomendamos reducir significativamente el riesgo por operación.
                            </p>
                            """, unsafe_allow_html=True)
                        
                        # Recomendación específica de tiempo
                        if stats['Dias_Promedio_Pasar_Fase1'] and stats['Dias_Promedio_Pasar_Fase1'] < 15:
                            st.markdown("""
                            <p style="margin: 0.5rem 0 0 0; color: #28a745;">
                                🚀 <strong>Ventaja Temporal:</strong> Puedes pasar la Fase 1 en menos de 15 días en promedio.
                            </p>
                            """, unsafe_allow_html=True)
                        elif stats['Dias_Promedio_Quemar_Fase1'] and stats['Dias_Promedio_Quemar_Fase1'] < 10:
                            st.markdown("""
                            <p style="margin: 0.5rem 0 0 0; color: #dc3545;">
                                ⚡ <strong>Riesgo Rápido:</strong> Podrías quemar la cuenta en menos de 10 días. 
                                Reduce el riesgo por operación.
                            </p>
                            """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Tabla de operaciones problemáticas basada en drawdown
                        # Calcular operaciones que excederían el drawdown diario
                        limite_drawdown_diario_dinero = tamaño_cuenta * (drawdown_maximo_diario / 100)
                        ops_exceden_drawdown_diario = len(df_csv[df_csv['Profit/Loss'] < -limite_drawdown_diario_dinero])
                        
                        if ops_exceden_drawdown_diario > 0:
                            st.markdown("""
                            <div class="card">
                                <div class="card-title">⚠️ Operaciones Problemáticas</div>
                            """, unsafe_allow_html=True)
                            
                            ops_problematicas = df_csv[df_csv['Profit/Loss'] < -limite_drawdown_diario_dinero].copy()
                            if not ops_problematicas.empty:
                                ops_problematicas['%_de_cuenta'] = (ops_problematicas['Profit/Loss'] / tamaño_cuenta * 100).round(2)
                                ops_problematicas['Excede_Drawdown_Diario'] = ops_problematicas['Profit/Loss'] < -limite_drawdown_diario_dinero
                                
                                st.dataframe(
                                    ops_problematicas[['Ticket', 'Symbol', 'Type', 'Open time', 'Close time', 'Profit/Loss', '%_de_cuenta', 'Excede_Drawdown_Diario']],
                                    use_container_width=True,
                                    hide_index=True
                                )
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Gráfico de drawdown
                        st.markdown("""
                        <div class="card">
                            <div class="card-title">📈 Análisis de Drawdown</div>
                        """, unsafe_allow_html=True)
                        
                        fig_drawdown = px.line(
                            df_csv_sorted,
                            x='Close time',
                            y='Drawdown',
                            title='Evolución del Drawdown',
                            labels={'Drawdown': 'Drawdown (%)', 'Close time': 'Fecha'}
                        )
                        
                        # Añadir líneas de referencia para las fases
                        fig_drawdown.add_hline(y=10, line_dash="dash", line_color="red", 
                                             annotation_text="Límite Fase 1 (10%)")
                        fig_drawdown.add_hline(y=5, line_dash="dash", line_color="orange", 
                                             annotation_text="Límite Fase 2 (5%)")
                        
                        fig_drawdown.update_layout(
                            height=400,
                            template="plotly_white",
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(color='#495057')
                        )
                        
                        st.plotly_chart(fig_drawdown, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"❌ Error al procesar el archivo CSV: {str(e)}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #6c757d;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
                <h3 style="margin-bottom: 1rem;">Sube tu archivo CSV</h3>
                <p style="margin-bottom: 1rem;">
                    Selecciona un archivo CSV con el formato de operaciones de trading para analizar 
                    tu probabilidad de pasar las pruebas de prop firm.
                </p>
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                    <h4 style="margin: 0 0 0.5rem 0;">📋 Formato esperado:</h4>
                    <p style="margin: 0; font-size: 0.9rem;">
                        Ticket; Symbol; Type; Open time; Open price; Size; Close time; Close price; 
                        Profit/Loss; Balance; Sample type; Close type; MAE ($); MFE ($); Time in trade; Comment
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# Array de promociones de Prop Firms
promociones = [
    {
        "id": "ftmo",
        "nombre": "FTMO Challenge",
        "logo": "https://tse2.mm.bing.net/th/id/OIP.YcCmzHSPPjrc2v3JTGTKNAHaB4?r=0&rs=1&pid=ImgDetMain&o=7&rm=3",
        "badge": "🔥 HOT",
        "descuento": "10% DESCUENTO",
        "codigo": "FTMO10",
        "descripcion": "Obtén hasta $400,000 en capital de trading con el 10% de descuento en tu primer challenge. La prop firm más confiable del mercado.",
        "url": "https://ftmo.com",
        "boton_texto": "Ir a FTMO",
        "tiene_timer": True,
        "timer": {
            "dias": 15,
            "horas": 12,
            "minutos": 30
        }
    },
    {
        "id": "ttp",
        "nombre": "The Trading Pit",
        "logo": "https://www.thetradingpit.com/assets/global/logo-dark-4.svg",
        "badge": "⭐ NUEVO",
        "descuento": "15% DESCUENTO",
        "codigo": "TTP15",
        "descripcion": "Accede a hasta $200,000 en capital con condiciones de trading flexibles y sin límite de tiempo. Perfecto para traders principiantes.",
        "url": "https://thetradingpit.com",
        "boton_texto": "Ir a The Trading Pit",
        "tiene_timer": False
    }
]

def generar_promo_html(promo):
    """Genera el HTML para una promoción individual"""
    timer_html = ""
    if promo.get("tiene_timer", False):
        timer = promo["timer"]
        timer_html = f'<div class="countdown-timer" id="{promo["id"]}-timer"><div class="timer-label">⏰ Oferta termina en:</div><div class="timer-display"><span id="{promo["id"]}-days">{timer["dias"]}</span>d<span id="{promo["id"]}-hours">{timer["horas"]}</span>h<span id="{promo["id"]}-minutes">{timer["minutos"]}</span>m</div></div>'
    
    return f'<div class="promo-card-full"><div class="promo-image-section"><img src="{promo["logo"]}" alt="{promo["nombre"]}" class="promo-main-logo"></div><div class="promo-content-section"><div class="promo-info"><div class="promo-discount">{promo["descuento"]}</div><div class="promo-code"><span>Código: </span><code id="{promo["id"]}-code">{promo["codigo"]}</code><button class="copy-btn" onclick="copyCode(\'{promo["id"]}-code\')">📋</button></div></div><div class="promo-actions">{timer_html}<a href="{promo["url"]}" target="_blank" class="promo-link"></a></div></div></div>'

# Tab de Prop Firms
with tab2:

    
    # Generar promociones dinámicamente desde el array
    for promo in promociones:
        st.markdown(generar_promo_html(promo), unsafe_allow_html=True)

# Tab de Configuración
with tab3:
    st.markdown("""
    <div class="card">
        <div class="card-title">⚙️ Configuración</div>
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;" class="construction-icon">⚡</div>
            <h2 style="color: #6c757d; margin-bottom: 1rem;">¡En Desarrollo!</h2>
            <p style="color: #6c757d; font-size: 1.1rem; margin-bottom: 2rem;">
                Próximamente disponible.
            </p>
            <div style="display: flex; justify-content: center; align-items: center; gap: 0.5rem; color: #6c757d;">
                <div class="loading-spinner">
                    <div></div>
                </div>
                <span style="margin-left: 1rem;">Desarrollando...</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Cerrar contenedor principal
st.markdown('</div>', unsafe_allow_html=True)