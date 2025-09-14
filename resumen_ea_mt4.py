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
        padding: 1rem !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
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
    .stTabs [data-baseweb="tab-list"] {
        margin-bottom: 2rem !important;
        display: flex !important;
        gap: 0.5rem !important;
        padding: 0.5rem !important;
        overflow: visible !important;
        align-items: center;
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
</style>
""", unsafe_allow_html=True)
# Estado de la sesión para manejar tabs
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'analisis'

# Barra de navegación superior con tabs de Streamlit
tab1, tab2, tab3 = st.tabs(["📊 Análisis", "🚀 Proyectos", "⚙️ Configuración"])

# Contenedor principal
st.markdown('<div class="main-container">', unsafe_allow_html=True)


# Panel central con contenido dinámico según el tab
with tab1:
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
                
                # Tarjeta de operaciones detalladas
                st.markdown("""
                <div class="card">
                    <div class="card-title">Operaciones por EA</div>
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

# Tab de Proyectos
with tab2:
    st.markdown("""
    <div class="card">
        <div class="card-title">🚀 Proyectos</div>
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;" class="construction-icon">🔨</div>
            <h2 style="color: #6c757d; margin-bottom: 1rem;">¡En Construcción!</h2>
            <p style="color: #6c757d; font-size: 1.1rem; margin-bottom: 2rem;">
                Próximamente disponible.
            </p>
            <div style="display: flex; justify-content: center; align-items: center; gap: 0.5rem; color: #6c757d;">
                <div class="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <span style="margin-left: 1rem;">Trabajando en ello...</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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