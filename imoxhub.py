import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import base64
import requests
import json
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import io
import os

# Configurar tema claro por defecto para Plotly
pio.templates.default = "plotly_white"

# Configuración de la URL del backend
# Por defecto usa localhost, pero puedes cambiar con variable de entorno API_URL
# Para producción: export API_URL=https://imoxhub-backend.onrender.com
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")

# Asegurar que la URL tenga esquema (https://) y dominio completo
if not API_BASE_URL.startswith(("http://", "https://")):
    # Si la URL viene sin esquema desde la variable de entorno, agregarlo
    if "imoxhub-backend" in API_BASE_URL and not API_BASE_URL.endswith(".onrender.com"):
        # Si es imoxhub-backend pero le falta el dominio completo
        API_BASE_URL = f"https://imoxhub-backend.onrender.com"
    else:
        API_BASE_URL = f"https://{API_BASE_URL}"

# AÑADIR ESTAS FUNCIONES AQUÍ (después de la línea 17)
def get_medal_emoji(position):
    """Obtener emoji de medalla según posición"""
    medals = {
        1: '🥇', 2: '🥈', 3: '🥉',
        4: '➃', 5: '➄', 6: '➅', 7: '➆', 8: '➇', 9: '➈', 10: '➉'
    }
    return medals.get(position, f"{position}.")

def create_custom_instagram_story(ranking_data, config=None):
    """Crea una imagen personalizable para Instagram Stories"""
    
    if config is None:
        config = {}
    
    width = 1080
    height = 1920
    background_color = '#1a1a1a'
    title_color = '#ffffff'
    name_color = '#ffffff'
    score_color = '#00ff88'
    medal_color = '#ffd700'
    accent_color = '#ff6b6b'
    title = config.get('title', '🏆 RANKING GLOBAL')
    subtitle = config.get('subtitle', 'Top 10 Mejores Jugadores')
    watermark = config.get('watermark', '@tu_marca')
    date_text = datetime.now().strftime('%d/%m/%Y')
    
    # Crear imagen base
    img = Image.new('RGB', (width, height), color=background_color)
    draw = ImageDraw.Draw(img)
    
    # Cargar fuentes
    try:
        title_font = ImageFont.truetype("Arial Bold", 80)
        subtitle_font = ImageFont.truetype("Arial", 40)
        name_font = ImageFont.truetype("Arial", 50)
        score_font = ImageFont.truetype("Arial Bold", 45)
        watermark_font = ImageFont.truetype("Arial", 30)
        date_font = ImageFont.truetype("Arial", 25)
    except:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        name_font = ImageFont.load_default()
        score_font = ImageFont.load_default()
        watermark_font = ImageFont.load_default()
        date_font = ImageFont.load_default()
    
    # Título principal
    draw.text((width//2, 200), title, fill=title_color, font=title_font, anchor='mm')
    
    # Subtítulo
    draw.text((width//2, 280), subtitle, fill=accent_color, font=subtitle_font, anchor='mm')
    
    # Fecha
    draw.text((width//2, 1850), date_text, fill='#888888', font=date_font, anchor='mm')
    
    # Ranking
    for i, (position, name, score) in enumerate(ranking_data[:10]):
        y_pos = 400 + i * 140
        
        # Medalla según posición
        medal = get_medal_emoji(position)
        
        # Fondo para cada fila
        row_rect = [50, y_pos-50, width-50, y_pos+50]
        draw.rectangle(row_rect, fill='#2a2a2a', outline='#333333', width=2)
        
        # Medalla
        draw.text((100, y_pos), medal, fill=medal_color, font=name_font, anchor='mm')
        
        # Nombre
        draw.text((200, y_pos), name, fill=name_color, font=name_font, anchor='lm')
        
        # Score
        draw.text((width - 100, y_pos), score, fill=score_color, font=score_font, anchor='rm')
        
        # Línea decorativa
        line_y = y_pos + 30
        draw.line([(200, line_y), (width-150, line_y)], fill='#333333', width=1)
    
    # Marca de agua
    draw.text((width//2, 1800), watermark, fill='#666666', font=watermark_font, anchor='mm')
    
    return img

# Función para codificar imágenes en base64
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Funciones para calcular ranking de riesgo-beneficio
def limpiar_nombre_ea(nombre_ea):
    """
    Limpia el nombre de la EA eliminando números adicionales y caracteres especiales.
    Ejemplo: 'NQ.H4 20397968 [sl]' -> 'NQ.H4'
    """
    if not nombre_ea or nombre_ea.strip() == "":
        return ""
    
    # Eliminar contenido entre corchetes primero
    nombre_limpio = nombre_ea.split('[')[0].strip()
    
    # Dividir por espacios y mantener solo las partes que contienen letras
    partes = nombre_limpio.split()
    partes_validas = []
    
    for parte in partes:
        # Si la parte contiene al menos una letra, la mantenemos
        if any(c.isalpha() for c in parte):
            partes_validas.append(parte)
        # Si es solo números, la ignoramos (como "20397968")
        elif parte.isdigit():
            continue
        # Si contiene puntos y letras (como "NQ.H4"), la mantenemos
        elif '.' in parte and any(c.isalpha() for c in parte):
            partes_validas.append(parte)
    
    return ' '.join(partes_validas)

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

# =============================================================================
# FUNCIONES PARA CONECTAR CON LA API DEL BACKEND
# =============================================================================

def get_headers():
    """Obtiene los headers con autenticación"""
    headers = {}
    if st.session_state.get('token'):
        headers['Authorization'] = f"Bearer {st.session_state.token}"
    return headers

@st.cache_data(ttl=60, show_spinner=False)  # Cache por 60 segundos
def get_payouts_from_api():
    """
    Obtiene todos los payouts desde la API del backend
    """
    try:
        headers = get_headers()
        response = requests.get(f"{API_BASE_URL}/payouts/", headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except requests.exceptions.Timeout:
            return []
    except requests.exceptions.ConnectionError:
        return []
    except Exception as e:
        return []

@st.cache_data(ttl=60, show_spinner=False)  # Cache por 60 segundos
def get_users_from_api():
    """
    Obtiene todos los usuarios desde la API del backend
    """
    try:
        headers = get_headers()
        response = requests.get(f"{API_BASE_URL}/users/", headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except requests.exceptions.Timeout:
            return []
    except requests.exceptions.ConnectionError:
        return []
    except Exception as e:
        return []

def create_user_api(nick: str, name: str, correo: str, rank: str = "silver"):
    """
    Crea un nuevo usuario en el backend
    """
    try:
        user_data = {
            "nick": nick,
            "name": name,
            "correo": correo,
            "rank": rank
        }
        headers = get_headers()
        response = requests.post(f"{API_BASE_URL}/users/", json=user_data, headers=headers, timeout=10)
        if response.status_code == 201:
            return True, "Usuario creado exitosamente"
        else:
            try:
                error_detail = response.json().get("detail", f"Error {response.status_code}")
            except:
                error_detail = f"Error {response.status_code}: {response.text}"
            return False, error_detail
    except requests.exceptions.ConnectionError:
        return False, f"No se puede conectar con el backend. Asegúrate de que esté ejecutándose en {API_BASE_URL}"
    except Exception as e:
        return False, f"Error inesperado: {str(e)}"

def create_monthly_payout_ranking(payouts_data, users_data, selected_month_offset=0, user_rank=None):
    """
    Crea un ranking mensual de payouts agrupado por usuario y opcionalmente por rango
    
    Args:
        payouts_data: Lista de payouts desde la API
        users_data: Lista de usuarios desde la API
        selected_month_offset: Offset del mes (0 = actual, -1 = anterior, etc.)
        user_rank: Filtro por rango de usuario ('gold', 'silver', 'admin') o None para todos
    
    Returns:
        DataFrame con el ranking mensual
    """
    # Validar datos de entrada
    if not payouts_data or not isinstance(payouts_data, list):
        return pd.DataFrame()
    
    if not users_data or not isinstance(users_data, list):
        return pd.DataFrame()
    
    # Convertir a DataFrame
    try:
        df_payouts = pd.DataFrame(payouts_data)
        if df_payouts.empty:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    
    # Verificar que las columnas necesarias existan
    required_payout_cols = ['nick', 'payout', 'fecha_payout']
    missing_cols = [col for col in required_payout_cols if col not in df_payouts.columns]
    if missing_cols:
        return pd.DataFrame()
    
    # Convertir fechas
    df_payouts['fecha_payout'] = pd.to_datetime(df_payouts['fecha_payout'], errors='coerce')
    if 'fecha_creacion' in df_payouts.columns:
        df_payouts['fecha_creacion'] = pd.to_datetime(df_payouts['fecha_creacion'], errors='coerce')
    
    # Calcular el mes objetivo
    target_date = datetime.now() + timedelta(days=30 * selected_month_offset)
    target_year = target_date.year
    target_month = target_date.month
    
    # Filtrar payouts del mes objetivo (solo los que tienen fecha_payout)
    df_payouts_filtered = df_payouts[
        (df_payouts['fecha_payout'].dt.year == target_year) &
        (df_payouts['fecha_payout'].dt.month == target_month)
    ].copy()
    
    if df_payouts_filtered.empty:
        return pd.DataFrame()
    
    # Crear diccionario de usuarios para obtener nombres y rangos
    try:
        users_dict = {}
        for user in users_data:
            if isinstance(user, dict) and 'nick' in user:
                users_dict[user['nick']] = {
                    'name': user.get('name', user.get('nick', '')),
                    'rank': user.get('rank', 'silver')
                }
    except Exception:
        users_dict = {}
    
    # Filtrar por rango de usuario si se especifica
    if user_rank:
        # Obtener nicks de usuarios del rango especificado
        target_nicks = [nick for nick, data in users_dict.items() if data['rank'] == user_rank]
        df_payouts_filtered = df_payouts_filtered[df_payouts_filtered['nick'].isin(target_nicks)]
        
        if df_payouts_filtered.empty:
            return pd.DataFrame()
    
    # Convertir payout a float
    df_payouts_filtered['payout_amount'] = df_payouts_filtered['payout'].astype(float)
    
    # Agrupar por nick y sumar payouts
    monthly_totals = df_payouts_filtered.groupby('nick').agg({
        'payout_amount': 'sum',
        'payout': 'count'
    }).reset_index()
    
    monthly_totals.columns = ['nick', 'total_payout', 'num_payouts']
    
    # Agregar nombres y rangos de usuarios
    monthly_totals['name'] = monthly_totals['nick'].map(lambda x: users_dict[x]['name'])
    monthly_totals['rank'] = monthly_totals['nick'].map(lambda x: users_dict[x]['rank'])
    monthly_totals['name'] = monthly_totals['name'].fillna(monthly_totals['nick'])
    
    # Ordenar por total de payout (descendente)
    monthly_totals = monthly_totals.sort_values('total_payout', ascending=False)
    
    # Agregar posición en el ranking
    monthly_totals['position'] = range(1, len(monthly_totals) + 1)
    
    # Formatear montos
    monthly_totals['total_payout_formatted'] = monthly_totals['total_payout'].apply(lambda x: f"${x:,.2f}")
    
    return monthly_totals

def get_month_name(month_offset):
    """
    Obtiene el nombre del mes basado en el offset
    """
    target_date = datetime.now() + timedelta(days=30 * month_offset)
    month_names = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]
    return month_names[target_date.month - 1]

# =============================================================================
# FUNCIONES PARA ANÁLISIS SECUENCIAL DE PROP FIRM
# =============================================================================

def simulate_sequential_prop_firm_challenges(df_trades, config):
    """
    Simula múltiples challenges de prop firm de forma secuencial
    usando los trades uno tras otro sin repetir.
    
    Args:
        df_trades: DataFrame con operaciones históricas ordenadas
        config: Dict con configuración del challenge
    
    Returns:
        Dict con estadísticas de todos los challenges simulados
    """
    
    # Preparar datos
    df_processed = prepare_trading_data(df_trades, config)
    
    # Variables de seguimiento
    challenges = []
    current_trade_index = 0
    challenge_number = 1
    
    # Ejecutar challenges secuenciales hasta agotar los trades
    while current_trade_index < len(df_processed) - 10:  # Mínimo 10 trades por challenge
        
        # Ejecutar un challenge empezando desde current_trade_index
        challenge_result, trades_used = run_sequential_challenge(
            df_processed, current_trade_index, config, challenge_number
        )
        
        challenges.append(challenge_result)
        
        # Avanzar al siguiente trade para el próximo challenge
        current_trade_index += trades_used
        challenge_number += 1
        
        # Límite de seguridad para evitar bucles infinitos
        if len(challenges) >= 100:
            break
    
    return analyze_sequential_results(challenges, config)

def prepare_trading_data(df, config):
    """Prepara y escala los datos de trading según el riesgo configurado"""
    
    df = df.copy()
    
    # Convertir y limpiar datos
    df['Open time'] = pd.to_datetime(df['Open time'])
    df['Close time'] = pd.to_datetime(df['Close time'])
    df['profit_loss'] = pd.to_numeric(df['Profit/Loss'])
    df['date'] = df['Close time'].dt.date
    
    # Ordenar por tiempo de cierre
    df = df.sort_values('Close time').reset_index(drop=True)
    
    # Calcular factor de escalado
    historical_risk = 2.0  # Riesgo promedio asumido en datos históricos
    scaling_factor = config['risk_per_trade'] / historical_risk
    
    # Escalar profits según el riesgo seleccionado
    df['scaled_profit'] = df['profit_loss'] * scaling_factor
    df['scaled_profit_pct'] = (df['scaled_profit'] / config['account_size']) * 100
    
    return df

def run_sequential_challenge(df_trades, start_index, config, challenge_num):
    """
    Ejecuta un challenge individual empezando desde start_index
    
    Returns:
        (challenge_result, number_of_trades_used)
    """
    
    account_size = config['account_size']
    phase1_target = config['phase1_target']
    phase2_target = config['phase2_target']
    daily_dd_limit = config['daily_dd_limit']
    total_dd_limit = config['total_dd_limit']
    
    # Estado inicial del challenge
    balance = account_size
    high_water_mark = account_size
    current_phase = 1
    total_profit_pct = 0
    
    # Objetivos de cada fase
    phase1_goal = phase1_target  # % de ganancia para Fase 1
    phase2_goal = phase2_target  # % adicional para Fase 2
    
    # Contadores
    trades_used = 0
    days_in_phase1 = 0
    days_in_phase2 = 0
    
    # Estado del challenge
    status = "In Progress"
    failure_reason = None
    
    # Procesar trades secuencialmente
    current_date = None
    daily_profit_pct = 0
    daily_trades = 0
    
    for i in range(start_index, len(df_trades)):
        trade = df_trades.iloc[i]
        trade_date = trade['date']
        trade_profit_pct = trade['scaled_profit_pct']
        
        # Detectar cambio de día
        if current_date != trade_date:
            # Si cambió el día, verificar límites del día anterior
            if current_date is not None and daily_profit_pct < -daily_dd_limit:
                status = f"SUSPENDIDO - DD Diario Fase {current_phase}"
                failure_reason = "Daily Drawdown Exceeded"
                break
            
            # Resetear para el nuevo día
            current_date = trade_date
            daily_profit_pct = 0
            daily_trades = 0
            
            # Contar días por fase
            if current_phase == 1:
                days_in_phase1 += 1
            else:
                days_in_phase2 += 1
        
        # Ejecutar el trade
        trades_used += 1
        daily_trades += 1
        daily_profit_pct += trade_profit_pct
        total_profit_pct += trade_profit_pct
        
        # Actualizar balance y high water mark
        balance += trade['scaled_profit']
        high_water_mark = max(high_water_mark, balance)
        
        # Verificar drawdown total
        current_dd_pct = (high_water_mark - balance) / high_water_mark * 100
        if current_dd_pct >= total_dd_limit:
            status = f"SUSPENDIDO - DD Total Fase {current_phase}"
            failure_reason = "Total Drawdown Exceeded"
            break
        
        # Verificar drawdown diario DESPUÉS de cada trade
        if daily_profit_pct < -daily_dd_limit:
            status = f"SUSPENDIDO - DD Diario Fase {current_phase}"
            failure_reason = "Daily Drawdown Exceeded"
            break
        
        # Verificar objetivos de fase
        if current_phase == 1 and total_profit_pct >= phase1_goal:
            # Pasó Fase 1, avanza a Fase 2
            current_phase = 2
            
        elif current_phase == 2 and total_profit_pct >= phase1_goal + phase2_goal:
            # Completó ambas fases
            status = "Passed Both Phases"
            break
    
    # Verificar el último día si no se completó el challenge
    if status == "In Progress" and current_date is not None and daily_profit_pct < -daily_dd_limit:
        status = f"SUSPENDIDO - DD Diario Fase {current_phase}"
        failure_reason = "Daily Drawdown Exceeded"
    
    # Si llegó al final de los datos sin completar
    if status == "In Progress":
        if current_phase == 1:
            status = "Incomplete Phase 1"
        else:
            status = "Incomplete Phase 2"
        failure_reason = "Insufficient Data"
    
    return {
        'challenge_number': challenge_num,
        'start_trade_index': start_index,
        'status': status,
        'failure_reason': failure_reason,
        'phase_reached': current_phase,
        'total_trades_used': trades_used,
        'days_phase1': days_in_phase1,
        'days_phase2': days_in_phase2,
        'total_days': days_in_phase1 + days_in_phase2,
        'final_profit_pct': total_profit_pct,
        'final_balance': balance,
        'max_drawdown_pct': current_dd_pct
    }, trades_used

def analyze_sequential_results(challenges, config):
    """Analiza los resultados de múltiples challenges secuenciales"""
    
    if not challenges:
        return None
    
    total_challenges = len(challenges)
    
    # Categorizar resultados
    passed_both = [c for c in challenges if c['status'] == "Passed Both Phases"]
    failed_phase1 = [c for c in challenges if "SUSPENDIDO" in c['status'] and "Fase 1" in c['status']]
    failed_phase2 = [c for c in challenges if "SUSPENDIDO" in c['status'] and "Fase 2" in c['status']]
    incomplete = [c for c in challenges if "Incomplete" in c['status']]
    
    # Separar por tipo de fallo
    failed_daily_dd = [c for c in challenges if c.get('failure_reason') == "Daily Drawdown Exceeded"]
    failed_total_dd = [c for c in challenges if c.get('failure_reason') == "Total Drawdown Exceeded"]
    
    # Calcular probabilidades
    prob_pass_both = len(passed_both) / total_challenges * 100
    prob_reach_phase2 = len([c for c in challenges if c['phase_reached'] >= 2]) / total_challenges * 100
    prob_fail_daily_dd = len(failed_daily_dd) / total_challenges * 100
    prob_fail_total_dd = len(failed_total_dd) / total_challenges * 100
    
    # Calcular métricas promedio
    def safe_avg(data_list, field):
        values = [item[field] for item in data_list if item[field] > 0]
        return np.mean(values) if values else 0
    
    avg_trades_to_pass = safe_avg(passed_both, 'total_trades_used')
    avg_days_to_pass = safe_avg(passed_both, 'total_days')
    avg_trades_to_fail_p1 = safe_avg(failed_phase1, 'total_trades_used')
    avg_days_to_fail_p1 = safe_avg(failed_phase1, 'total_days')
    
    # Análisis de tiempo por fase
    phase1_completers = [c for c in challenges if c['phase_reached'] >= 2]
    avg_days_phase1 = safe_avg(phase1_completers, 'days_phase1')
    avg_trades_phase1 = safe_avg(phase1_completers, 'total_trades_used')  # Aproximado
    
    return {
        'summary': {
            'total_challenges': total_challenges,
            'total_trades_analyzed': sum(c['total_trades_used'] for c in challenges),
            'data_utilization_pct': (sum(c['total_trades_used'] for c in challenges) / 1000) * 100  # Asumiendo 1000 trades
        },
        'probabilities': {
            'pass_both_phases': prob_pass_both,
            'reach_phase2': prob_reach_phase2,
            'fail_daily_drawdown': prob_fail_daily_dd,
            'fail_total_drawdown': prob_fail_total_dd,
            'incomplete_challenges': len(incomplete) / total_challenges * 100
        },
        'average_metrics': {
            'trades_to_complete': avg_trades_to_pass,
            'days_to_complete': avg_days_to_pass,
            'trades_to_fail_phase1': avg_trades_to_fail_p1,
            'days_to_fail_phase1': avg_days_to_fail_p1,
            'days_for_phase1': avg_days_phase1
        },
        'risk_analysis': {
            'scaling_factor': config['risk_per_trade'] / 2.0,
            'daily_dd_violations': len(failed_daily_dd),
            'total_dd_violations': len(failed_total_dd),
            'success_rate': prob_pass_both,
            'risk_level': 'Low' if prob_pass_both >= 70 else 'Medium' if prob_pass_both >= 50 else 'High'
        },
        'detailed_results': challenges[:20]  # Primeros 20 challenges para inspección
    }

def create_detailed_visualizations(results):
    """Crea visualizaciones detalladas de los resultados"""
    
    # Gráfico de distribución de resultados
    fig_distribution = go.Figure()
    
    categories = ['Pasa Ambas Fases', 'Llega a Fase 2', 'Falla DD Diario', 'Falla DD Total']
    values = [
        results['probabilities']['pass_both_phases'],
        results['probabilities']['reach_phase2'],
        results['probabilities']['fail_daily_drawdown'],
        results['probabilities']['fail_total_drawdown']
    ]
    colors = ['#10B981', '#3B82F6', '#EF4444', '#F97316']
    
    fig_distribution.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition='auto',
    ))
    
    fig_distribution.update_layout(
        title="Distribución de Resultados - Simulación Secuencial",
        yaxis_title="Probabilidad (%)",
        template="plotly_white",
        height=400
    )
    
    # Gráfico de evolución de challenges (primeros 20)
    detailed = results['detailed_results']
    if detailed:
        fig_evolution = go.Figure()
        
        challenge_nums = [c['challenge_number'] for c in detailed]
        trades_used = [c['total_trades_used'] for c in detailed]
        success = [1 if "Passed" in c['status'] else 0 for c in detailed]
        
        fig_evolution.add_trace(go.Scatter(
            x=challenge_nums,
            y=trades_used,
            mode='markers+lines',
            name='Trades Usados',
            yaxis='y',
            marker=dict(
                size=10,
                color=[colors[1] if s else colors[2] for s in success]
            )
        ))
        
        fig_evolution.update_layout(
            title="Evolución de Challenges (Primeros 20)",
            xaxis_title="Número de Challenge",
            yaxis_title="Trades Utilizados",
            template="plotly_white",
            height=400
        )
    else:
        fig_evolution = None
    
    return fig_distribution, fig_evolution

def run_sequential_prop_firm_analysis(df_csv, config_dict):
    """
    Función principal para ejecutar análisis secuencial
    
    Args:
        df_csv: DataFrame con datos del CSV
        config_dict: Configuración del challenge
    
    Returns:
        results, figures
    """
    
    # Ejecutar simulación secuencial
    results = simulate_sequential_prop_firm_challenges(df_csv, config_dict)
    
    # Crear visualizaciones
    fig_dist, fig_evol = create_detailed_visualizations(results)
    
    return results, (fig_dist, fig_evol)

def compare_risk_levels_sequential(df_csv, base_config, risk_levels=[1, 2, 3, 4, 5]):
    """Compara diferentes niveles de riesgo usando simulación secuencial"""
    
    comparison_results = []
    
    for risk in risk_levels:
        config = base_config.copy()
        config['risk_per_trade'] = risk
        
        results = simulate_sequential_prop_firm_challenges(df_csv, config)
        
        comparison_results.append({
            'risk_percent': risk,
            'success_rate': results['probabilities']['pass_both_phases'],
            'avg_trades_to_complete': results['average_metrics']['trades_to_complete'],
            'avg_days_to_complete': results['average_metrics']['days_to_complete'],
            'daily_dd_failure_rate': results['probabilities']['fail_daily_drawdown'],
            'total_challenges': results['summary']['total_challenges']
        })
    
    return pd.DataFrame(comparison_results)

def run_detailed_challenge_tracking(df_trades, start_index, config, challenge_num, max_trades=None):
    """
    Ejecuta un challenge individual con seguimiento detallado trade por trade
    
    Returns:
        detailed_evolution: Lista con la evolución completa del challenge
    """
    
    account_size = config['account_size']
    phase1_target = config['phase1_target']
    phase2_target = config['phase2_target']
    daily_dd_limit = config['daily_dd_limit']
    total_dd_limit = config['total_dd_limit']
    
    # Calcular el riesgo por operación en dólares
    risk_per_trade_dollars = (config['risk_per_trade'] / 100) * account_size
    
    # Calcular el factor de escalado basado en la pérdida máxima del CSV
    max_loss_csv = abs(df_trades['profit_loss'].min())  # Pérdida máxima en el CSV
    scaling_factor = risk_per_trade_dollars / max_loss_csv if max_loss_csv > 0 else 1
    
    # Estado inicial
    balance = account_size
    high_water_mark = account_size
    current_phase = 1
    total_profit_pct = 0
    
    # Objetivos
    phase1_goal = phase1_target
    phase2_goal = phase2_target
    
    # Seguimiento detallado
    evolution = []
    trades_used = 0
    current_date = None
    daily_profit_pct = 0
    daily_trades_count = 0
    
    # Estado del challenge
    status = "FASE 1"
    
    # Agregar estado inicial
    evolution.append({
        'challenge_num': challenge_num,
        'trade_index': start_index,
        'trade_number': 0,
        'date': None,
        'symbol': 'INICIAL',
        'type': '-',
        'original_profit': 0,
        'scaled_profit': 0,
        'scaled_profit_pct': 0,
        'balance': balance,
        'total_profit_pct': 0,
        'daily_profit_pct': 0,
        'phase': current_phase,
        'progress_to_target': 0,
        'drawdown_from_hwm': 0,
        'status': status,
        'daily_trades': 0
    })
    
    # Procesar trades
    end_index = start_index + max_trades if max_trades else len(df_trades)
    for i in range(start_index, min(end_index, len(df_trades))):
        trade = df_trades.iloc[i]
        trade_date = trade['date']
        original_profit = trade['profit_loss']
        
        # Escalar el profit proporcionalmente basado en el riesgo por operación
        scaled_profit = original_profit * scaling_factor
        position_size = scaling_factor
        
        scaled_profit_pct = (scaled_profit / balance) * 100
        
        # Detectar cambio de día
        if current_date != trade_date:
            # Verificar límite del día anterior
            if current_date is not None and daily_profit_pct < -daily_dd_limit:
                status = f"SUSPENDIDO - DD Diario Fase {current_phase}"
                break
            
            # Resetear para nuevo día
            current_date = trade_date
            daily_profit_pct = 0
            daily_trades_count = 0
        
        # Ejecutar trade
        trades_used += 1
        daily_trades_count += 1
        daily_profit_pct += scaled_profit_pct
        total_profit_pct += scaled_profit_pct
        balance += scaled_profit
        high_water_mark = max(high_water_mark, balance)
        
        # Calcular drawdown actual
        current_dd = (high_water_mark - balance) / high_water_mark * 100
        
        # Calcular progreso hacia el objetivo
        if current_phase == 1:
            progress_to_target = (total_profit_pct / phase1_goal) * 100
        else:
            progress_to_target = ((total_profit_pct - phase1_goal) / phase2_goal) * 100
        
        # Verificar drawdown total
        if current_dd >= total_dd_limit:
            status = f"SUSPENDIDO - DD Total Fase {current_phase}"
        
        # Verificar drawdown diario DESPUÉS de cada trade
        elif daily_profit_pct < -daily_dd_limit:
            status = f"SUSPENDIDO - DD Diario Fase {current_phase}"
        
        # Verificar objetivos de fase
        elif current_phase == 1 and total_profit_pct >= phase1_goal:
            current_phase = 2
            status = "FASE 2"
        elif current_phase == 2 and total_profit_pct >= phase1_goal + phase2_goal:
            status = "APROBADO"
        elif current_phase == 1:
            status = "FASE 1"
        
        # Registrar evolución del trade
        evolution.append({
            'challenge_num': challenge_num,
            'trade_index': i,
            'trade_number': trades_used,
            'date': trade_date,
            'symbol': trade['Symbol'] if 'Symbol' in trade else 'N/A',
            'type': trade['Type'] if 'Type' in trade else 'N/A',
            'close_type': trade['close_type'] if 'close_type' in trade else 'Unknown',
            'original_profit': original_profit,
            'scaled_profit': scaled_profit,
            'scaled_profit_pct': scaled_profit_pct,
            'position_size': position_size,
            'risk_amount': risk_per_trade_dollars,
            'scaling_factor': scaling_factor,
            'max_loss_csv': max_loss_csv,
            'balance': balance,
            'total_profit_pct': total_profit_pct,
            'daily_profit_pct': daily_profit_pct,
            'phase': current_phase,
            'progress_to_target': progress_to_target,
            'drawdown_from_hwm': current_dd,
            'status': status,
            'daily_trades': daily_trades_count
        })
        
        # Si terminó el challenge, salir
        if status in ["APROBADO", "SUSPENDIDO - DD Diario Fase 1", "SUSPENDIDO - DD Diario Fase 2", 
                     "SUSPENDIDO - DD Total Fase 1", "SUSPENDIDO - DD Total Fase 2"]:
            break
    
    # Verificar último día si no terminó
    if status in ["FASE 1", "FASE 2"] and daily_profit_pct < -daily_dd_limit:
        evolution[-1]['status'] = f"SUSPENDIDO - DD Diario Fase {current_phase}"
        status = f"SUSPENDIDO - DD Diario Fase {current_phase}"
    
    # Si el challenge no terminó pero se quedó sin datos, marcarlo como incompleto
    if status in ["FASE 1", "FASE 2"]:
        evolution[-1]['status'] = f"INCOMPLETO - {status}"
        status = f"INCOMPLETO - {status}"
    
    return evolution, trades_used, status

def create_challenges_evolution_table(df_trades, config, num_challenges=5):
    """
    Crea tabla de evolución para múltiples challenges
    """
    
    all_evolutions = []
    current_index = 0
    
    for challenge_num in range(1, num_challenges + 1):
        if current_index >= len(df_trades) - 10:
            break
            
        evolution, trades_used, final_status = run_detailed_challenge_tracking(
            df_trades, current_index, config, challenge_num, max_trades=None
        )
        
        all_evolutions.extend(evolution)
        
        # Solo avanzar al siguiente challenge si el actual terminó completamente
        if final_status in ["APROBADO", "SUSPENDIDO - DD Diario Fase 1", "SUSPENDIDO - DD Diario Fase 2", 
                          "SUSPENDIDO - DD Total Fase 1", "SUSPENDIDO - DD Total Fase 2",
                          "INCOMPLETO - FASE 1", "INCOMPLETO - FASE 2"]:
            current_index += trades_used
        else:
            # Si el challenge no terminó (está en FASE 1 o FASE 2), no generar más challenges
            break
        
        if current_index >= len(df_trades) - 10:
            break
    
    return pd.DataFrame(all_evolutions)

def display_evolution_interface(df_csv, config):
    """
    Interfaz para mostrar la evolución de todos los challenges disponibles
    """
    
    # Inicializar estado de sesión al principio
    if 'evolution_df' not in st.session_state:
        st.session_state.evolution_df = None
    if 'evolution_config_hash' not in st.session_state:
        st.session_state.evolution_config_hash = None
    
    # Preparar datos
    df_processed = df_csv.copy()
    df_processed['Open time'] = pd.to_datetime(df_processed['Open time'])
    df_processed['Close time'] = pd.to_datetime(df_processed['Close time'])
    df_processed['profit_loss'] = pd.to_numeric(df_processed['Profit/Loss'])
    df_processed['date'] = df_processed['Close time'].dt.date
    df_processed = df_processed.sort_values('Close time').reset_index(drop=True)
    
    # Procesar tipo de cierre
    df_processed['close_type'] = df_processed['Close type'].fillna('Unknown')
    
    # Crear hash de la configuración para detectar cambios
    import hashlib
    config_str = str(sorted(config.items()))
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    
    # Solo recalcular si la configuración cambió o no hay datos en caché
    if (st.session_state.evolution_df is None or 
        st.session_state.evolution_config_hash != config_hash):
        
        with st.spinner("Generando evolución de challenges..."):
            evolution_df = create_challenges_evolution_table(df_processed, config, num_challenges=20)
            st.session_state.evolution_df = evolution_df
            st.session_state.evolution_config_hash = config_hash
    else:
        evolution_df = st.session_state.evolution_df
    
    if evolution_df.empty:
        st.warning("No se pudieron generar challenges con los datos disponibles.")
        return
    
    # Calcular estadísticas del CSV
    total_dias_csv = df_processed['date'].nunique()
    fecha_inicio = df_processed['date'].min()
    fecha_fin = df_processed['date'].max()
    
    # Calcular estadísticas de challenges
    available_challenges = sorted(evolution_df['challenge_num'].unique())
    total_challenges = len(available_challenges)
    
    # Contar challenges por estado
    challenges_aprobados = 0
    challenges_suspendidos = 0
    challenges_incompletos = 0
    
    for challenge_num in available_challenges:
        challenge_data = evolution_df[evolution_df['challenge_num'] == challenge_num]
        final_status = challenge_data.iloc[-1]['status']
        
        if final_status == "APROBADO":
            challenges_aprobados += 1
        elif "SUSPENDIDO" in final_status:
            challenges_suspendidos += 1
        else:
            challenges_incompletos += 1
    
    # Mostrar estadísticas en métricas
    st.markdown("### Estadísticas del Análisis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📅 Días Totales",
            f"{total_dias_csv}",
            help=f"Del {fecha_inicio} al {fecha_fin}"
        )
    
    with col2:
        st.metric(
            "🎯 Challenges Totales",
            f"{total_challenges}",
            help="Número total de challenges simulados"
        )
    
    with col3:
        st.metric(
            "✅ Challenges Aprobados",
            f"{challenges_aprobados}",
            f"{challenges_aprobados/total_challenges*100:.1f}%" if total_challenges > 0 else "0%",
            help="Challenges que completaron ambas fases"
        )
    
    with col4:
        st.metric(
            "❌ Challenges Suspendidos",
            f"{challenges_suspendidos}",
            f"{challenges_suspendidos/total_challenges*100:.1f}%" if total_challenges > 0 else "0%",
            help="Challenges que superaron los límites de drawdown"
        )
    
    # Mostrar información adicional
    if challenges_incompletos > 0:
        st.info(f"ℹ️ **{challenges_incompletos} challenges incompletos** (no terminaron por falta de datos)")
    
    st.markdown("---")
    
    st.markdown("""
    <div class="card">
        <div class="card-title">Evolución Detallada de Challenges</div>
    """, unsafe_allow_html=True)

    # Información sobre la funcionalidad
    st.info("Esta tabla muestra la evolución trade por trade de cada challenge, permitiendo ver exactamente cómo progresa el balance y en qué momento se aprueba o suspende cada examen.")
    
    # Mostrar información de todos los challenges
    st.markdown(f"**Mostrando todos los {total_challenges} challenges disponibles**")
    
    # Mostrar todos los challenges
    challenges_to_show = available_challenges
    
    for challenge_num in challenges_to_show:
        # Filtrar datos del challenge actual
        challenge_data = evolution_df[evolution_df['challenge_num'] == challenge_num].copy()
        
        # Mostrar información del challenge
        final_status = challenge_data.iloc[-1]['status']
        total_trades = challenge_data.iloc[-1]['trade_number']
        final_balance = challenge_data.iloc[-1]['balance']
        final_profit_pct = challenge_data.iloc[-1]['total_profit_pct']
        
        # Calcular días usados y fechas
        start_date = challenge_data.iloc[1]['date']  # Primer trade (excluyendo estado inicial)
        end_date = challenge_data.iloc[-1]['date']   # Último trade
        days_used = (end_date - start_date).days + 1  # +1 para incluir ambos días
        
        # Determinar color del estado
        status_color = "green" if "APROBADO" in final_status else "red" if "SUSPENDIDO" in final_status else "orange" if "INCOMPLETO" in final_status else "blue"
        status_icon = "✅" if "APROBADO" in final_status else "❌" if "SUSPENDIDO" in final_status else "⚠️" if "INCOMPLETO" in final_status else "🔄"
        
        # Crear panel expandible para cada challenge
        with st.expander(f"{status_icon} Challenge {challenge_num} - {final_status} | Trades: {total_trades} | Días: {days_used} | {start_date.strftime('%d/%m')} - {end_date.strftime('%d/%m')} | Ganancia: {final_profit_pct:.2f}%", expanded=False):
            
            # Obtener información del escalado del primer trade
            first_trade = challenge_data[challenge_data['trade_number'] > 0].iloc[0] if len(challenge_data[challenge_data['trade_number'] > 0]) > 0 else None
            scaling_info = ""
            if first_trade is not None:
                scaling_factor = first_trade['scaling_factor']
                max_loss_csv = first_trade['max_loss_csv']
                risk_amount = first_trade['risk_amount']
            
            # Header del challenge dentro del panel
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 7px solid {status_color};">
                <h4 style="margin: 0; color: {status_color};">Challenge {challenge_num}</h4>
                <p style="margin: 0.5rem 0 0 0; color: #6c757d;">
                    <strong>Estado Final:</strong> {final_status} | 
                    <strong>Trades Utilizados:</strong> {total_trades} | 
                    <strong>Días Usados:</strong> {days_used} | 
                    <strong>Período:</strong> {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')} | 
                    <strong>Ganancia Total:</strong> {final_profit_pct:.2f}%{scaling_info}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
            # Preparar tabla para mostrar
            display_df = challenge_data[challenge_data['trade_number'] > 0].copy()  # Excluir estado inicial
            
            # Formatear columnas para mejor visualización
            display_df['Balance'] = display_df['balance'].apply(lambda x: f"${x:,.2f}")
            display_df['Profit Original'] = display_df['original_profit'].apply(lambda x: f"${x:.2f}")
            display_df['Profit Escalado'] = display_df['scaled_profit'].apply(lambda x: f"${x:.2f}")
            display_df['Profit %'] = display_df['scaled_profit_pct'].apply(lambda x: f"{x:.2f}%")
            display_df['Profit Total %'] = display_df['total_profit_pct'].apply(lambda x: f"{x:.2f}%")
            display_df['Profit Diario %'] = display_df['daily_profit_pct'].apply(lambda x: f"{x:.2f}%")
            display_df['Progreso Objetivo'] = display_df['progress_to_target'].apply(lambda x: f"{x:.1f}%")
            display_df['Drawdown'] = display_df['drawdown_from_hwm'].apply(lambda x: f"{x:.2f}%")
            display_df['Factor Escalado'] = display_df['scaling_factor'].apply(lambda x: f"{x:.2f}x")
            display_df['Riesgo $'] = display_df['risk_amount'].apply(lambda x: f"${x:,.2f}")
            
            # Formatear tipo de cierre con siglas
            def format_close_type(close_type):
                if pd.isna(close_type) or close_type == 'Unknown':
                    return 'N/A'
                elif close_type == 'SL':
                    return 'SL'
                elif close_type == 'TrailingStop':
                    return 'TS'
                elif 'End Of Friday' in str(close_type):
                    return 'EOF'
                elif 'TP' in str(close_type):
                    return 'TP'
                else:
                    return str(close_type)[:3].upper()
            
            display_df['Tipo Cierre'] = display_df['close_type'].apply(format_close_type)
            
            # Seleccionar columnas para mostrar
            columns_to_show = [
                'trade_number', 'date', 'symbol', 'type', 'Tipo Cierre', 'Profit %', 
                'Balance', 'Profit Total %', 'Profit Diario %', 'Drawdown', 'status', 'daily_trades'
            ]
            
            final_display_df = display_df[columns_to_show].rename(columns={
                'trade_number': 'Trade #',
                'date': 'Fecha',
                'symbol': 'Symbol',
                'type': 'Tipo',
                'Tipo Cierre': 'Cierre',
                'phase': 'Fase',
                'status': 'Estado',
                'daily_trades': 'Trades Día'
            })
            
            # Mostrar tabla con colores
            def highlight_rows(row):
                if 'SUSPENDIDO' in str(row['Estado']):
                    return ['background-color: #ffebee'] * len(row)
                elif 'APROBADO' in str(row['Estado']):
                    return ['background-color: #e8f5e8'] * len(row)
                elif 'FASE 2' in str(row['Estado']):
                    return ['background-color: #e3f2fd'] * len(row)
                else:
                    return [''] * len(row)
            
            # Aplicar estilo y mostrar tabla
            styled_df = final_display_df.style.apply(highlight_rows, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Gráfico de evolución del balance
            st.markdown("#### Gráfico de Evolución del Balance")
            
            fig = go.Figure()
            
            # Línea del balance
            fig.add_trace(go.Scatter(
                x=challenge_data['trade_number'],
                y=challenge_data['balance'],
                mode='lines+markers',
                name='Balance',
                line=dict(color='blue', width=2),
                marker=dict(size=5)
            ))
            
            # Línea objetivo Fase 1
            phase1_target_balance = config['account_size'] * (1 + config['phase1_target'] / 100)
            fig.add_hline(
                y=phase1_target_balance, 
                line_dash="dash", 
                line_color="green", 
                annotation_text=f"Objetivo Fase 1: ${phase1_target_balance:,.0f}"
            )
            
            # Línea objetivo Fase 2
            phase2_target_balance = config['account_size'] * (1 + (config['phase1_target'] + config['phase2_target']) / 100)
            fig.add_hline(
                y=phase2_target_balance, 
                line_dash="dash", 
                line_color="darkgreen", 
                annotation_text=f"Objetivo Final: ${phase2_target_balance:,.0f}"
            )
            
            # Línea de drawdown máximo
            max_dd_balance = config['account_size'] * (1 - config['total_dd_limit'] / 100)
            fig.add_hline(
                y=max_dd_balance, 
                line_dash="dot", 
                line_color="red", 
                annotation_text=f"Límite DD: ${max_dd_balance:,.0f}"
            )
            
            fig.update_layout(
                title=f"Evolución del Balance - Challenge {challenge_num}",
                xaxis_title="Número de Trade",
                yaxis_title="Balance ($)",
                template="plotly_white",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Resumen de todos los challenges
    st.markdown("#### Resumen de Todos los Challenges")
    
    summary_data = []
    for challenge_num in available_challenges:
        challenge_summary = evolution_df[evolution_df['challenge_num'] == challenge_num].iloc[-1]
        
        # Calcular días y fechas para el resumen
        challenge_data = evolution_df[evolution_df['challenge_num'] == challenge_num].copy()
        start_date = challenge_data.iloc[1]['date']  # Primer trade
        end_date = challenge_data.iloc[-1]['date']    # Último trade
        days_used = (end_date - start_date).days + 1
        
        summary_data.append({
            'Challenge': challenge_num,
            'Estado Final': challenge_summary['status'],
            'Trades Usados': challenge_summary['trade_number'],
            'Días Usados': days_used,
            'Período': f"{start_date.strftime('%d/%m')} - {end_date.strftime('%d/%m')}",
            'Ganancia Total': f"{challenge_summary['total_profit_pct']:.2f}%",
            'Fase Alcanzada': challenge_summary['phase'],
            'DD Máximo': f"{challenge_summary['drawdown_from_hwm']:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Aplicar colores al resumen
    def highlight_summary(row):
        if 'SUSPENDIDO' in str(row['Estado Final']):
            return ['background-color: #ffebee'] * len(row)
        elif 'APROBADO' in str(row['Estado Final']):
            return ['background-color: #e8f5e8'] * len(row)
        else:
            return [''] * len(row)
    
    styled_summary = summary_df.style.apply(highlight_summary, axis=1)
    st.dataframe(styled_summary, use_container_width=True, hide_index=True)
    
    # Información sobre todos los challenges mostrados
    st.markdown("---")
    st.info(f"💡 **Información**: Se muestran todos los {total_challenges} challenges disponibles. Puedes expandir cada uno para ver los detalles completos.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Configurar tema claro para Streamlit
import streamlit.components.v1 as components

# Configuración de la página
st.set_page_config(
    page_title="IMOXHUB", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_icon="logo.png"
)

# Inicializar estado de sesión para el login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user_rank' not in st.session_state:
    st.session_state.user_rank = None
if 'first_login' not in st.session_state:
    st.session_state.first_login = False

def login_user(nick: str, password: str):
    """
    Autentica al usuario contra el backend
    Returns: (success, message, data)
    """
    try:
        # Normalizar el nick a minúsculas para hacer login case-insensitive
        login_data = {
            "nick": nick.strip(),
            "password": password
        }
        response = requests.post(f"{API_BASE_URL}/auth/login", json=login_data, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return True, "Login exitoso", data
        elif response.status_code == 429:
            return False, "Demasiados intentos fallidos. Por favor espera unos minutos.", None
        else:
            # Intentar obtener el mensaje de error
            try:
                error_detail = response.json().get("detail", f"Error {response.status_code}")
            except:
                error_detail = f"Error {response.status_code}: {response.text}"
            return False, error_detail, None
    except requests.exceptions.ConnectionError:
        return False, f"No se puede conectar con el backend. Asegúrate de que esté ejecutándose en {API_BASE_URL}", None
    except requests.exceptions.Timeout:
        return False, "Tiempo de espera agotado. Por favor intenta de nuevo.", None
    except Exception as e:
        return False, f"Error inesperado: {str(e)}", None

def reset_password_api(target_nick: str, token: str):
    """
    Resetea la contraseña de un usuario (solo admin)
    Returns: (success, message)
    """
    try:
        reset_data = {
            "target_nick": target_nick
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        response = requests.post(f"{API_BASE_URL}/auth/reset-password", json=reset_data, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return True, data.get("message", "Contraseña reseteada correctamente")
        elif response.status_code == 403:
            return False, "No tienes permisos para realizar esta acción. Solo los administradores pueden resetear contraseñas."
        elif response.status_code == 404:
            return False, "Usuario no encontrado"
        elif response.status_code == 500:
            # Internal Server Error - intentar obtener el detalle
            try:
                error_response = response.json()
                error_detail = error_response.get("detail", "Error interno del servidor")
            except:
                error_detail = f"Error interno del servidor (500). {response.text[:200] if response.text else ''}"
            return False, error_detail
        else:
            # Intentar obtener el mensaje de error del JSON
            try:
                error_response = response.json()
                error_detail = error_response.get("detail", f"Error {response.status_code}")
            except ValueError:
                error_detail = response.text.strip() if response.text.strip() else f"Error {response.status_code}: Respuesta no válida del servidor"
            except Exception as json_error:
                error_detail = f"Error {response.status_code}: No se pudo procesar la respuesta del servidor ({str(json_error)})"
            return False, error_detail
    except requests.exceptions.ConnectionError:
        return False, f"No se puede conectar con el backend. Asegúrate de que esté ejecutándose en {API_BASE_URL}"
    except requests.exceptions.Timeout:
        return False, "Tiempo de espera agotado. Por favor intenta de nuevo."
    except Exception as e:
        return False, f"Error inesperado: {str(e)}"

def set_password_api(password: str, confirm_password: str, token: str):
    """
    Establece la contraseña en el primer login
    Returns: (success, message)
    """
    try:
        password_data = {
            "password": password,
            "confirm_password": confirm_password
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        response = requests.post(f"{API_BASE_URL}/auth/set-password", json=password_data, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return True, "Contraseña establecida correctamente"
        elif response.status_code == 401:
            # Token inválido o expirado
            try:
                error_detail = response.json().get("detail", "Token inválido o expirado")
            except:
                error_detail = "Token inválido o expirado. Por favor, cierra sesión y vuelve a iniciar sesión."
            return False, error_detail
        elif response.status_code == 500:
            # Internal Server Error - intentar obtener el detalle
            try:
                error_response = response.json()
                error_detail = error_response.get("detail", "Error interno del servidor")
            except:
                error_detail = f"Error interno del servidor (500). {response.text[:200] if response.text else ''}"
            return False, error_detail
        else:
            # Intentar obtener el mensaje de error del JSON
            try:
                error_response = response.json()
                error_detail = error_response.get("detail", f"Error {response.status_code}")
            except ValueError:
                # Si no es JSON válido (respuesta vacía o HTML), usar el texto de la respuesta
                error_detail = response.text.strip() if response.text.strip() else f"Error {response.status_code}: Respuesta no válida del servidor"
            except Exception as json_error:
                error_detail = f"Error {response.status_code}: No se pudo procesar la respuesta del servidor ({str(json_error)})"
            return False, error_detail
    except requests.exceptions.ConnectionError:
        return False, f"No se puede conectar con el backend. Asegúrate de que esté ejecutándose en {API_BASE_URL}"
    except requests.exceptions.Timeout:
        return False, "Tiempo de espera agotado. Por favor intenta de nuevo."
    except Exception as e:
        return False, f"Error inesperado: {str(e)}"

# Función para mostrar la página de login
def show_login_page():
    """Muestra la página de login"""
    st.markdown("""
    <style>
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    .login-form {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    .login-title {
        color: #495057;
    }
    .login-error {
        background: #fff5f5;
        border: 1px solid #f5c2c7;
        color: #842029;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin: 0 auto 1rem auto;
        max-width: 400px;
        text-align: center;
        font-weight: 500;
    }
    /* Animación de vibración */
    @keyframes shake {
        10%, 90% { transform: translateX(-1px); }
        20%, 80% { transform: translateX(2px); }
        30%, 50%, 70% { transform: translateX(-4px); }
        40%, 60% { transform: translateX(4px); }
    }
    #login-panel.shake {
        animation: shake 0.6s ease;
    }
    #login-panel.error-flash {
        border-color: #de3a3a !important;
        box-shadow: 0 0 0 3px rgba(222, 58, 58, 0.15);
    }
    /* Ocultar el contenido principal de Streamlit cuando se muestra el login */
    .main .block-container {
        padding-top: 0 !important;
    }

    .stMainBlockContainer{
        width: auto !important;
        min-width: 400px !important;
    }

    .stForm{
        background-color: white;
        border-radius: 0.75rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        padding: 2rem;
        width: 100%;
        min-width: 350px;
        max-width: 400px;
        border: 1px solid #e9ecef;
        text-align: center;
        margin: 0 auto;
    }
    
    [data-testid="InputInstructions"] {
        display: none !important;
    }
    
    /* Eliminar líneas separadoras entre filas de las tablas */
    [data-testid="stDataFrame"] table tbody tr {
        border-bottom: none !important;
    }
    
    [data-testid="stDataFrame"] table tbody tr td {
        border-bottom: none !important;
    }
    
    /* Mantener solo las líneas del header */
    [data-testid="stDataFrame"] table thead tr {
        border-bottom: 1px solid #e0e0e0 !important;
    }
    
    [data-testid="stDataFrame"] table thead tr th {
        border-bottom: 1px solid #e0e0e0 !important;
    }    
    
    /* Alinear los headers de las columnas */
    div[data-testid="column"] {
        align-items: flex-start !important;
    }
    
    div[data-testid="column"] > div {
        align-items: flex-start !important;
    }
    
    </style>
    """, unsafe_allow_html=True)
    
    logo_base64 = get_base64_encoded_image("logo.png")
    
    st.markdown(f"""
    <div class="login-container">
        <div class="">
            <div class="login-logo">
                <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="height: 100px;">
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        st.markdown('<div id="login-panel" class="login-form">', unsafe_allow_html=True)
        
        usuario = st.text_input("Usuario (Nick)", placeholder="Ingresa tu nick de usuario")
        contraseña = st.text_input("Contraseña", type="password", placeholder="Deja vacío si es tu primer acceso")
        st.caption("💡 Si es tu primer acceso, puedes dejar la contraseña vacía")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            login_button = st.form_submit_button("Entrar", use_container_width=True)
        
        if login_button:
            if not usuario:
                st.error("Por favor ingresa tu usuario (nick)")
            else:
                # Para primer login, la contraseña puede estar vacía
                password = contraseña if contraseña else ""
                
                with st.spinner("Autenticando..."):
                    success, message, data = login_user(usuario, password)
                
                if success:
                    # Guardar datos de sesión
                    st.session_state.logged_in = True
                    st.session_state.username = data['user']['nick']
                    st.session_state.token = data['access_token']
                    st.session_state.user_rank = data['user']['rank']
                    st.session_state.first_login = data['first_login']
                    st.rerun()
                else:
                    # Mostrar error
                    st.error(message)
                    # Marcar error y animar el panel
                    st.session_state.login_error = True
                    st.markdown("""
                    <script>
                    setTimeout(() => {
                        const panel = document.getElementById('login-panel');
                        if (panel) {
                            panel.classList.remove('shake', 'error-flash');
                            void panel.offsetWidth; // reflow
                            panel.classList.add('error-flash', 'shake');
                            setTimeout(() => {
                                panel.classList.remove('shake', 'error-flash');
                            }, 900);
                        }
                    }, 100);
                    </script>
                    """, unsafe_allow_html=True)

    # Mostrar error fuera del panel si existe
    if 'login_error' in st.session_state and st.session_state.login_error:
        st.markdown('<div class="login-error">Credenciales inválidas</div>', unsafe_allow_html=True)
        st.session_state.login_error = False  # Resetear después de mostrar

# Verificar si el usuario está logueado
if not st.session_state.logged_in:
    show_login_page()
    st.stop()

# Si es primer login, mostrar formulario para establecer contraseña
if st.session_state.first_login:
    st.markdown("""
    <style>
    .password-setup-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
        min-height: 400px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="password-setup-container">
    """, unsafe_allow_html=True)
    
    st.warning("⚠️ **Primer acceso detectado** - Por favor establece tu contraseña para continuar")
    
    with st.form("set_password_form"):
        st.markdown("### Establecer Contraseña")
        
        password1 = st.text_input("Nueva Contraseña", type="password", placeholder="Mínimo 6 caracteres")
        password2 = st.text_input("Confirmar Contraseña", type="password", placeholder="Repite la contraseña")
        
        submit_password = st.form_submit_button("Establecer Contraseña", use_container_width=True)
        
        if submit_password:
            if not password1 or not password2:
                st.error("Por favor completa ambos campos")
            elif password1 != password2:
                st.error("Las contraseñas no coinciden")
            elif len(password1) < 6:
                st.error("La contraseña debe tener al menos 6 caracteres")
            else:
                # Verificar que tenemos un token
                if not st.session_state.get('token'):
                    st.error("❌ Error: No se encontró el token de autenticación. Por favor, cierra sesión y vuelve a iniciar sesión.")
                else:
                    token = st.session_state.token
                    # Verificar que el token no esté vacío
                    if not token or not token.strip():
                        st.error("❌ Error: El token de autenticación está vacío. Por favor, cierra sesión y vuelve a iniciar sesión.")
                    else:
                        with st.spinner("Estableciendo contraseña..."):
                            success, message = set_password_api(password1, password2, token)
                        
                        if success:
                            st.success(message)
                            # Invalidar caché de usuarios
                            get_users_from_api.clear()
                            st.session_state.first_login = False
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                            # Si el error es de autenticación, sugerir cerrar sesión
                            if "401" in str(message) or "token" in str(message).lower() or "autorizado" in str(message).lower():
                                st.warning("💡 Tu sesión podría haber expirado. Por favor, cierra sesión y vuelve a iniciar sesión.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Botón de logout en la barra lateral
with st.sidebar:
    if st.session_state.logged_in:
        st.markdown(f"**Usuario:** {st.session_state.username}")
        if st.button("Cerrar sesión", use_container_width=True):
            # Limpiar toda la sesión
            for key in ["logged_in", "username", "token", "user_rank", "first_login"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

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
        border: 1px solid #e9ecef;
        box-sizing: border-box;
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #495057;
    }

    .card.ranking-panel {
        display: flex;
        align-items: center;
        width: calc(100% + 1rem);
        height: 50px;
        text-transform: uppercase;
    }

    .card-month{
        background-color: white;
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        padding: 0.5rem;
    }

    .card-month h2{
        text-transform: uppercase;
        text-align: center;
        padding: 0;
    }
    
    .stHorizontalBlock:has(.ranking-panel) > div:nth-child(2) button {
        height: 50px;
        width: 50px;
        border-radius: 0.75rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    }

    .stHorizontalBlock:has(.ranking-panel) > div:nth-child(2) button img{
        font-size: 25px;
    }

    .stHorizontalBlock:has(.ranking-panel) [data-testid="stExpanderDetails"]{
        margin: 0;
        padding: 0;
        overflow: hidden;
    }

    .stHorizontalBlock:has(.ranking-panel) [data-testid="stExpander"]{
        display: flex;
    }

    .stHorizontalBlock:has(.ranking-panel) [data-testid="stExpander"]{
        background-color: white;
    }

    .stHorizontalBlock:has(.ranking-panel) [data-testid="stExpander"] summary{
        background-color: white;
    }

    .stHorizontalBlock:has(.ranking-panel) [data-testid="stFullScreenFrame"]{
        display: flex;
        overflow: hidden;
    }

    [data-testid="stFileUploaderDeleteBtn"] button{
        border-radius: 50px;
    }

    .stHorizontalBlock:has(.ranking-panel) details{
        overflow: hidden !important;
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
        max-width: 1700px !important;
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

    h1{
        font-size: 2rem !important;
        padding-left: 0.5rem !important;
        color: black !important;
        padding: 0 !important;
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
        height: 20px !important;
        width: 30px !important;
        padding: 10px;
    }
    
    .stFileUploader button:hover {
        background-color: #5a6268 !important;
    }
    
    /* Estilos para gráficos de Plotly - forzar tema claro */
    .stPlotlyChart {
        border-radius: 0.75rem !important;
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
    


    .st-dv.st-fv, .st-bv.st-g5 {
        background-color: #495057 !important;
    }

    span.st-ae.st-af.st-ag.st-ah.st-ai.st-aj.st-cj {
        background-color: white !important;
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
    
    /* Responsive design para paneles de ranking */
    @media (max-width: 900px) {
        .ranking-panel {
            width: 100% !important;
            margin-bottom: 1rem !important;
        }
    }
    
    /* Forzar alineación del botón izquierdo a la derecha */
    .st-emotion-cache-wfksaw:first-child {
        align-items: flex-end !important;
        justify-content: center !important;
    }
    
    /* Forzar alineación del botón derecho al centro */
    .st-emotion-cache-wfksaw:last-child {
        align-items: end !important;
        justify-content: center !important;
    }

    .stElementContainer:has(.ranking-total) {
        margin-top: auto !important;
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

# Añadir logo a la izquierda de los tabs
logo_base64 = get_base64_encoded_image("logo.png")

# Header con logo y botón de logout
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 0.5rem; margin-left: 0.5rem;">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="height: 30px; margin-right: 10px;">
        <h1>IMOXHUB</h1>
        <div style="flex-grow: 1;"></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if st.button("Cerrar Sesión", key="logout"):
        # Limpiar toda la sesión
        for key in ["logged_in", "username", "token", "user_rank", "first_login"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Verificar si el usuario es admin
is_admin = st.session_state.user_rank == "admin"

# Crear tabs según si es admin o no
if is_admin:
    tab1, tab4 = st.tabs(["Análisis", "Administración"])
else:
    tab1, = st.tabs(["Análisis"])

# Contenedor principal
st.markdown('<div class="main-container">', unsafe_allow_html=True)


# Panel central con contenido dinámico según el tab
with tab1:
    # Crear tabs anidados dentro del tab 1
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Análisis de Portafolio", "Análisis de EA", "Análisis de Prop Firm"])
    
    with sub_tab1:
        # Tarjeta de carga de archivo
        st.markdown("""
        <div class="card">
            <div class="card-title">Analiza tu portafolio</div>
        """, unsafe_allow_html=True)

        archivos = st.file_uploader("Elegir archivo(s)", type=["htm", "html"], label_visibility="collapsed", accept_multiple_files=True)

        if archivos:
            try:
                datos = []
                eas_filtradas = 0
                total_operaciones = 0
                archivos_procesados = 0

                # Procesar cada archivo subido
                for archivo in archivos:
                    try:
                        soup = BeautifulSoup(archivo, 'html.parser')
                        rows = soup.find_all('tr', align='right')

                        for i in range(len(rows)):
                            try:
                                fila_op = rows[i].find_all('td')
                        
                                # Verificar que la fila tenga suficientes columnas para ser una operación
                                if len(fila_op) < 14:
                                    continue  # Saltar filas con estructura incompleta
                        
                                # Verificar que la primera celda contenga un número (ticket)
                                try:
                                    ticket = int(fila_op[0].text.strip())
                                except:
                                    continue  # No es una fila de operación válida
                                
                                # Extraer información del atributo title para detectar [sl] o [tp]
                                tipo_cierre = None
                                try:
                                    title_attr = fila_op[0].get('title', '')
                                    if title_attr:
                                        title_lower = title_attr.lower()
                                        if '[sl]' in title_lower:
                                            tipo_cierre = 'SL'
                                        elif '[tp]' in title_lower:
                                            tipo_cierre = 'TP'
                                except:
                                    tipo_cierre = None
                                
                                # Verificar que la segunda celda contenga una fecha
                                try:
                                    open_time = datetime.strptime(fila_op[1].text.strip(), "%Y.%m.%d %H:%M:%S")
                                except:
                                    continue  # No es una fila de operación válida
                        
                                # Buscar la fila EA correspondiente (siguiente fila)
                                ea_raw = ""
                                if i + 1 < len(rows):
                                    fila_ea = rows[i+1].find_all('td')
                                    if len(fila_ea) > 0:
                                        ea_raw = fila_ea[-1].text.strip()

                                if "cancelled" in ea_raw.lower():
                                    continue

                                tipo = fila_op[2].text.strip().lower()
                                size = float(fila_op[3].text.replace(' ', ''))
                                symbol = fila_op[4].text.strip().lower()
                                
                                # Extraer precios, SL y TP
                                try:
                                    open_price = float(fila_op[5].text.strip().replace(' ', ''))
                                except:
                                    open_price = None
                                sl_str = fila_op[6].text.strip()
                                tp_str = fila_op[7].text.strip()
                                
                                close_time = datetime.strptime(fila_op[8].text.strip(), "%Y.%m.%d %H:%M:%S")
                                try:
                                    close_price = float(fila_op[9].text.strip().replace(' ', ''))
                                except:
                                    close_price = None
                                profit = float(fila_op[13].text.replace(' ', ''))

                                # Limpiar el nombre de la EA eliminando números adicionales
                                ea_name = limpiar_nombre_ea(ea_raw)
                                total_operaciones += 1

                                # Si no hay nombre de EA válido, asignar un nombre genérico
                                if not es_ea_valida(ea_name):
                                    if ea_name.strip() == "":
                                        ea_name = "Sin EA"  # Operaciones sin nombre de EA
                                    else:
                                        eas_filtradas += 1
                                        continue

                                # Parsear TP numérico si es válido
                                tp_val = None
                                tp_decimals = None
                                try:
                                    tp_val = float(tp_str)
                                    if "." in tp_str:
                                        tp_decimals = len(tp_str.split(".")[-1])
                                    else:
                                        tp_decimals = 0
                                except:
                                    tp_val = None
                                    tp_decimals = None

                                datos.append({
                                    "EA": ea_name,
                                    "Símbolo": symbol,
                                    "Tipo": tipo,
                                    "Volumen": size,  # Volumen de la operación
                                    "Beneficio": profit,
                                    "Open": open_time,
                                    "Close": close_time,
                                    "PrecioOpen": open_price,
                                    "PrecioClose": close_price,
                                    "SL": sl_str,
                                    "TP": tp_str,
                                    "TP_val": tp_val,
                                    "TP_decimales": tp_decimals,
                                    "TipoCierre": tipo_cierre,  # 'SL', 'TP', o None
                                    "Duración": (close_time - open_time).total_seconds() / 60  # en minutos
                                })
                            except Exception as e:
                                # Log del error para debugging (opcional)
                                # print(f"Error procesando fila {i}: {e}")
                                continue
                        
                        archivos_procesados += 1
                    except Exception as e:
                        st.error(f"Error procesando archivo {archivo.name}: {str(e)}")
                        continue

                if datos:
                    df = pd.DataFrame(datos)
                    
                    # Mostrar información sobre el procesamiento
                    info_mensaje = f"ℹ️ Se procesaron {archivos_procesados} archivo(s). Total de {total_operaciones} operaciones encontradas."
                    if eas_filtradas > 0:
                        info_mensaje += f" Se eliminaron {eas_filtradas} operaciones que no pertenecen a ninguna EA. Se procesaron {len(datos)} operaciones válidas."
                    st.info(info_mensaje)

                
                    # Filtrar "Sin EA" para el resto del análisis
                    df = df[df['EA'] != 'Sin EA']
                    
                    if df.empty:
                        st.warning("No hay operaciones válidas para mostrar en las tablas.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        # Funciones auxiliares para calcular métricas
                        def calcular_max_drawdown(serie_beneficios):
                            """Calcula el drawdown máximo de una serie de beneficios acumulados"""
                            acumulado = serie_beneficios.cumsum()
                            running_max = acumulado.expanding().max()
                            drawdown = acumulado - running_max
                            return drawdown.min()
                        
                        def calcular_max_consecutive_loss(serie_beneficios):
                            """Calcula la máxima racha de pérdidas consecutivas"""
                            consecutivo = 0
                            max_consecutivo = 0
                            for val in serie_beneficios:
                                if val < 0:
                                    consecutivo += 1
                                    max_consecutivo = max(max_consecutivo, consecutivo)
                                else:
                                    consecutivo = 0
                            return max_consecutivo
                        
                        # Detectar TP real comparando precio de cierre con el TP configurado
                        def es_tp_real(fila):
                            try:
                                if pd.isna(fila.get('Beneficio')) or fila['Beneficio'] <= 0:
                                    return False
                                tp_val = fila.get('TP_val')
                                close_price = fila.get('PrecioClose')
                                if tp_val is None or close_price is None:
                                    return False
                                decs = fila.get('TP_decimales') if fila.get('TP_decimales') is not None else 0
                                # Tolerancia basada en decimales del TP (2 unidades del último decimal)
                                tol = max(10 ** (-(decs)), 1e-6)
                                return abs(close_price - tp_val) <= 2 * tol
                            except Exception:
                                return False

                        def contar_sl_tp(grupo):
                            """Cuenta cuántos trades tienen SL y cuántos tienen TP"""
                            cantidad_sl = 0
                            cantidad_tp = 0
                            
                            for sl in grupo['SL']:
                                try:
                                    if sl and str(sl).strip() != '' and float(sl) != 0.0:
                                        cantidad_sl += 1
                                except (ValueError, TypeError):
                                    pass
                            
                            # TP reales: TP definido y cierre en el TP (no TS positivo)
                            for _, fila in grupo.iterrows():
                                if es_tp_real(fila):
                                    cantidad_tp += 1
                            
                            return cantidad_sl, cantidad_tp
                        
                        def calcular_avg_trades_por_mes(grupo):
                            """Calcula el promedio de trades por mes"""
                            import numpy as np
                            # Obtener fechas de apertura
                            fechas = grupo['Open'].dt.to_period('M').value_counts()
                            if len(fechas) > 0:
                                return fechas.mean()
                            return 0
                        
                        # Análisis de riesgo por EA y SL directo vs trailing stop
                    st.markdown("""
                        <div style="margin-top: 2rem;">
                            <h3>Resumen de Estrategias</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Función helper para detectar múltiples niveles de SL (normal y alta volatilidad)
                    def detectar_niveles_sl(perdidas_serie, min_trades_por_nivel=2, tolerancia_agrupacion=0.15, umbral_min_sl=100):
                        """
                        Detecta múltiples niveles de SL agrupando pérdidas similares.
                        Identifica tanto SL normales como SL de alta volatilidad.
                        Si no hay alta volatilidad clara, usa un umbral más alto (P90) como SL único.
                        Retorna una lista de niveles de SL válidos ordenados de mayor a menor.
                        """
                        if len(perdidas_serie) == 0:
                            return []
                        
                        perdidas_sorted = perdidas_serie.sort_values(ascending=False)
                        candidatos_niveles = []
                        perdidas_usadas = set()
                        
                        # Agrupar pérdidas similares para detectar niveles de SL
                        for perdida in perdidas_sorted.values:
                            if perdida in perdidas_usadas:
                                continue
                            
                            # Buscar todas las pérdidas dentro del rango de tolerancia
                            rango_min = perdida * (1 - tolerancia_agrupacion)
                            rango_max = perdida * (1 + tolerancia_agrupacion)
                            perdidas_en_rango = perdidas_serie[
                                (perdidas_serie >= rango_min) & 
                                (perdidas_serie <= rango_max)
                            ]
                            
                            # Si hay suficientes trades en este rango, es un candidato
                            if len(perdidas_en_rango) >= min_trades_por_nivel:
                                # Usar la mediana del rango como nivel representativo
                                nivel = perdidas_en_rango.median()
                                
                                candidatos_niveles.append({
                                    'nivel': nivel,
                                    'count': len(perdidas_en_rango),
                                    'perdidas': perdidas_en_rango.values
                                })
                                # Marcar estas pérdidas como usadas
                                perdidas_usadas.update(perdidas_en_rango.values)
                        
                        if len(candidatos_niveles) == 0:
                            # Si no hay candidatos agrupados, usar percentil 90 como fallback
                            p90 = perdidas_serie.quantile(0.90)
                            return [round(p90 / 10) * 10]
                        
                        # Ordenar candidatos por cantidad de trades (más común primero)
                        candidatos_niveles.sort(key=lambda x: x['count'], reverse=True)
                        
                        nivel_principal = candidatos_niveles[0]['nivel']
                        p90 = perdidas_serie.quantile(0.90)
                        
                        # Verificar si hay alta volatilidad:
                        # 1. El nivel principal no debe ser demasiado bajo comparado con P90 (< 80% de P90)
                        # 2. Necesita pérdidas significativamente mayores (> 1.3x el nivel principal) 
                        # 3. Y que estén agrupadas (formen un nivel claro con al menos 3 trades)
                        nivel_principal_es_bajo = nivel_principal < p90 * 0.80
                        
                        perdidas_alta_volatilidad = perdidas_serie[perdidas_serie > nivel_principal * 1.3]
                        # Verificar si las pérdidas de alta volatilidad están agrupadas (forman un nivel)
                        hay_alta_volatilidad = False
                        if not nivel_principal_es_bajo and len(perdidas_alta_volatilidad) >= 3:
                            # Verificar si están agrupadas (dentro de una tolerancia del 15%)
                            for perdida_alta in perdidas_alta_volatilidad.sort_values(ascending=False).head(5).values:
                                rango_min = perdida_alta * (1 - tolerancia_agrupacion)
                                rango_max = perdida_alta * (1 + tolerancia_agrupacion)
                                perdidas_en_rango_alta = perdidas_alta_volatilidad[
                                    (perdidas_alta_volatilidad >= rango_min) & 
                                    (perdidas_alta_volatilidad <= rango_max)
                                ]
                                if len(perdidas_en_rango_alta) >= 3:
                                    hay_alta_volatilidad = True
                                    break
                        
                        niveles_sl = []
                        
                        if hay_alta_volatilidad:
                            # Hay alta volatilidad: usar múltiples niveles
                            # Agregar el nivel principal si es válido
                            if nivel_principal >= max(umbral_min_sl, 80):
                                niveles_sl.append(nivel_principal)
                            
                            # Agregar otros niveles si:
                            # - Tienen al menos el 30% de los trades del nivel principal
                            # - Están dentro de un rango razonable (no más de 3x el nivel principal)
                            # - Son >= umbral_min_sl (o al menos 80)
                            umbral_min_trades = max(min_trades_por_nivel, candidatos_niveles[0]['count'] * 0.3)
                            max_nivel_aceptable = nivel_principal * 3.0
                            
                            for candidato in candidatos_niveles[1:]:
                                nivel = candidato['nivel']
                                count = candidato['count']
                                
                                if (count >= umbral_min_trades and 
                                    nivel <= max_nivel_aceptable and 
                                    nivel >= max(umbral_min_sl, 80)):
                                    niveles_sl.append(nivel)
                            
                            # Si no hay niveles válidos, usar el nivel principal aunque sea bajo
                            if len(niveles_sl) == 0 and nivel_principal >= 80:
                                niveles_sl.append(nivel_principal)
                        else:
                            # NO hay alta volatilidad: usar un umbral más alto (P90) como SL único
                            # Esto asegura que solo las pérdidas más grandes se consideren SL directo
                            p90 = perdidas_serie.quantile(0.90)
                            p85 = perdidas_serie.quantile(0.85)
                            
                            # Usar P90 si es razonable, sino P85, sino el nivel principal
                            if p90 >= max(umbral_min_sl, 80):
                                umbral_final = p90
                            elif p85 >= max(umbral_min_sl, 80):
                                umbral_final = p85
                            elif nivel_principal >= 80:
                                umbral_final = nivel_principal
                            else:
                                # Fallback: usar la pérdida máxima
                                umbral_final = perdidas_serie.max()
                            
                            niveles_sl.append(umbral_final)
                        
                        # Ordenar niveles de mayor a menor
                        niveles_sl.sort(reverse=True)
                        
                        # Redondear a múltiplos de 10 (solo para visualización)
                        # Mantener precisión en los cálculos usando los valores originales
                        niveles_sl_redondeados = [round(n / 10) * 10 for n in niveles_sl]
                        
                        return niveles_sl_redondeados
                    
                    analisis_data = []
                    for (ea, simbolo), grupo in df.groupby(["EA", "Símbolo"]):
                        try:
                            ordenado = grupo.sort_values(by='Open')
                            
                            # Detectar múltiples niveles de SL (para manejar días de alta volatilidad)
                            perdidas = ordenado[ordenado['Beneficio'] < 0]['Beneficio'].abs()
                            
                            if len(perdidas) > 0:
                                # Detectar múltiples niveles de SL
                                niveles_sl = detectar_niveles_sl(perdidas, min_trades_por_nivel=2, tolerancia_agrupacion=0.15, umbral_min_sl=100)
                                
                                # Si no encontramos niveles claros, usar el método anterior como fallback
                                if len(niveles_sl) == 0:
                                    perdida_maxima = perdidas.max()
                                    margen_busqueda = perdida_maxima * 0.10
                                    sl_detected = perdidas[perdidas >= (perdida_maxima - margen_busqueda)].mode()
                                    if len(sl_detected) > 0:
                                        niveles_sl = [sl_detected.iloc[0]]
                                    else:
                                        niveles_sl = [perdida_maxima]
                            
                                # Usar el nivel más común como riesgo_ea para compatibilidad (pero usaremos todos los niveles)
                                riesgo_ea = niveles_sl[0] if len(niveles_sl) > 0 else perdidas.max()
                                
                                # Analizar SL directo vs trailing stop (positivo y loss)
                                sl_directos = 0
                                sl_trailing = 0
                                tsp = 0  # Trailing Stop Positivo
                                tsl = 0  # Trailing Stop Loss
                                perdidas_nulas = 0
                            
                                # Margen de tolerancia (15% para considerar múltiples niveles)
                                margen_tolerancia = riesgo_ea * 0.15
                                
                                # Función helper para verificar si una pérdida es un SL directo
                                def es_sl_directo(perdida_abs):
                                    """
                                    Verifica si una pérdida es un SL directo:
                                    - Si es mayor o igual al nivel principal (si solo hay un nivel, con pequeño margen hacia abajo)
                                    - Si hay múltiples niveles (alta volatilidad), también verifica si está cerca de alguno
                                    """
                                    if len(niveles_sl) == 0:
                                        return False
                                    
                                    # Nivel principal es el primero (el más común y alto)
                                    nivel_principal = niveles_sl[0]
                                    
                                    # Si solo hay un nivel (no hay alta volatilidad), usar umbral con pequeño margen
                                    if len(niveles_sl) == 1:
                                        # Aplicar un pequeño margen hacia abajo (2.5% o mínimo $5 menos)
                                        umbral_ajustado = max(nivel_principal * 0.975, nivel_principal - 5)
                                        # Si la pérdida es muy cercana al umbral (dentro de $5), también considerarla SL
                                        # Esto evita que pérdidas como $191 se clasifiquen como TSL cuando el umbral es $190
                                        if perdida_abs >= umbral_ajustado:
                                            return True
                                        # Si está muy cerca del umbral principal (dentro de $5), también es SL
                                        if abs(perdida_abs - nivel_principal) <= 5:
                                            return True
                                        return False
                                    
                                    # Si hay múltiples niveles (alta volatilidad):
                                    # 1. Si es mayor o igual al nivel principal, es SL directo
                                    if perdida_abs >= nivel_principal:
                                        return True
                                    
                                    # 2. Si está cerca de alguno de los niveles detectados (dentro de tolerancia)
                                    for nivel in niveles_sl:
                                        tolerancia_nivel = nivel * 0.15
                                        if abs(perdida_abs - nivel) <= tolerancia_nivel:
                                            return True
                                    
                                    return False
                            
                            for _, trade in ordenado.iterrows():
                                beneficio = trade['Beneficio']
                                tipo_cierre = trade.get('TipoCierre')
                                    
                                # Si el HTML tiene información explícita de [sl] o [tp], usarla PERO validar con el beneficio
                                if tipo_cierre == 'SL':
                                    # [sl] en HTML: verificar si es pérdida o ganancia
                                    if beneficio < 0:
                                        # [sl] con pérdida: verificar si es SL directo o TSL
                                        perdida_abs = abs(beneficio)
                                        if es_sl_directo(perdida_abs):
                                            # SL directo: pérdida que alcanzó alguno de los SL esperados
                                            sl_directos += 1
                                        else:
                                            # TSL: pérdida menor que los SL esperados (trailing stop activó antes)
                                            sl_trailing += 1
                                            tsl += 1  # Trailing Stop Loss
                                    elif beneficio > 0:
                                        # [sl] en HTML pero con ganancia: es un Trailing Stop Positivo
                                        sl_trailing += 1
                                        tsp += 1  # Trailing Stop Positivo
                                    else:
                                        # Break-even con [sl]: TSL
                                        sl_trailing += 1
                                        tsl += 1
                                elif tipo_cierre == 'TP':
                                    # [tp] en HTML: confiamos en el HTML, es un TP real
                                    # Se contará abajo para mantener claridad
                                    pass
                                else:
                                    # Si no hay información explícita, usar la lógica de cálculo
                                    if beneficio < 0:  # Pérdida
                                        perdida_abs = abs(beneficio)
                                        if es_sl_directo(perdida_abs):
                                            sl_directos += 1
                                        else:
                                            sl_trailing += 1
                                            tsl += 1  # Trailing Stop Loss
                                    elif beneficio > 0:  # Ganancia
                                        if es_tp_real(trade):
                                            # TP real (tocó el TP)
                                            pass  # se contará abajo para mantener claridad
                                        else:
                                            # TS positivo (o cierre manual en ganancia que no tocó TP)
                                            sl_trailing += 1
                                            tsp += 1  # Trailing Stop Positivo
                                    else:
                                        # Break-even -> lo consideramos TS (sin ganancia ni pérdida, lo contamos como TSL)
                                        sl_trailing += 1
                                        tsl += 1
                            else:
                                # No hay pérdidas, no podemos calcular el riesgo
                                riesgo_ea = 0
                                sl_directos = 0
                                sl_trailing = 0
                                tsp = 0
                                tsl = 0
                                perdidas_nulas = 0
                            
                            # Calcular métricas completas
                            net_profit = ordenado['Beneficio'].sum()
                            ganancias_totales = ordenado[ordenado['Beneficio'] > 0]['Beneficio'].sum()
                            perdidas_totales = abs(ordenado[ordenado['Beneficio'] < 0]['Beneficio'].sum())
                                
                            # Profit Factor
                            profit_factor = ganancias_totales / perdidas_totales if perdidas_totales > 0 else float('inf')
                                
                            # Max Drawdown
                            max_dd = calcular_max_drawdown(ordenado['Beneficio'])
                                
                            # Return on Drawdown (ratio, no porcentaje)
                            ret_dd = net_profit / abs(max_dd) if max_dd != 0 else 0
                                
                            # Max Consecutive Loss
                            max_consec_loss = calcular_max_consecutive_loss(ordenado['Beneficio'])
                                
                            # Avg Trades per Month
                            avg_trades_mes = calcular_avg_trades_por_mes(ordenado)
                                
                            # Contar TP reales (cierre exactamente en TP, no TS positivo)
                            # Si el HTML tiene [tp], confiamos en eso directamente
                            if 'TipoCierre' in ordenado.columns:
                                # Contar todos los trades con TipoCierre == 'TP' como TP (confiamos en el HTML)
                                tp_explicitos = int((ordenado['TipoCierre'] == 'TP').sum())
                                # Para los que no tienen TipoCierre explícito, usar es_tp_real
                                sin_tipo_explicito = ordenado[ordenado['TipoCierre'] != 'TP']
                                tp_por_calculo = int(sin_tipo_explicito.apply(es_tp_real, axis=1).sum()) if len(sin_tipo_explicito) > 0 else 0
                                tp_trades = tp_explicitos + tp_por_calculo
                            else:
                                tp_trades = int(ordenado.apply(es_tp_real, axis=1).sum())
                                
                            # Calcular total de trades
                            total_trades = len(ordenado)
                            
                            # Calcular métricas mensuales
                            ordenado['Mes'] = ordenado['Open'].dt.to_period('M')
                            beneficios_mensuales = ordenado.groupby('Mes')['Beneficio'].sum()
                            
                            total_meses = len(beneficios_mensuales)
                            meses_rentables = len(beneficios_mensuales[beneficios_mensuales >= 0])
                            
                            # Calcular % meses rentables
                            pct_meses_rentables = (meses_rentables / total_meses * 100) if total_meses > 0 else 0
                                
                            analisis_data.append({
                                "Nombre": ea,
                                "Activo": simbolo.upper(),  # Agregar columna de activo
                                "retDD": f"{ret_dd:.2f}",
                                "Net Profit": net_profit,  # Mantener numérico para ordenamiento correcto
                                "maxDD": max_dd,  # Mantener numérico para ordenamiento correcto
                                "PF": f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞",
                                "Total Trades": int(total_trades),
                                "TP": int(tp_trades),
                                "SL": int(sl_directos),
                                "TSP": int(tsp),  # Trailing Stop Positivo
                                "TSL": int(tsl),  # Trailing Stop Loss
                                "Max Consec Loss": int(max_consec_loss),
                                "Avg Trade Mensual": round(avg_trades_mes, 1),
                                "% Meses Rentables": round(pct_meses_rentables, 1)
                            })
                        except Exception as e:
                            st.error(f"Error procesando estrategia {ea} - {simbolo}: {str(e)}")
                            import traceback
                            st.error(f"Traceback: {traceback.format_exc()}")
                            continue
                    
                    # Crear DataFrame del análisis
                    df_analisis = pd.DataFrame(analisis_data)
                    
                    # Verificar que hay datos antes de procesar
                    if df_analisis.empty:
                        st.warning("⚠️ No se encontraron estrategias válidas en los archivos. Verifica que los archivos HTML contengan trades con nombres de EA válidos.")
                    else:
                        # Ordenar por retDD de forma descendente
                        if 'retDD' in df_analisis.columns:
                            df_analisis['retDD_num'] = df_analisis['retDD'].str.replace('∞', '999999').astype(float)
                            df_analisis = df_analisis.sort_values(by='retDD_num', ascending=False)
                            df_analisis = df_analisis.drop(columns=['retDD_num'])
                    
                    # Mostrar tabla de análisis
                    column_config_analisis = {
                        "Nombre": st.column_config.TextColumn(
                                "Nombre",
                                help="Nombre del Expert Advisor"
                            ),
                        "Activo": st.column_config.TextColumn(
                                "Activo",
                                help="Símbolo del activo que opera la estrategia (ej: NAS100.FS, XAUUSD.PRO)"
                            ),
                            "retDD": st.column_config.TextColumn(
                                "retDD",
                                help="Return on Drawdown - Ratio: Net Profit / maxDD. Ej: 20 = beneficio es 20x el drawdown"
                            ),
                            "Net Profit": st.column_config.NumberColumn(
                                "Net Profit",
                                help="Beneficio neto total",
                                format="$%.2f"
                            ),
                            "maxDD": st.column_config.NumberColumn(
                                "maxDD",
                                help="Máximo drawdown (desde el pico más alto)",
                                format="$%.2f"
                            ),
                            "PF": st.column_config.TextColumn(
                                "PF",
                                help="Profit Factor - Ratio ganancias totales / pérdidas totales"
                            ),
                            "Total Trades": st.column_config.NumberColumn(
                                "Total Trades",
                                help="Total de trades realizados",
                                format="%d"
                            ),
                            "TP": st.column_config.NumberColumn(
                                "TP",
                                help="Trades con TP (Take Profit)",
                                format="%d"
                            ),
                            "SL": st.column_config.NumberColumn(
                                "SL",
                                help="Trades con SL directo",
                                format="%d"
                            ),
                            "TSP": st.column_config.NumberColumn(
                                "TSP",
                                help="Trailing Stop Positivo (trades con ganancia cerrados antes del TP)",
                                format="%d"
                            ),
                            "TSL": st.column_config.NumberColumn(
                                "TSL",
                                help="Trailing Stop Loss (trades con pérdida menor al SL directo)",
                                format="%d"
                            ),
                            "Max Consec Loss": st.column_config.NumberColumn(
                                "Max Consec Loss",
                                help="Máxima racha de pérdidas consecutivas",
                                format="%d"
                            ),
                            "Avg Trade Mensual": st.column_config.NumberColumn(
                                "Avg Trade Mensual",
                                help="Promedio de trades por mes",
                                format="%.1f"
                            ),
                            "% Meses Rentables": st.column_config.NumberColumn(
                                "% Meses Rentables",
                                help="Porcentaje de meses con beneficio >= 0",
                                format="%.1f%%"
                            )
                    }
                    
                    if not df_analisis.empty and 'retDD' in df_analisis.columns:
                        st.dataframe(
                        df_analisis,
                        use_container_width=True,
                        column_config=column_config_analisis,
                        hide_index=True
                        )
                    else:
                        st.warning("⚠️ No se pudo generar la tabla de análisis. Verifica que los archivos contengan datos válidos.")
                        
                    # Selector de estrategias para combinar
                    # Obtener el rango de fechas disponible del dataframe completo
                    fecha_min_disponible = df['Close'].min().date() if 'Close' in df.columns and len(df) > 0 else None
                    fecha_max_disponible = df['Close'].max().date() if 'Close' in df.columns and len(df) > 0 else None
                    
                    # Crear columnas para título y selector de fechas
                    col_titulo, col_fechas = st.columns([2, 3])
                    
                    with col_titulo:
                        st.markdown("""
                            <div>
                                <h4>Combinar Estrategias</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    with col_fechas:
                        # Selector de rango de fechas (opcional)
                        if fecha_min_disponible and fecha_max_disponible:
                            # Checkbox para activar/desactivar el filtro de fechas
                            filtrar_por_fecha = st.checkbox(
                                "Filtrar por rango de fechas",
                                value=False,
                                help="Activa esta opción para filtrar las estadísticas por un rango de fechas"
                            )
                            
                            if filtrar_por_fecha:
                                fecha_seleccionada = st.date_input(
                                    "Selecciona el rango de fechas:",
                                    value=(fecha_min_disponible, fecha_max_disponible),
                                    min_value=fecha_min_disponible,
                                    max_value=fecha_max_disponible,
                                )
                                # Manejar el caso cuando date_input devuelve None o una tupla
                                if fecha_seleccionada is None:
                                    fecha_inicio = None
                                    fecha_fin = None
                                elif isinstance(fecha_seleccionada, tuple) and len(fecha_seleccionada) == 2:
                                    fecha_inicio, fecha_fin = fecha_seleccionada
                                elif isinstance(fecha_seleccionada, tuple) and len(fecha_seleccionada) == 1:
                                    # Solo se seleccionó una fecha
                                    fecha_inicio = fecha_seleccionada[0]
                                    fecha_fin = fecha_seleccionada[0]
                                else:
                                    # Si solo se selecciona una fecha (no tupla)
                                    fecha_inicio = fecha_seleccionada
                                    fecha_fin = fecha_seleccionada
                            else:
                                fecha_inicio = None
                                fecha_fin = None
                        else:
                            fecha_inicio = None
                            fecha_fin = None
                            st.info("No hay fechas disponibles en los datos")
                    
                    # Lista de estrategias con formato "Nombre - Activo" para el multiselect
                    df_analisis['Estrategia_Completa'] = df_analisis['Nombre'] + ' - ' + df_analisis['Activo']
                    nombres_estrategias = df_analisis['Estrategia_Completa'].tolist()
                    
                    estrategias_seleccionadas = st.multiselect(
                        "Selecciona las estrategias que deseas combinar:",
                        options=nombres_estrategias,
                        default=[],
                        help="Selecciona una o más estrategias para ver sus estadísticas combinadas"
                    )
                    
                    if estrategias_seleccionadas:
                        if st.button("Ver Estadísticas Combinadas", use_container_width=True):
                            try:
                                # Filtrar# Filtrar el dataframe original por las estrategias seleccionadas
                                # Extraer EA y Símbolo de las estrategias seleccionadas
                                estrategias_filtradas = []
                                for estrategia_sel in estrategias_seleccionadas:
                                    # Formato: "Nombre - Activo"
                                    partes = estrategia_sel.split(' - ')
                                    if len(partes) == 2:
                                        ea_nombre = partes[0]
                                        activo = partes[1].lower()
                                        estrategias_filtradas.append((ea_nombre, activo))
                                
                                if not estrategias_filtradas:
                                    st.error("No se pudieron extraer las estrategias seleccionadas. Verifica el formato.")
                                    st.stop()
                                
                                # Filtrar el dataframe
                                mask = pd.Series([False] * len(df), index=df.index)
                                for ea_nombre, activo in estrategias_filtradas:
                                    mask |= ((df['EA'] == ea_nombre) & (df['Símbolo'].str.lower() == activo))
                                
                                df_combinado = df[mask].copy()
                                
                                if df_combinado.empty:
                                    st.warning("No hay datos disponibles para las estrategias seleccionadas.")
                                    st.stop()
                                
                                # Filtrar por rango de fechas solo si se seleccionó un rango
                                if fecha_inicio is not None and fecha_fin is not None:
                                    # Convertir fechas a datetime para comparación
                                    if isinstance(fecha_inicio, tuple):
                                        fecha_inicio = fecha_inicio[0]
                                    if isinstance(fecha_fin, tuple):
                                        fecha_fin = fecha_fin[1] if len(fecha_fin) > 1 else fecha_fin[0]
                                    
                                    # Filtrar por rango de fechas usando la columna Close
                                    mask_fecha = (df_combinado['Close'].dt.date >= fecha_inicio) & (df_combinado['Close'].dt.date <= fecha_fin)
                                    df_combinado = df_combinado[mask_fecha].copy()
                                    
                                    # Verificar que hay datos después del filtro
                                    if len(df_combinado) == 0:
                                        st.warning("No hay datos disponibles para el rango de fechas seleccionado con las estrategias elegidas.")
                                    else:
                                        df_combinado = df_combinado.sort_values(by='Open')
                                        
                                        # Calcular métricas combinadas
                                        # Detectar riesgo típico (promedio de los riesgos de las estrategias seleccionadas)
                                        riesgo_combinado = 0
                                        sl_directos_combinado = 0
                                        sl_trailing_combinado = 0
                                        
                                        # Detectar todos los niveles de SL de todas las EAs combinadas
                                        todas_perdidas = df_combinado[df_combinado['Beneficio'] < 0]['Beneficio'].abs()
                                        if len(todas_perdidas) > 0:
                                            niveles_sl_combinados = detectar_niveles_sl(todas_perdidas, min_trades_por_nivel=2, tolerancia_agrupacion=0.15, umbral_min_sl=100)
                                            if len(niveles_sl_combinados) == 0:
                                                perdida_max = todas_perdidas.max()
                                                margen = perdida_max * 0.10
                                                sl_detected = todas_perdidas[todas_perdidas >= (perdida_max - margen)].mode()
                                                if len(sl_detected) > 0:
                                                    niveles_sl_combinados = [sl_detected.iloc[0]]
                                                else:
                                                    niveles_sl_combinados = [perdida_max]
                                            riesgo_combinado = niveles_sl_combinados[0] if len(niveles_sl_combinados) > 0 else todas_perdidas.max()
                                            
                                            # Función helper para verificar si una pérdida es un SL directo
                                            def es_sl_directo_comb(perdida_abs):
                                                """
                                                Verifica si una pérdida es un SL directo:
                                                - Si es mayor o igual al nivel principal (si solo hay un nivel, con pequeño margen hacia abajo)
                                                - Si hay múltiples niveles (alta volatilidad), también verifica si está cerca de alguno
                                                """
                                                if len(niveles_sl_combinados) == 0:
                                                    return False
                                                
                                                # Nivel principal es el primero (el más común y alto)
                                                nivel_principal = niveles_sl_combinados[0]
                                                
                                                # Si solo hay un nivel (no hay alta volatilidad), usar umbral con pequeño margen
                                                if len(niveles_sl_combinados) == 1:
                                                    # Aplicar un pequeño margen hacia abajo (2.5% o mínimo $5 menos)
                                                    umbral_ajustado = max(nivel_principal * 0.975, nivel_principal - 5)
                                                    # Si la pérdida es muy cercana al umbral (dentro de $5), también considerarla SL
                                                    if perdida_abs >= umbral_ajustado:
                                                        return True
                                                    # Si está muy cerca del umbral principal (dentro de $5), también es SL
                                                    if abs(perdida_abs - nivel_principal) <= 5:
                                                        return True
                                                    return False
                                                
                                                # Si hay múltiples niveles (alta volatilidad):
                                                # 1. Si es mayor o igual al nivel principal, es SL directo
                                                if perdida_abs >= nivel_principal:
                                                    return True
                                                
                                                # 2. Si está cerca de alguno de los niveles detectados (dentro de tolerancia)
                                                for nivel in niveles_sl_combinados:
                                                    tolerancia_nivel = nivel * 0.15
                                                    if abs(perdida_abs - nivel) <= tolerancia_nivel:
                                                        return True
                                                
                                                return False
                                        else:
                                            riesgo_combinado = 0
                                            niveles_sl_combinados = []
                                            def es_sl_directo_comb(perdida_abs):
                                                return False
                                        
                                        # Calcular SL directos y trailing stops para el conjunto combinado
                                        margen_tolerancia_comb = riesgo_combinado * 0.15 if riesgo_combinado > 0 else 0
                                        sl_directos_combinado = 0  # Inicializar contador de SL directos
                                        sl_trailing_combinado = 0  # Inicializar contador de trailing stops
                                        tsp_combinado = 0  # Trailing Stop Positivo
                                        tsl_combinado = 0  # Trailing Stop Loss
                                        
                                        for _, trade in df_combinado.iterrows():
                                            beneficio = trade['Beneficio']
                                            tipo_cierre = trade.get('TipoCierre')
                                            
                                            # Si el HTML tiene información explícita de [sl] o [tp], usarla PERO validar con el beneficio
                                            if tipo_cierre == 'SL':
                                                if beneficio < 0:
                                                    # [sl] con pérdida: verificar si es SL directo o TSL
                                                    perdida_abs = abs(beneficio)
                                                    if es_sl_directo_comb(perdida_abs):
                                                        sl_directos_combinado += 1
                                                    else:
                                                        sl_trailing_combinado += 1
                                                        tsl_combinado += 1  # Trailing Stop Loss
                                                elif beneficio > 0:
                                                    sl_trailing_combinado += 1
                                                    tsp_combinado += 1  # Trailing Stop Positivo
                                                else:
                                                    sl_trailing_combinado += 1
                                                    tsl_combinado += 1  # Break-even con [sl]
                                            elif tipo_cierre == 'TP':
                                                pass  # TP real se contabiliza aparte
                                            else:
                                                # Si no hay información explícita, usar la lógica de cálculo
                                                if beneficio < 0:
                                                    perdida_abs = abs(beneficio)
                                                    if es_sl_directo_comb(perdida_abs):
                                                        sl_directos_combinado += 1
                                                    else:
                                                        sl_trailing_combinado += 1
                                                        tsl_combinado += 1  # Trailing Stop Loss
                                                elif beneficio > 0:
                                                    if es_tp_real(trade):
                                                        pass  # TP real se contabiliza aparte
                                                    else:
                                                        sl_trailing_combinado += 1
                                                        tsp_combinado += 1  # Trailing Stop Positivo
                                                else:
                                                    sl_trailing_combinado += 1
                                                    tsl_combinado += 1  # Break-even como TSL
                                        
                                        # Calcular métricas combinadas
                                        net_profit_comb = df_combinado['Beneficio'].sum()
                                        ganancias_totales_comb = df_combinado[df_combinado['Beneficio'] > 0]['Beneficio'].sum()
                                        perdidas_totales_comb = abs(df_combinado[df_combinado['Beneficio'] < 0]['Beneficio'].sum())
                                        
                                        profit_factor_comb = ganancias_totales_comb / perdidas_totales_comb if perdidas_totales_comb > 0 else float('inf')
                                        max_dd_comb = calcular_max_drawdown(df_combinado['Beneficio'])
                                        ret_dd_comb = net_profit_comb / abs(max_dd_comb) if max_dd_comb != 0 else 0
                                        max_consec_loss_comb = calcular_max_consecutive_loss(df_combinado['Beneficio'])
                                        avg_trades_mes_comb = calcular_avg_trades_por_mes(df_combinado)
                                        # TP reales combinados
                                        tp_trades_comb = int(df_combinado.apply(es_tp_real, axis=1).sum())
                                        
                                        # Mostrar estadísticas combinadas
                                        st.markdown("""
                                        <div style="margin-top: 2rem; margin-bottom: 1rem;">
                                            <h4>Estadísticas Combinadas de las Estrategias Seleccionadas</h4>
                                            <p><strong>Estrategias:</strong> {}</p>
                                        </div>
                                        """.format(" | ".join(estrategias_seleccionadas)), unsafe_allow_html=True)
                                        
                                        # Calcular total de trades
                                        total_trades_comb = len(df_combinado)
                                        
                                        stats_combinadas = {
                                            "retDD": f"{ret_dd_comb:.2f}",
                                            "Net Profit": net_profit_comb,  # Mantener numérico para ordenamiento correcto
                                            "maxDD": max_dd_comb,  # Mantener numérico para ordenamiento correcto
                                            "PF": f"{profit_factor_comb:.2f}" if profit_factor_comb != float('inf') else "∞",
                                            "Trades": int(total_trades_comb),
                                            "TP": int(tp_trades_comb),
                                            "SL": int(sl_directos_combinado),
                                            "TSP": int(tsp_combinado),  # Trailing Stop Positivo
                                            "TSL": int(tsl_combinado),  # Trailing Stop Loss
                                            "Max Consec Loss": int(max_consec_loss_comb),
                                            "Avg Trade Mensual": round(avg_trades_mes_comb, 1)
                                        }
                                        
                                        df_stats_comb = pd.DataFrame([stats_combinadas])
                                        
                                        # Configurar columnas sin "Nombre"
                                        column_config_comb = {
                                            "retDD": st.column_config.TextColumn(
                                                "retDD",
                                                help="Return on Drawdown - Ratio: Net Profit / maxDD. Ej: 20 = beneficio es 20x el drawdown"
                                            ),
                                            "Net Profit": st.column_config.NumberColumn(
                                                "Net Profit",
                                                help="Beneficio neto total",
                                                format="$%.2f"
                                            ),
                                            "maxDD": st.column_config.NumberColumn(
                                                "maxDD",
                                                help="Máximo drawdown (desde el pico más alto)",
                                                format="$%.2f"
                                            ),
                                            "PF": st.column_config.TextColumn(
                                                "PF",
                                                help="Profit Factor - Ratio ganancias totales / pérdidas totales"
                                            ),
                                            "Trades": st.column_config.NumberColumn(
                                                "Trades",
                                                help="Total de trades realizados",
                                                format="%d"
                                            ),
                                            "TP": st.column_config.NumberColumn(
                                                "TP",
                                                help="Trades con TP (Take Profit)",
                                                format="%d"
                                            ),
                                            "SL": st.column_config.NumberColumn(
                                                "SL",
                                                help="Trades con SL directo",
                                                format="%d"
                                            ),
                                            "TSP": st.column_config.NumberColumn(
                                                "TSP",
                                                help="Trailing Stop Positivo (trades con ganancia cerrados antes del TP)",
                                                format="%d"
                                            ),
                                            "TSL": st.column_config.NumberColumn(
                                                "TSL",
                                                help="Trailing Stop Loss (trades con pérdida menor al SL directo)",
                                                format="%d"
                                            ),
                                            "Max Consec Loss": st.column_config.NumberColumn(
                                                "Max Consec Loss",
                                                help="Máxima racha de pérdidas consecutivas",
                                                format="%d"
                                            ),
                                            "Avg Trade Mensual": st.column_config.NumberColumn(
                                                "Avg Trade Mensual",
                                                help="Promedio de trades por mes",
                                                format="%.1f"
                                            )
                                        }
                                        
                                        st.dataframe(
                                            df_stats_comb,
                                            use_container_width=True,
                                            column_config=column_config_comb,
                                            hide_index=True
                                        )
                                        
                                        # Calcular resumen mensual combinado
                                        df_combinado['Mes'] = df_combinado['Open'].dt.to_period('M')
                                        resumen_mensual_comb = []
                                        
                                        # Agrupar por mes y ordenar cronológicamente
                                        grupos_mensuales = list(df_combinado.groupby('Mes'))
                                        grupos_mensuales.sort(key=lambda x: x[0])  # Ordenar por período
                                        
                                        for mes, grupo_mes in grupos_mensuales:
                                            ano = mes.year
                                            mes_num = mes.month
                                            meses_es = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                                                      'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
                                            mes_nombre = meses_es[mes_num - 1]
                                            
                                            # Calcular métricas del mes
                                            beneficio_mes = grupo_mes['Beneficio'].sum()
                                            total_trades = len(grupo_mes)
                                            
                                            # Calcular máximo drawdown del mes
                                            max_dd_mes = calcular_max_drawdown(grupo_mes['Beneficio'])
                                            
                                            # Contar SL, TS, TP, TSP, TSL
                                            sl_mes = 0
                                            ts_mes = 0
                                            tp_mes = 0
                                            tsp_mes = 0  # Trailing Stop Positivo
                                            tsl_mes = 0  # Trailing Stop Loss
                                            
                                            for _, trade in grupo_mes.iterrows():
                                                beneficio = trade['Beneficio']
                                                tipo_cierre = trade.get('TipoCierre')
                                                
                                                # Si el HTML tiene información explícita de [sl] o [tp], usarla PERO validar con el beneficio
                                                if tipo_cierre == 'SL':
                                                    # [sl] en HTML: verificar si es pérdida o ganancia
                                                    if beneficio < 0:
                                                        # [sl] con pérdida: verificar si es SL directo o TSL
                                                        perdida_abs = abs(beneficio)
                                                        if es_sl_directo_comb(perdida_abs):
                                                            # SL directo: pérdida que alcanzó alguno de los SL esperados
                                                            sl_mes += 1
                                                        else:
                                                            # TSL: pérdida menor que el SL esperado (trailing stop activó antes)
                                                            ts_mes += 1
                                                            tsl_mes += 1  # Trailing Stop Loss
                                                    elif beneficio > 0:
                                                        # [sl] en HTML pero con ganancia: es un Trailing Stop Positivo
                                                        ts_mes += 1
                                                        tsp_mes += 1  # Trailing Stop Positivo
                                                    else:
                                                        # Break-even con [sl]: TSL
                                                        ts_mes += 1
                                                        tsl_mes += 1
                                                elif tipo_cierre == 'TP':
                                                    # [tp] en HTML: confiamos en el HTML, es un TP real
                                                    tp_mes += 1
                                                else:
                                                    # Si no hay información explícita, usar la lógica de cálculo
                                                    if beneficio < 0:  # Es pérdida
                                                        perdida_abs = abs(beneficio)
                                                        if es_sl_directo_comb(perdida_abs):
                                                            sl_mes += 1
                                                        elif perdida_abs > 0:
                                                            ts_mes += 1
                                                            tsl_mes += 1  # Trailing Stop Loss
                                                        else:
                                                            # Beneficio == 0 pero es pérdida (break-even)
                                                            ts_mes += 1
                                                            tsl_mes += 1
                                                    else:  # Es ganancia o break-even positivo
                                                        if es_tp_real(trade):
                                                            tp_mes += 1
                                                        else:
                                                            ts_mes += 1
                                                            tsp_mes += 1  # Trailing Stop Positivo
                                            
                                            resumen_mensual_comb.append({
                                                "Año - Mes": f"{ano} - {mes_nombre}",
                                                "Beneficio": beneficio_mes,  # Mantener numérico para ordenamiento correcto
                                                "maxDD": max_dd_mes,  # Mantener numérico para ordenamiento correcto
                                                "Trades": total_trades,
                                                "TP": tp_mes,
                                                "SL": sl_mes,
                                                "TSP": tsp_mes,
                                                "TSL": tsl_mes
                                            })
                                        
                                        # Crear DataFrame (ya ordenado por el sort anterior)
                                        df_resumen_mes_comb = pd.DataFrame(resumen_mensual_comb)
                                        if not df_resumen_mes_comb.empty:
                                            
                                            # Función para estilizar el beneficio (ahora recibe valores numéricos)
                                            def estilizar_beneficio_comb(valor):
                                                if isinstance(valor, (int, float)):
                                                    color = '#90EE90' if valor >= 0 else '#FFB6C1'  # Verde claro si es ganancia, rojo claro si es pérdida
                                                    return f'background-color: {color}'
                                                return ''
                                            
                                            column_config_mes_comb = {
                                                "Año - Mes": st.column_config.TextColumn("Año - Mes", help="Año y mes"),
                                                "Beneficio": st.column_config.NumberColumn("Beneficio", help="Beneficio del mes", format="$%.2f"),
                                                "maxDD": st.column_config.NumberColumn("maxDD", help="Máximo drawdown del mes", format="$%.2f"),
                                                "Trades": st.column_config.NumberColumn("Trades", help="Total de operaciones", format="%d"),
                                                "TP": st.column_config.NumberColumn("TP", help="Trades con TP", format="%d"),
                                                "SL": st.column_config.NumberColumn("SL", help="Trades con SL directo", format="%d"),
                                                "TSP": st.column_config.NumberColumn("TSP", help="Trailing Stop Positivo", format="%d"),
                                                "TSL": st.column_config.NumberColumn("TSL", help="Trailing Stop Loss", format="%d")
                                            }
                                            
                                            st.dataframe(
                                                df_resumen_mes_comb.style.applymap(estilizar_beneficio_comb, subset=['Beneficio']),
                                                use_container_width=True,
                                                hide_index=True,
                                                column_config=column_config_mes_comb
                                            )
                                            
                                            # Análisis de correlación de precios de entrada por activo
                                            # Agrupar estrategias por activo
                                            estrategias_por_activo = {}
                                            for ea_nombre, activo in estrategias_filtradas:
                                                if activo not in estrategias_por_activo:
                                                    estrategias_por_activo[activo] = []
                                                estrategias_por_activo[activo].append(ea_nombre)
                                            
                                            # Mostrar matriz de correlación solo para activos con más de una estrategia
                                            activos_con_multiples = {activo: eas for activo, eas in estrategias_por_activo.items() if len(eas) > 1}
                                            
                                            if activos_con_multiples:
                                                st.markdown("""
                                                <div style="margin-top: 2rem; margin-bottom: 2.5rem;">
                                                    <h4>Matriz de Correlación de Timing y Dirección de Entrada</h4>
                                                    <p style="color: #6c757d; font-size: 0.9em;">Matriz de correlación entre estrategias del mismo activo basada en timing de entrada (hora del día, día de la semana) y dirección de operación (Buy/Sell). Valores cercanos a 1 indican que las estrategias tienen comportamientos similares (o inversos), valores cercanos a 0 indican comportamientos independientes.</p>
                                                    <p style="color: #6c757d; font-size: 0.85em; font-style: italic;">Nota: La correlación se calcula comparando el timing de entrada y la dirección de operación (dentro de 7 días), usando el valor absoluto para mostrar solo valores entre 0 y 1. "N/A" aparece cuando no hay suficientes trades coincidentes (mínimo 3 puntos).</p>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                
                                                # Función helper para calcular correlación entre dos estrategias
                                                def calcular_correlacion_estrategias(ea1, ea2, activo, df_data):
                                                    """Calcula la correlación basada en timing de entrada y dirección de operación"""
                                                    df_ea1 = df_data[
                                                        (df_data['EA'] == ea1) & 
                                                        (df_data['Símbolo'].str.lower() == activo)
                                                    ][['Open', 'Tipo']].copy()
                                                    df_ea2 = df_data[
                                                        (df_data['EA'] == ea2) & 
                                                        (df_data['Símbolo'].str.lower() == activo)
                                                    ][['Open', 'Tipo']].copy()
                                                    
                                                    if len(df_ea1) == 0 or len(df_ea2) == 0:
                                                        return np.nan
                                                    
                                                    # Calcular características de entrada para cada estrategia
                                                    # 1. Hora del día (0-23) normalizada a 0-1
                                                    # 2. Día de la semana (0-6) normalizado a 0-1
                                                    # 3. Dirección (1 para Buy, -1 para Sell)
                                                    
                                                    caracteristicas_ea1 = []
                                                    caracteristicas_ea2 = []
                                                    
                                                    # Ventana amplia para encontrar coincidencias
                                                    ventana = pd.Timedelta(days=7)  # Una semana completa
                                                    
                                                    for _, row1 in df_ea1.iterrows():
                                                        fecha1 = row1['Open']
                                                        df_cercanos = df_ea2[
                                                            (df_ea2['Open'] >= fecha1 - ventana) & 
                                                            (df_ea2['Open'] <= fecha1 + ventana)
                                                        ]
                                                        if len(df_cercanos) > 0:
                                                            # Encontrar el más cercano en tiempo
                                                            df_cercanos = df_cercanos.copy()
                                                            df_cercanos['diff'] = abs(df_cercanos['Open'] - fecha1)
                                                            mas_cercano = df_cercanos.nsmallest(1, 'diff').iloc[0]
                                                            
                                                            # Características de la estrategia 1
                                                            hora1 = fecha1.hour / 23.0  # Normalizar a 0-1
                                                            dia_semana1 = fecha1.weekday() / 6.0  # Normalizar a 0-1
                                                            tipo1 = 1.0 if row1['Tipo'] == 'Buy' else -1.0
                                                            
                                                            # Características de la estrategia 2
                                                            fecha2 = mas_cercano['Open']
                                                            hora2 = fecha2.hour / 23.0
                                                            dia_semana2 = fecha2.weekday() / 6.0
                                                            tipo2 = 1.0 if mas_cercano['Tipo'] == 'Buy' else -1.0
                                                            
                                                            # Crear vector de características (combinación ponderada)
                                                            # Peso mayor a la dirección y hora del día
                                                            caracteristica1 = hora1 * 0.4 + dia_semana1 * 0.2 + tipo1 * 0.4
                                                            caracteristica2 = hora2 * 0.4 + dia_semana2 * 0.2 + tipo2 * 0.4
                                                            
                                                            caracteristicas_ea1.append(caracteristica1)
                                                            caracteristicas_ea2.append(caracteristica2)
                                                    
                                                    # Calcular correlación de las características
                                                    if len(caracteristicas_ea1) >= 3:
                                                        try:
                                                            correlacion = np.corrcoef(caracteristicas_ea1, caracteristicas_ea2)[0, 1]
                                                            # Usar valor absoluto para que la correlación esté entre 0 y 1
                                                            # 0 = sin correlación, 1 = correlación perfecta (positiva o negativa)
                                                            if not np.isnan(correlacion):
                                                                return abs(correlacion)
                                                            return np.nan
                                                        except:
                                                            return np.nan
                                                    return np.nan
                                                
                                                for activo, eas in activos_con_multiples.items():
                                                    # Crear matriz de correlación
                                                    n_estrategias = len(eas)
                                                    matriz_correlacion = np.full((n_estrategias, n_estrategias), np.nan)
                                                    
                                                    # Calcular correlación para cada par de estrategias
                                                    for i in range(n_estrategias):
                                                        for j in range(n_estrategias):
                                                            if i == j:
                                                                # Diagonal: correlación consigo mismo es 1.0
                                                                matriz_correlacion[i, j] = 1.0
                                                            elif i < j:
                                                                # Calcular correlación solo una vez para cada par
                                                                corr = calcular_correlacion_estrategias(eas[i], eas[j], activo, df_combinado)
                                                                matriz_correlacion[i, j] = corr
                                                                matriz_correlacion[j, i] = corr  # Matriz simétrica
                                                    
                                                    # Crear DataFrame para la matriz
                                                    df_matriz = pd.DataFrame(
                                                        matriz_correlacion,
                                                        index=eas,
                                                        columns=eas
                                                    )
                                                    
                                                    # Crear heatmap con plotly
                                                    # Preparar texto con mejor contraste
                                                    text_matrix = []
                                                    for row in matriz_correlacion:
                                                        text_row = []
                                                        for val in row:
                                                            if not np.isnan(val):
                                                                text_row.append(f"{val:.3f}")
                                                            else:
                                                                text_row.append("N/A")
                                                        text_matrix.append(text_row)
                                                    
                                                    # Escala de colores: < 0.4 verde, 0.4-0.7 naranja, > 0.7 rojo
                                                    colorscale_custom = [
                                                        [0,    '#48F94A'],   # Verde para valores < 0.4
                                                        [0.4,  '#48F94A'],   # Verde hasta 0.4
                                                        [0.4,  '#F98948'],   # Naranja desde 0.4
                                                        [0.7,  '#F98948'],   # Naranja hasta 0.7
                                                        [0.7,  '#F93636'],   # Rojo desde 0.7
                                                        [1,    '#F93636']    # Rojo para valores > 0.7
                                                    ]
                                                    
                                                    # Función para obtener el color de borde (todos los bordes del mismo color)
                                                    def obtener_color_borde(valor):
                                                        if pd.isna(valor):
                                                            return None
                                                        return '#d5d5da'  # Todos los bordes del mismo color
                                                    
                                                    # Añadir bordes de colores a cada celda usando shapes
                                                    shapes = []
                                                    n = len(eas)
                                                    for i in range(n):
                                                        for j in range(n):
                                                            valor = matriz_correlacion[i, j]
                                                            if not pd.isna(valor):
                                                                color_borde = obtener_color_borde(valor)
                                                                if color_borde:
                                                                    # Usar las etiquetas de las estrategias como coordenadas
                                                                    x0 = eas[j] if j > 0 else eas[0]
                                                                    x1 = eas[j] if j < n-1 else eas[n-1]
                                                                    y0 = eas[i] if i > 0 else eas[0]
                                                                    y1 = eas[i] if i < n-1 else eas[n-1]
                                                                    
                                                                    # Crear rectángulo para el borde usando coordenadas de categorías
                                                                    shapes.append(dict(
                                                                        type="rect",
                                                                        xref="x",
                                                                        yref="y",
                                                                        x0=j - 0.5,
                                                                        y0=i - 0.5,
                                                                        x1=j + 0.5,
                                                                        y1=i + 0.5,
                                                                        line=dict(color=color_borde, width=2),
                                                                        fillcolor="rgba(0,0,0,0)",  # Transparente, solo borde
                                                                        layer="above"
                                                                    ))
                                                    
                                                    fig_heatmap = go.Figure(data=go.Heatmap(
                                                        z=matriz_correlacion,
                                                        x=eas,
                                                        y=eas,
                                                        colorscale=colorscale_custom,
                                                        zmin=0,
                                                        zmax=1,
                                                        text=text_matrix,
                                                        texttemplate="%{text}",
                                                        textfont={"size": 13},  # Texto gris/negro por defecto (mejor contraste con fondos claros)
                                                        showscale=False,  # Ocultar la leyenda (colorbar)
                                                        xgap=2,  # Espacio horizontal entre celdas
                                                        ygap=2   # Espacio vertical entre celdas
                                                    ))
                                                    
                                                    # Crear título con leyenda de colores
                                                    titulo_con_leyenda = f"Matriz de Correlación - {activo.upper()}                    🟢 < 0.4    🟠 0.4-0.7    🔴 > 0.7"
                                                    
                                                    fig_heatmap.update_layout(
                                                        title=titulo_con_leyenda,
                                                        width=600,
                                                        height=600,
                                                        margin=dict(l=80, r=20, t=60, b=60),  # Margen izquierdo más amplio para separar del borde
                                                        plot_bgcolor='#f8f9fa',  # Fondo del área del gráfico
                                                        paper_bgcolor='#f8f9fa',  # Fondo del papel/contenedor
                                                        xaxis=dict(showgrid=False, zeroline=False, showline=False),  # Quitar líneas y bordes del eje X
                                                        yaxis=dict(showgrid=False, zeroline=False, showline=False),  # Quitar líneas y bordes del eje Y
                                                        shapes=shapes
                                                    )
                                                    
                                                    st.plotly_chart(fig_heatmap, use_container_width=True)
                                                    
                                                    # Mostrar también como tabla para mejor legibilidad
                                                    st.markdown(f"**Tabla de Correlación - {activo.upper()}**")
                                                    
                                                    # Formatear valores para la tabla
                                                    df_matriz_display = df_matriz.copy()
                                                    for col in df_matriz_display.columns:
                                                        df_matriz_display[col] = df_matriz_display[col].apply(
                                                            lambda x: f"{x:.3f}" if not np.isnan(x) else "N/A"
                                                        )
                                                    
                                                    # Configurar columnas para la tabla
                                                    column_config_matriz = {}
                                                    for col in df_matriz_display.columns:
                                                        column_config_matriz[col] = st.column_config.TextColumn(
                                                            col,
                                                            help=f"Correlación con {col}"
                                                        )
                                                    
                                                    st.dataframe(
                                                        df_matriz_display,
                                                        use_container_width=True,
                                                        column_config=column_config_matriz,
                                                        hide_index=False
                                                    )
                                            
                                            # Preparar datos de cada estrategia individual
                                            df_combinado['Fecha'] = df_combinado['Close'].dt.date
                                            
                                            fig_comb = go.Figure()
                                            
                                            # Agregar línea para cada estrategia individual (EA + Símbolo)
                                            for ea_nombre, activo in estrategias_filtradas:
                                                df_ea = df_combinado[(df_combinado['EA'] == ea_nombre) & (df_combinado['Símbolo'].str.lower() == activo)].copy()
                                                if len(df_ea) > 0:
                                                    df_ea['Fecha'] = df_ea['Close'].dt.date
                                                    df_ea = df_ea.sort_values('Fecha')
                                                    beneficios_ea = df_ea.groupby('Fecha')['Beneficio'].sum().reset_index()
                                                    beneficios_ea = beneficios_ea.sort_values('Fecha')
                                                    beneficios_ea['Beneficio_acumulado'] = beneficios_ea['Beneficio'].cumsum()
                                                    
                                                    nombre_leyenda = f"{ea_nombre} - {activo.upper()}"
                                                    fig_comb.add_trace(go.Scatter(
                                                        x=beneficios_ea['Fecha'],
                                                        y=beneficios_ea['Beneficio_acumulado'],
                                                        mode='lines+markers',
                                                        name=nombre_leyenda,
                                                        line=dict(width=2),
                                                        marker=dict(size=4)
                                                    ))
                                            
                                            # Calcular y agregar línea del conjunto combinado
                                            beneficios_combinados = df_combinado.groupby('Fecha')['Beneficio'].sum().reset_index()
                                            beneficios_combinados = beneficios_combinados.sort_values('Fecha')
                                            beneficios_combinados['Beneficio_acumulado'] = beneficios_combinados['Beneficio'].cumsum()
                                            
                                            fig_comb.add_trace(go.Scatter(
                                                x=beneficios_combinados['Fecha'],
                                                y=beneficios_combinados['Beneficio_acumulado'],
                                                mode='lines+markers',
                                                name='Combinado',
                                                line=dict(width=3, color='#FF6B6B', dash='dash'),
                                                marker=dict(size=5)
                                            ))
                                            
                                            fig_comb.update_layout(
                                                xaxis_title="Fecha",
                                                yaxis_title="Beneficio Acumulado ($)",
                                                hovermode='x unified',
                                                showlegend=True,
                                                legend=dict(
                                                    orientation="h",
                                                    yanchor="bottom",
                                                    y=1.02,
                                                    xanchor="right",
                                                    x=1
                                                )
                                            )
                                            
                                            st.plotly_chart(fig_comb, use_container_width=True)
                                else:
                                    # No se seleccionó rango de fechas, usar todos los datos
                                    df_combinado = df_combinado.sort_values(by='Open')
                                    
                                    # Verificar que hay datos y calcular métricas combinadas
                                    if len(df_combinado) > 0:
                                        # Detectar riesgo típico (promedio de los riesgos de las estrategias seleccionadas)
                                        riesgo_combinado = 0
                                        sl_directos_combinado = 0
                                        sl_trailing_combinado = 0
                                        
                                        # Detectar todos los niveles de SL de todas las EAs combinadas
                                        todas_perdidas = df_combinado[df_combinado['Beneficio'] < 0]['Beneficio'].abs()
                                        if len(todas_perdidas) > 0:
                                            niveles_sl_combinados = detectar_niveles_sl(todas_perdidas, min_trades_por_nivel=2, tolerancia_agrupacion=0.15, umbral_min_sl=100)
                                            if len(niveles_sl_combinados) == 0:
                                                perdida_max = todas_perdidas.max()
                                                margen = perdida_max * 0.10
                                                sl_detected = todas_perdidas[todas_perdidas >= (perdida_max - margen)].mode()
                                                if len(sl_detected) > 0:
                                                    niveles_sl_combinados = [sl_detected.iloc[0]]
                                                else:
                                                    niveles_sl_combinados = [perdida_max]
                                            riesgo_combinado = niveles_sl_combinados[0] if len(niveles_sl_combinados) > 0 else todas_perdidas.max()
                                            
                                            # Función helper para verificar si una pérdida es un SL directo
                                            def es_sl_directo_comb(perdida_abs):
                                                """
                                                Verifica si una pérdida es un SL directo:
                                                - Si es mayor o igual al nivel principal (si solo hay un nivel, con pequeño margen hacia abajo)
                                                - Si hay múltiples niveles (alta volatilidad), también verifica si está cerca de alguno
                                                """
                                                if len(niveles_sl_combinados) == 0:
                                                    return False
                                                
                                                # Nivel principal es el primero (el más común y alto)
                                                nivel_principal = niveles_sl_combinados[0]
                                                
                                                # Si solo hay un nivel (no hay alta volatilidad), usar umbral con pequeño margen
                                                if len(niveles_sl_combinados) == 1:
                                                    # Aplicar un pequeño margen hacia abajo (2.5% o mínimo $5 menos)
                                                    umbral_ajustado = max(nivel_principal * 0.975, nivel_principal - 5)
                                                    # Si la pérdida es muy cercana al umbral (dentro de $5), también considerarla SL
                                                    if perdida_abs >= umbral_ajustado:
                                                        return True
                                                    # Si está muy cerca del umbral principal (dentro de $5), también es SL
                                                    if abs(perdida_abs - nivel_principal) <= 5:
                                                        return True
                                                    return False
                                                
                                                # Si hay múltiples niveles (alta volatilidad):
                                                # 1. Si es mayor o igual al nivel principal, es SL directo
                                                if perdida_abs >= nivel_principal:
                                                    return True
                                                
                                                # 2. Si está cerca de alguno de los niveles detectados (dentro de tolerancia)
                                                for nivel in niveles_sl_combinados:
                                                    tolerancia_nivel = nivel * 0.15
                                                    if abs(perdida_abs - nivel) <= tolerancia_nivel:
                                                        return True
                                                
                                                return False
                                        else:
                                            riesgo_combinado = 0
                                            niveles_sl_combinados = []
                                            def es_sl_directo_comb(perdida_abs):
                                                return False
                                        
                                        # Calcular SL directos y trailing stops para el conjunto combinado
                                        margen_tolerancia_comb = riesgo_combinado * 0.15 if riesgo_combinado > 0 else 0
                                        sl_directos_combinado = 0  # Inicializar contador de SL directos
                                        sl_trailing_combinado = 0  # Inicializar contador de trailing stops
                                        tsp_combinado = 0  # Trailing Stop Positivo
                                        tsl_combinado = 0  # Trailing Stop Loss
                                        
                                        for _, trade in df_combinado.iterrows():
                                            beneficio = trade['Beneficio']
                                            tipo_cierre = trade.get('TipoCierre')
                                            
                                            # Si el HTML tiene información explícita de [sl] o [tp], usarla PERO validar con el beneficio
                                            if tipo_cierre == 'SL':
                                                if beneficio < 0:
                                                    # [sl] con pérdida: verificar si es SL directo o TSL
                                                    perdida_abs = abs(beneficio)
                                                    if es_sl_directo_comb(perdida_abs):
                                                        sl_directos_combinado += 1
                                                    else:
                                                        sl_trailing_combinado += 1
                                                        tsl_combinado += 1  # Trailing Stop Loss
                                                elif beneficio > 0:
                                                    sl_trailing_combinado += 1
                                                    tsp_combinado += 1  # Trailing Stop Positivo
                                                else:
                                                    sl_trailing_combinado += 1
                                                    tsl_combinado += 1  # Break-even con [sl]
                                            elif tipo_cierre == 'TP':
                                                pass  # TP real se contabiliza aparte
                                            else:
                                                # Si no hay información explícita, usar la lógica de cálculo
                                                if beneficio < 0:
                                                    perdida_abs = abs(beneficio)
                                                    if es_sl_directo_comb(perdida_abs):
                                                        sl_directos_combinado += 1
                                                    else:
                                                        sl_trailing_combinado += 1
                                                        tsl_combinado += 1  # Trailing Stop Loss
                                                elif beneficio > 0:
                                                    if es_tp_real(trade):
                                                        pass  # TP real se contabiliza aparte
                                                    else:
                                                        sl_trailing_combinado += 1
                                                        tsp_combinado += 1  # Trailing Stop Positivo
                                                else:
                                                    sl_trailing_combinado += 1
                                                    tsl_combinado += 1  # Break-even como TSL
                                        
                                        # Calcular métricas combinadas
                                        net_profit_comb = df_combinado['Beneficio'].sum()
                                        ganancias_totales_comb = df_combinado[df_combinado['Beneficio'] > 0]['Beneficio'].sum()
                                        perdidas_totales_comb = abs(df_combinado[df_combinado['Beneficio'] < 0]['Beneficio'].sum())
                                        
                                        profit_factor_comb = ganancias_totales_comb / perdidas_totales_comb if perdidas_totales_comb > 0 else float('inf')
                                        max_dd_comb = calcular_max_drawdown(df_combinado['Beneficio'])
                                        ret_dd_comb = net_profit_comb / abs(max_dd_comb) if max_dd_comb != 0 else 0
                                        max_consec_loss_comb = calcular_max_consecutive_loss(df_combinado['Beneficio'])
                                        avg_trades_mes_comb = calcular_avg_trades_por_mes(df_combinado)
                                        # TP reales combinados
                                        tp_trades_comb = int(df_combinado.apply(es_tp_real, axis=1).sum())
                                        
                                        # Mostrar estadísticas combinadas
                                        st.markdown("""
                                        <div style="margin-top: 2rem; margin-bottom: 1rem;">
                                            <h4>Estadísticas Combinadas de las Estrategias Seleccionadas</h4>
                                            <p><strong>Estrategias:</strong> {}</p>
                                        </div>
                                        """.format(" | ".join(estrategias_seleccionadas)), unsafe_allow_html=True)
                                        
                                        # Calcular total de trades
                                        total_trades_comb = len(df_combinado)
                                        
                                        stats_combinadas = {
                                            "retDD": f"{ret_dd_comb:.2f}",
                                            "Net Profit": net_profit_comb,  # Mantener numérico para ordenamiento correcto
                                            "maxDD": max_dd_comb,  # Mantener numérico para ordenamiento correcto
                                            "PF": f"{profit_factor_comb:.2f}" if profit_factor_comb != float('inf') else "∞",
                                            "Trades": int(total_trades_comb),
                                            "TP": int(tp_trades_comb),
                                            "SL": int(sl_directos_combinado),
                                            "TSP": int(tsp_combinado),  # Trailing Stop Positivo
                                            "TSL": int(tsl_combinado),  # Trailing Stop Loss
                                            "Max Consec Loss": int(max_consec_loss_comb),
                                            "Avg Trade Mensual": round(avg_trades_mes_comb, 1)
                                        }
                                        
                                        df_stats_comb = pd.DataFrame([stats_combinadas])
                                        
                                        # Configurar columnas sin "Nombre"
                                        column_config_comb = {
                                            "retDD": st.column_config.TextColumn(
                                                "retDD",
                                                help="Return on Drawdown - Ratio: Net Profit / maxDD. Ej: 20 = beneficio es 20x el drawdown"
                                            ),
                                            "Net Profit": st.column_config.NumberColumn(
                                                "Net Profit",
                                                help="Beneficio neto total",
                                                format="$%.2f"
                                            ),
                                            "maxDD": st.column_config.NumberColumn(
                                                "maxDD",
                                                help="Máximo drawdown (desde el pico más alto)",
                                                format="$%.2f"
                                            ),
                                            "PF": st.column_config.TextColumn(
                                                "PF",
                                                help="Profit Factor - Ratio ganancias totales / pérdidas totales"
                                            ),
                                            "Trades": st.column_config.NumberColumn(
                                                "Trades",
                                                help="Total de trades realizados",
                                    format="%d"
                                ),
                                "TP": st.column_config.NumberColumn(
                                    "TP",
                                    help="Trades con TP (Take Profit)",
                                    format="%d"
                                ),
                                            "SL": st.column_config.NumberColumn(
                                                "SL",
                                                help="Trades con SL directo",
                                                format="%d"
                                            ),
                                            "TSP": st.column_config.NumberColumn(
                                                "TSP",
                                                help="Trailing Stop Positivo (trades con ganancia cerrados antes del TP)",
                                                format="%d"
                                            ),
                                            "TSL": st.column_config.NumberColumn(
                                                "TSL",
                                                help="Trailing Stop Loss (trades con pérdida menor al SL directo)",
                                    format="%d"
                                ),
                                "Max Consec Loss": st.column_config.NumberColumn(
                                    "Max Consec Loss",
                                    help="Máxima racha de pérdidas consecutivas",
                                    format="%d"
                                ),
                                "Avg Trade Mensual": st.column_config.NumberColumn(
                                    "Avg Trade Mensual",
                                    help="Promedio de trades por mes",
                                    format="%.1f"
                            )
                            }
                                        
                                        st.dataframe(
                                            df_stats_comb,
                                            use_container_width=True,
                                            column_config=column_config_comb,
                                            hide_index=True
                                        )
                                        
                                        # Calcular resumen mensual combinado
                                        df_combinado['Mes'] = df_combinado['Open'].dt.to_period('M')
                                        resumen_mensual_comb = []
                                        
                                        # Agrupar por mes y ordenar cronológicamente
                                        grupos_mensuales = list(df_combinado.groupby('Mes'))
                                        grupos_mensuales.sort(key=lambda x: x[0])  # Ordenar por período
                                        
                                        for mes, grupo_mes in grupos_mensuales:
                                            ano = mes.year
                                            mes_num = mes.month
                                            meses_es = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                                                      'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
                                            mes_nombre = meses_es[mes_num - 1]
                                            
                                            # Calcular métricas del mes
                                            beneficio_mes = grupo_mes['Beneficio'].sum()
                                            total_trades = len(grupo_mes)
                                            
                                            # Calcular máximo drawdown del mes
                                            max_dd_mes = calcular_max_drawdown(grupo_mes['Beneficio'])
                                            
                                            # Contar SL, TS, TP, TSP, TSL
                                            sl_mes = 0
                                            ts_mes = 0
                                            tp_mes = 0
                                            tsp_mes = 0  # Trailing Stop Positivo
                                            tsl_mes = 0  # Trailing Stop Loss
                                            
                                            for _, trade in grupo_mes.iterrows():
                                                beneficio = trade['Beneficio']
                                                tipo_cierre = trade.get('TipoCierre')
                                                
                                                # Si el HTML tiene información explícita de [sl] o [tp], usarla PERO validar con el beneficio
                                                if tipo_cierre == 'SL':
                                                    # [sl] en HTML: verificar si es pérdida o ganancia
                                                    if beneficio < 0:
                                                        # [sl] con pérdida: verificar si es SL directo o TSL
                                                        perdida_abs = abs(beneficio)
                                                        if es_sl_directo_comb(perdida_abs):
                                                            # SL directo: pérdida que alcanzó alguno de los SL esperados
                                                            sl_mes += 1
                                                        else:
                                                            # TSL: pérdida menor que el SL esperado (trailing stop activó antes)
                                                            ts_mes += 1
                                                            tsl_mes += 1  # Trailing Stop Loss
                                                    elif beneficio > 0:
                                                        # [sl] en HTML pero con ganancia: es un Trailing Stop Positivo
                                                        ts_mes += 1
                                                        tsp_mes += 1  # Trailing Stop Positivo
                                                    else:
                                                        # Break-even con [sl]: TSL
                                                        ts_mes += 1
                                                        tsl_mes += 1
                                                elif tipo_cierre == 'TP':
                                                    # [tp] en HTML: confiamos en el HTML, es un TP real
                                                    tp_mes += 1
                                                else:
                                                    # Si no hay información explícita, usar la lógica de cálculo
                                                    if beneficio < 0:  # Es pérdida
                                                        perdida_abs = abs(beneficio)
                                                        if es_sl_directo_comb(perdida_abs):
                                                            sl_mes += 1
                                                        elif perdida_abs > 0:
                                                            ts_mes += 1
                                                            tsl_mes += 1  # Trailing Stop Loss
                                                        else:
                                                            # Beneficio == 0 pero es pérdida (break-even)
                                                            ts_mes += 1
                                                            tsl_mes += 1
                                                    else:  # Es ganancia o break-even positivo
                                                        if es_tp_real(trade):
                                                            tp_mes += 1
                                                        else:
                                                            ts_mes += 1
                                                            tsp_mes += 1  # Trailing Stop Positivo
                                            
                                            resumen_mensual_comb.append({
                                                "Año - Mes": f"{ano} - {mes_nombre}",
                                                "Beneficio": beneficio_mes,  # Mantener numérico para ordenamiento correcto
                                                "maxDD": max_dd_mes,  # Mantener numérico para ordenamiento correcto
                                                "Trades": total_trades,
                                                "TP": tp_mes,
                                                "SL": sl_mes,
                                                "TSP": tsp_mes,
                                                "TSL": tsl_mes
                                            })
                                        
                                        # Crear DataFrame (ya ordenado por el sort anterior)
                                        df_resumen_mes_comb = pd.DataFrame(resumen_mensual_comb)
                                        if not df_resumen_mes_comb.empty:
                                            
                                            # Función para estilizar el beneficio (ahora recibe valores numéricos)
                                            def estilizar_beneficio_comb(valor):
                                                if isinstance(valor, (int, float)):
                                                    color = '#90EE90' if valor >= 0 else '#FFB6C1'  # Verde claro si es ganancia, rojo claro si es pérdida
                                                    return f'background-color: {color}'
                                                return ''
                                            
                                            column_config_mes_comb = {
                                                "Año - Mes": st.column_config.TextColumn("Año - Mes", help="Año y mes"),
                                                "Beneficio": st.column_config.NumberColumn("Beneficio", help="Beneficio del mes", format="$%.2f"),
                                                "maxDD": st.column_config.NumberColumn("maxDD", help="Máximo drawdown del mes", format="$%.2f"),
                                                "Trades": st.column_config.NumberColumn("Trades", help="Total de operaciones", format="%d"),
                                                "TP": st.column_config.NumberColumn("TP", help="Trades con TP", format="%d"),
                                                "SL": st.column_config.NumberColumn("SL", help="Trades con SL directo", format="%d"),
                                                "TSP": st.column_config.NumberColumn("TSP", help="Trailing Stop Positivo", format="%d"),
                                                "TSL": st.column_config.NumberColumn("TSL", help="Trailing Stop Loss", format="%d")
                                            }
                                            
                                            st.dataframe(
                                                df_resumen_mes_comb.style.applymap(estilizar_beneficio_comb, subset=['Beneficio']),
                                                use_container_width=True,
                                                hide_index=True,
                                                column_config=column_config_mes_comb
                                            )
                                            
                                            # Análisis de correlación de precios de entrada por activo
                                            # Agrupar estrategias por activo
                                            estrategias_por_activo = {}
                                            for ea_nombre, activo in estrategias_filtradas:
                                                if activo not in estrategias_por_activo:
                                                    estrategias_por_activo[activo] = []
                                                estrategias_por_activo[activo].append(ea_nombre)
                                            
                                            # Mostrar matriz de correlación solo para activos con más de una estrategia
                                            activos_con_multiples = {activo: eas for activo, eas in estrategias_por_activo.items() if len(eas) > 1}
                                            
                                            if activos_con_multiples:
                                                st.markdown("""
                                                <div style="margin-top: 2rem; margin-bottom: 2.5rem;">
                                                    <h4>Matriz de Correlación de Timing y Dirección de Entrada</h4>
                                                    <p style="color: #6c757d; font-size: 0.9em;">Matriz de correlación entre estrategias del mismo activo basada en timing de entrada (hora del día, día de la semana) y dirección de operación (Buy/Sell). Valores cercanos a 1 indican que las estrategias tienen comportamientos similares (o inversos), valores cercanos a 0 indican comportamientos independientes.</p>
                                                    <p style="color: #6c757d; font-size: 0.85em; font-style: italic;">Nota: La correlación se calcula comparando el timing de entrada y la dirección de operación (dentro de 7 días), usando el valor absoluto para mostrar solo valores entre 0 y 1. "N/A" aparece cuando no hay suficientes trades coincidentes (mínimo 3 puntos).</p>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                
                                                # Función helper para calcular correlación entre dos estrategias
                                                def calcular_correlacion_estrategias(ea1, ea2, activo, df_data):
                                                    """Calcula la correlación basada en timing de entrada y dirección de operación"""
                                                    df_ea1 = df_data[
                                                        (df_data['EA'] == ea1) & 
                                                        (df_data['Símbolo'].str.lower() == activo)
                                                    ][['Open', 'Tipo']].copy()
                                                    df_ea2 = df_data[
                                                        (df_data['EA'] == ea2) & 
                                                        (df_data['Símbolo'].str.lower() == activo)
                                                    ][['Open', 'Tipo']].copy()
                                                    
                                                    if len(df_ea1) == 0 or len(df_ea2) == 0:
                                                        return np.nan
                                                    
                                                    # Calcular características de entrada para cada estrategia
                                                    # 1. Hora del día (0-23) normalizada a 0-1
                                                    # 2. Día de la semana (0-6) normalizado a 0-1
                                                    # 3. Dirección (1 para Buy, -1 para Sell)
                                                    
                                                    caracteristicas_ea1 = []
                                                    caracteristicas_ea2 = []
                                                    
                                                    # Ventana amplia para encontrar coincidencias
                                                    ventana = pd.Timedelta(days=7)  # Una semana completa
                                                    
                                                    for _, row1 in df_ea1.iterrows():
                                                        fecha1 = row1['Open']
                                                        df_cercanos = df_ea2[
                                                            (df_ea2['Open'] >= fecha1 - ventana) & 
                                                            (df_ea2['Open'] <= fecha1 + ventana)
                                                        ]
                                                        if len(df_cercanos) > 0:
                                                            # Encontrar el más cercano en tiempo
                                                            df_cercanos = df_cercanos.copy()
                                                            df_cercanos['diff'] = abs(df_cercanos['Open'] - fecha1)
                                                            mas_cercano = df_cercanos.nsmallest(1, 'diff').iloc[0]
                                                            
                                                            # Características de la estrategia 1
                                                            hora1 = fecha1.hour / 23.0  # Normalizar a 0-1
                                                            dia_semana1 = fecha1.weekday() / 6.0  # Normalizar a 0-1
                                                            tipo1 = 1.0 if row1['Tipo'] == 'Buy' else -1.0
                                                            
                                                            # Características de la estrategia 2
                                                            fecha2 = mas_cercano['Open']
                                                            hora2 = fecha2.hour / 23.0
                                                            dia_semana2 = fecha2.weekday() / 6.0
                                                            tipo2 = 1.0 if mas_cercano['Tipo'] == 'Buy' else -1.0
                                                            
                                                            # Crear vector de características (combinación ponderada)
                                                            # Peso mayor a la dirección y hora del día
                                                            caracteristica1 = hora1 * 0.4 + dia_semana1 * 0.2 + tipo1 * 0.4
                                                            caracteristica2 = hora2 * 0.4 + dia_semana2 * 0.2 + tipo2 * 0.4
                                                            
                                                            caracteristicas_ea1.append(caracteristica1)
                                                            caracteristicas_ea2.append(caracteristica2)
                                                    
                                                    # Calcular correlación de las características
                                                    if len(caracteristicas_ea1) >= 3:
                                                        try:
                                                            correlacion = np.corrcoef(caracteristicas_ea1, caracteristicas_ea2)[0, 1]
                                                            # Usar valor absoluto para que la correlación esté entre 0 y 1
                                                            # 0 = sin correlación, 1 = correlación perfecta (positiva o negativa)
                                                            if not np.isnan(correlacion):
                                                                return abs(correlacion)
                                                            return np.nan
                                                        except:
                                                            return np.nan
                                                    return np.nan
                                                
                                                for activo, eas in activos_con_multiples.items():
                                                    # Crear matriz de correlación
                                                    n_estrategias = len(eas)
                                                    matriz_correlacion = np.full((n_estrategias, n_estrategias), np.nan)
                                                    
                                                    # Calcular correlación para cada par de estrategias
                                                    for i in range(n_estrategias):
                                                        for j in range(n_estrategias):
                                                            if i == j:
                                                                # Diagonal: correlación consigo mismo es 1.0
                                                                matriz_correlacion[i, j] = 1.0
                                                            elif i < j:
                                                                # Calcular correlación solo una vez para cada par
                                                                corr = calcular_correlacion_estrategias(eas[i], eas[j], activo, df_combinado)
                                                                matriz_correlacion[i, j] = corr
                                                                matriz_correlacion[j, i] = corr  # Matriz simétrica
                                                    
                                                    # Crear DataFrame para la matriz
                                                    df_matriz = pd.DataFrame(
                                                        matriz_correlacion,
                                                        index=eas,
                                                        columns=eas
                                                    )
                                                    
                                                    # Crear heatmap con plotly
                                                    # Preparar texto con mejor contraste
                                                    text_matrix = []
                                                    for row in matriz_correlacion:
                                                        text_row = []
                                                        for val in row:
                                                            if not np.isnan(val):
                                                                text_row.append(f"{val:.3f}")
                                                            else:
                                                                text_row.append("N/A")
                                                        text_matrix.append(text_row)
                                                    
                                                    # Escala de colores: < 0.4 verde, 0.4-0.7 naranja, > 0.7 rojo
                                                    colorscale_custom = [
                                                        [0,    '#46D285'],   # Verde para valores < 0.4
                                                        [0.4,  '#46D285'],   # Verde hasta 0.4
                                                        [0.4,  '#FBBE5B'],   # Naranja desde 0.4
                                                        [0.7,  '#FBBE5B'],   # Naranja hasta 0.7
                                                        [0.7,  '#FA5A5A'],   # Rojo desde 0.7
                                                        [1,    '#FA5A5A']    # Rojo para valores > 0.7
                                                    ]
                                                    
                                                    # Función para obtener el color de borde (todos los bordes del mismo color)
                                                    def obtener_color_borde(valor):
                                                        if pd.isna(valor):
                                                            return None
                                                        return '#d5d5da'  # Todos los bordes del mismo color
                                                    
                                                    # Añadir bordes de colores a cada celda usando shapes
                                                    shapes = []
                                                    n = len(eas)
                                                    for i in range(n):
                                                        for j in range(n):
                                                            valor = matriz_correlacion[i, j]
                                                            if not pd.isna(valor):
                                                                color_borde = obtener_color_borde(valor)
                                                                if color_borde:
                                                                    # Usar las etiquetas de las estrategias como coordenadas
                                                                    x0 = eas[j] if j > 0 else eas[0]
                                                                    x1 = eas[j] if j < n-1 else eas[n-1]
                                                                    y0 = eas[i] if i > 0 else eas[0]
                                                                    y1 = eas[i] if i < n-1 else eas[n-1]
                                                                    
                                                                    # Crear rectángulo para el borde usando coordenadas de categorías
                                                                    shapes.append(dict(
                                                                        type="rect",
                                                                        xref="x",
                                                                        yref="y",
                                                                        x0=j - 0.5,
                                                                        y0=i - 0.5,
                                                                        x1=j + 0.5,
                                                                        y1=i + 0.5,
                                                                        line=dict(color=color_borde, width=2),
                                                                        fillcolor="rgba(0,0,0,0)",  # Transparente, solo borde
                                                                        layer="above"
                                                                    ))
                                                    
                                                    fig_heatmap = go.Figure(data=go.Heatmap(
                                                        z=matriz_correlacion,
                                                        x=eas,
                                                        y=eas,
                                                        colorscale=colorscale_custom,
                                                        zmin=0,
                                                        zmax=1,
                                                        text=text_matrix,
                                                        texttemplate="%{text}",
                                                        textfont={"size": 13},  # Texto gris/negro por defecto (mejor contraste con fondos claros)
                                                        showscale=False,  # Ocultar la leyenda (colorbar)
                                                        xgap=2,  # Espacio horizontal entre celdas
                                                        ygap=2   # Espacio vertical entre celdas
                                                    ))
                                                    
                                                    # Crear título con leyenda de colores
                                                    titulo_con_leyenda = f"Matriz de Correlación - {activo.upper()}                    🟢 < 0.4    🟠 0.4 - 0.7    🔴 > 0.7"
                                                    
                                                    fig_heatmap.update_layout(
                                                        title=titulo_con_leyenda,
                                                        width=600,
                                                        height=600,
                                                        margin=dict(l=80, r=20, t=60, b=60),  # Margen izquierdo más amplio para separar del borde
                                                        plot_bgcolor='#f8f9fa',  # Fondo del área del gráfico
                                                        paper_bgcolor='#f8f9fa',  # Fondo del papel/contenedor
                                                        xaxis=dict(showgrid=False, zeroline=False, showline=False),  # Quitar líneas y bordes del eje X
                                                        yaxis=dict(showgrid=False, zeroline=False, showline=False),  # Quitar líneas y bordes del eje Y
                                                        shapes=shapes
                                                    )
                                                    
                                                    st.plotly_chart(fig_heatmap, use_container_width=True)
                                                    
                                                    # Mostrar también como tabla para mejor legibilidad
                                                    st.markdown(f"**Tabla de Correlación - {activo.upper()}**")
                                                    
                                                    # Formatear valores para la tabla
                                                    df_matriz_display = df_matriz.copy()
                                                    for col in df_matriz_display.columns:
                                                        df_matriz_display[col] = df_matriz_display[col].apply(
                                                            lambda x: f"{x:.3f}" if not np.isnan(x) else "N/A"
                                                        )
                                                    
                                                    # Configurar columnas para la tabla
                                                    column_config_matriz = {}
                                                    for col in df_matriz_display.columns:
                                                        column_config_matriz[col] = st.column_config.TextColumn(
                                                            col,
                                                            help=f"Correlación con {col}"
                                                        )
                                                    
                                                    st.dataframe(
                                                        df_matriz_display,
                                                        use_container_width=True,
                                                        column_config=column_config_matriz,
                                                        hide_index=False
                                                    )
                                        
                                        # Preparar datos de cada estrategia individual
                                        df_combinado['Fecha'] = df_combinado['Close'].dt.date
                                        
                                        fig_comb = go.Figure()
                                        
                                        # Agregar línea para cada estrategia individual (EA + Símbolo)
                                        for ea_nombre, activo in estrategias_filtradas:
                                            df_ea = df_combinado[(df_combinado['EA'] == ea_nombre) & (df_combinado['Símbolo'].str.lower() == activo)].copy()
                                            if len(df_ea) > 0:
                                                df_ea['Fecha'] = df_ea['Close'].dt.date
                                                df_ea = df_ea.sort_values('Fecha')
                                                beneficios_ea = df_ea.groupby('Fecha')['Beneficio'].sum().reset_index()
                                                beneficios_ea = beneficios_ea.sort_values('Fecha')
                                                beneficios_ea['Beneficio_acumulado'] = beneficios_ea['Beneficio'].cumsum()
                                                
                                                nombre_leyenda = f"{ea_nombre} - {activo.upper()}"
                                                fig_comb.add_trace(go.Scatter(
                                                    x=beneficios_ea['Fecha'],
                                                    y=beneficios_ea['Beneficio_acumulado'],
                                                    mode='lines+markers',
                                                    name=nombre_leyenda,
                                                    line=dict(width=2),
                                                    marker=dict(size=4)
                                                ))
                                        
                                        # Calcular y agregar línea del conjunto combinado
                                        beneficios_combinados = df_combinado.groupby('Fecha')['Beneficio'].sum().reset_index()
                                        beneficios_combinados = beneficios_combinados.sort_values('Fecha')
                                        beneficios_combinados['Beneficio_acumulado'] = beneficios_combinados['Beneficio'].cumsum()
                                        
                                        fig_comb.add_trace(go.Scatter(
                                            x=beneficios_combinados['Fecha'],
                                            y=beneficios_combinados['Beneficio_acumulado'],
                                            mode='lines+markers',
                                            name='Combinado',
                                            line=dict(width=3, color='#FF6B6B', dash='dash'),
                                            marker=dict(size=5)
                                        ))
                                        
                                        fig_comb.update_layout(
                                            xaxis_title="Fecha",
                                            yaxis_title="Beneficio Acumulado ($)",
                                            hovermode='x unified',
                                            showlegend=True,
                                            legend=dict(
                                                orientation="h",
                                                yanchor="bottom",
                                                y=1.02,
                                                xanchor="right",
                                                x=1
                                            )
                                        )
                                        
                                        st.plotly_chart(fig_comb, use_container_width=True)
                            except Exception as e:
                                st.error(f"❌ Error al procesar las estadísticas combinadas: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                
                    # Separador visual entre Combinar Estrategias y los tabs
                    st.markdown("""
                    <div style="margin-top: 2rem; margin-bottom: 1rem; border-top: 2px solid #e9ecef; padding-top: 1rem;">
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Resumen mensual por EA - Usar tabs para organizar mejor
                    # Crear tabs para separar Estadísticas por Mes y Trades por Estrategia
                    # Usar nombres únicos para evitar conflicto con tabs principales
                    # Inicializar selección compartida en session_state (antes de los tabs)
                    if 'estrategias_seleccionadas_analisis' not in st.session_state:
                        st.session_state.estrategias_seleccionadas_analisis = []
                    
                    analisis_tab1, analisis_tab2 = st.tabs(["Estadísticas por Mes", "Trades por Estrategia"])
                    
                    # Guardar datos para el resumen mensual con expandables
                    resumen_mensual_data = []
                    
                    # Crear resumen mensual para cada EA
                    for (ea, simbolo), grupo in df.groupby(["EA", "Símbolo"]):
                        ordenado = grupo.sort_values(by='Open')
                        
                        # Detectar múltiples niveles de SL para calcular correctamente
                        perdidas = ordenado[ordenado['Beneficio'] < 0]['Beneficio'].abs()
                        if len(perdidas) > 0:
                            niveles_sl = detectar_niveles_sl(perdidas, min_trades_por_nivel=2, tolerancia_agrupacion=0.15)
                            if len(niveles_sl) == 0:
                                perdida_maxima = perdidas.max()
                                margen_busqueda = perdida_maxima * 0.10
                                sl_detected = perdidas[perdidas >= (perdida_maxima - margen_busqueda)].mode()
                                if len(sl_detected) > 0:
                                    niveles_sl = [sl_detected.iloc[0]]
                                else:
                                    niveles_sl = [perdida_maxima]
                            riesgo_ea = niveles_sl[0] if len(niveles_sl) > 0 else perdidas.max()
                            margen_tolerancia = riesgo_ea * 0.15
                            
                            # Función helper para verificar si una pérdida es un SL directo
                            def es_sl_directo_mes(perdida_abs):
                                """
                                Verifica si una pérdida es un SL directo:
                                - Si es mayor o igual al nivel principal (si solo hay un nivel, con pequeño margen hacia abajo)
                                - Si hay múltiples niveles (alta volatilidad), también verifica si está cerca de alguno
                                """
                                if len(niveles_sl) == 0:
                                    return False
                                
                                # Nivel principal es el primero (el más común y alto)
                                nivel_principal = niveles_sl[0]
                                
                                # Si solo hay un nivel (no hay alta volatilidad), usar umbral con pequeño margen
                                if len(niveles_sl) == 1:
                                    # Aplicar un pequeño margen hacia abajo (2.5% o mínimo $5 menos)
                                    umbral_ajustado = max(nivel_principal * 0.975, nivel_principal - 5)
                                    # Si la pérdida es muy cercana al umbral (dentro de $5), también considerarla SL
                                    if perdida_abs >= umbral_ajustado:
                                        return True
                                    # Si está muy cerca del umbral principal (dentro de $5), también es SL
                                    # Aumentar la tolerancia para capturar pérdidas cercanas al umbral redondeado
                                    if abs(perdida_abs - nivel_principal) <= 5:
                                        return True
                                    # Si la pérdida está cerca del umbral redondeado (dentro de $10), también es SL
                                    # Esto captura casos como $192.99 cuando el umbral es $190
                                    umbral_redondeado = round(nivel_principal / 10) * 10
                                    if abs(perdida_abs - umbral_redondeado) <= 10:
                                        return True
                                    return False
                                
                                # Si hay múltiples niveles (alta volatilidad):
                                # 1. Si es mayor o igual al nivel principal, es SL directo
                                if perdida_abs >= nivel_principal:
                                    return True
                                
                                # 2. Si está cerca de alguno de los niveles detectados (dentro de tolerancia)
                                for nivel in niveles_sl:
                                    tolerancia_nivel = nivel * 0.15
                                    if abs(perdida_abs - nivel) <= tolerancia_nivel:
                                        return True
                                
                                return False
                        else:
                            riesgo_ea = 0
                            margen_tolerancia = 0
                            niveles_sl = []
                            def es_sl_directo_mes(perdida_abs):
                                return False
                        
                        # Agrupar por mes
                        ordenado['Mes'] = ordenado['Open'].dt.to_period('M')
                        
                        resumen_mensual = []
                        for mes, grupo_mes in ordenado.groupby('Mes'):
                            ano = mes.year
                            mes_num = mes.month
                            meses_es = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                                      'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
                            mes_nombre = meses_es[mes_num - 1]
                            
                            # Calcular métricas del mes
                            beneficio_mes = grupo_mes['Beneficio'].sum()
                            total_trades = len(grupo_mes)
                            
                            # Calcular máximo drawdown del mes
                            max_dd_mes = calcular_max_drawdown(grupo_mes['Beneficio'])
                            
                            # Contar SL, TS, TP, TSP, TSL
                            sl_mes = 0
                            ts_mes = 0
                            tp_mes = 0
                            tsp_mes = 0  # Trailing Stop Positivo
                            tsl_mes = 0  # Trailing Stop Loss
                            
                            for _, trade in grupo_mes.iterrows():
                                beneficio = trade['Beneficio']
                                tipo_cierre = trade.get('TipoCierre')
                                
                                # Si el HTML tiene información explícita de [sl] o [tp], usarla PERO validar con el beneficio
                                if tipo_cierre == 'SL':
                                    # [sl] en HTML: verificar si es pérdida o ganancia
                                    if beneficio < 0:
                                        # [sl] con pérdida: verificar si es SL directo o TSL
                                        perdida_abs = abs(beneficio)
                                        if es_sl_directo_mes(perdida_abs):
                                            # SL directo: pérdida que alcanzó alguno de los SL esperados
                                            sl_mes += 1
                                        else:
                                            # TSL: pérdida menor que el SL esperado (trailing stop activó antes)
                                            ts_mes += 1
                                            tsl_mes += 1  # Trailing Stop Loss
                                    elif beneficio > 0:
                                        # [sl] en HTML pero con ganancia: es un Trailing Stop Positivo
                                        ts_mes += 1
                                        tsp_mes += 1  # Trailing Stop Positivo
                                    else:
                                        # Break-even con [sl]: TSL
                                        ts_mes += 1
                                        tsl_mes += 1
                                elif tipo_cierre == 'TP':
                                    # [tp] en HTML: verificar si realmente tocó el TP
                                    if es_tp_real(trade):
                                        tp_mes += 1
                                    else:
                                        # [tp] en HTML pero no tocó exactamente el TP: es un TSP
                                        ts_mes += 1
                                        tsp_mes += 1  # Trailing Stop Positivo
                                else:
                                    # Si no hay información explícita, usar la lógica de cálculo
                                    if beneficio < 0:  # Es pérdida
                                        perdida_abs = abs(beneficio)
                                        if es_sl_directo_mes(perdida_abs):
                                            sl_mes += 1
                                        elif perdida_abs > 0:
                                            ts_mes += 1
                                            tsl_mes += 1  # Trailing Stop Loss
                                        else:
                                            # Beneficio == 0 pero es pérdida (break-even)
                                            ts_mes += 1
                                            tsl_mes += 1
                                    else:  # Es ganancia o break-even positivo
                                        if es_tp_real(trade):
                                            tp_mes += 1
                                        else:
                                            ts_mes += 1
                                            tsp_mes += 1  # Trailing Stop Positivo
                            
                            resumen_mensual.append({
                                "Año - Mes": f"{ano} - {mes_nombre}",
                                "Beneficio": beneficio_mes,  # Mantener numérico para ordenamiento correcto
                                "maxDD": max_dd_mes,  # Mantener numérico para ordenamiento correcto
                                "Trades": total_trades,
                                "TP": tp_mes,
                                "SL": sl_mes,
                                "TSP": tsp_mes,
                                "TSL": tsl_mes
                            })
                        
                        # Guardar datos para mostrar después con expandables
                        resumen_mensual_data.append({
                                'EA': ea,
                            'Simbolo': simbolo,
                            'Data': resumen_mensual
                        })
                    
                    # Mostrar con expandables dentro del analisis_tab1
                    with analisis_tab1:
                        # Crear selector de EA con cálculo de retDD
                        grupos_ordenados_mes = df.groupby(["EA", "Símbolo"]).agg(Beneficio_total=('Beneficio', 'sum')).reset_index()
                        
                        # Calcular retDD para cada grupo
                        ret_dd_list = []
                        for _, row in grupos_ordenados_mes.iterrows():
                            ea = row["EA"]
                            symbol = row["Símbolo"]
                            grupo = df[(df["EA"] == ea) & (df["Símbolo"] == symbol)]
                            ordenado = grupo.sort_values(by='Open')
                            
                            net_profit = ordenado['Beneficio'].sum()
                            max_dd = calcular_max_drawdown(ordenado['Beneficio'])
                            ret_dd = net_profit / abs(max_dd) if max_dd != 0 else 0
                            ret_dd_list.append(ret_dd)
                        
                        grupos_ordenados_mes['retDD'] = ret_dd_list
                        grupos_ordenados_mes = grupos_ordenados_mes.sort_values(by="retDD", ascending=False)
                        
                        # Crear lista de opciones para el selector múltiple
                        opciones_ea_mes = [f"{row['EA']} - {row['Símbolo'].upper()}" for _, row in grupos_ordenados_mes.iterrows()]
                        
                        # Inicializar selección compartida en session_state
                        if 'estrategias_seleccionadas_analisis' not in st.session_state:
                            st.session_state.estrategias_seleccionadas_analisis = []
                        
                        # Inicializar la key del multiselect solo si no existe
                        if 'multiselect_ea_mes' not in st.session_state:
                            # Filtrar las opciones seleccionadas que existen en este tab
                            seleccion_validas_mes = [s for s in st.session_state.estrategias_seleccionadas_analisis if s in opciones_ea_mes]
                            st.session_state.multiselect_ea_mes = seleccion_validas_mes
                        
                        # Filtrar solo las selecciones válidas del multiselect (respeta eliminaciones del usuario)
                        seleccion_actual_mes = [s for s in st.session_state.multiselect_ea_mes if s in opciones_ea_mes]
                        
                        # Selector múltiple de EAs (compartido entre ambos tabs)
                        # No usar default, dejar que Streamlit use el valor de session_state[key]
                        estrategias_seleccionadas_mes = st.multiselect(
                            "Selecciona una o más estrategias para ver sus estadísticas mensuales:",
                            options=opciones_ea_mes,
                            key="multiselect_ea_mes",
                            help="Selecciona una o más estrategias para ver sus estadísticas mensuales."
                        )
                        
                        # Actualizar estado compartido: combinar con las del otro tab
                        if 'multiselect_ea_trades' in st.session_state:
                            # Usar las selecciones del otro tab directamente (ya están filtradas por sus opciones)
                            otras_validas = st.session_state.multiselect_ea_trades
                            todas_selecciones = list(set(estrategias_seleccionadas_mes + otras_validas))
                            st.session_state.estrategias_seleccionadas_analisis = todas_selecciones
                        else:
                            st.session_state.estrategias_seleccionadas_analisis = estrategias_seleccionadas_mes
                        
                        # Función para mostrar tabla de una EA
                        def mostrar_tabla_mensual(ea, symbol, resumen_ea):
                            if resumen_ea and resumen_ea['Data']:
                                df_resumen_mes = pd.DataFrame(resumen_ea['Data'])
                                
                                # Aplicar colores a la columna Beneficio (ahora recibe valores numéricos)
                                def estilizar_beneficio(valor):
                                    if isinstance(valor, (int, float)):
                                        if valor >= 0:
                                            return 'background-color: #90EE90'  # Verde claro
                                        else:
                                            return 'background-color: #FFB6C1'  # Rojo claro
                                    return ''
                                
                                # Aplicar estilo solo a la columna Beneficio
                                styled_df = df_resumen_mes.style.applymap(estilizar_beneficio, subset=['Beneficio'])
                                
                                column_config_mes = {
                                    "Año - Mes": st.column_config.TextColumn("Año - Mes", help="Año y mes"),
                                    "Beneficio": st.column_config.NumberColumn("Beneficio", help="Beneficio del mes", format="$%.2f"),
                                    "maxDD": st.column_config.NumberColumn("maxDD", help="Máximo drawdown del mes", format="$%.2f"),
                                    "Trades": st.column_config.NumberColumn("Trades", help="Total de operaciones", format="%d"),
                                    "TP": st.column_config.NumberColumn("TP", help="Trades con TP", format="%d"),
                                    "SL": st.column_config.NumberColumn("SL", help="Trades con SL directo", format="%d"),
                                    "TSP": st.column_config.NumberColumn("TSP", help="Trailing Stop Positivo (trades con ganancia cerrados antes del TP)", format="%d"),
                                    "TSL": st.column_config.NumberColumn("TSL", help="Trailing Stop Loss (trades con pérdida menor al SL directo)", format="%d")
                                }
                                
                                st.dataframe(styled_df, use_container_width=True, column_config=column_config_mes, hide_index=True)
                        
                        # Mostrar las estrategias seleccionadas
                        for estrategia_sel in estrategias_seleccionadas_mes:
                            # Formato: "EA - SÍMBOLO"
                            partes = estrategia_sel.split(" - ")
                            if len(partes) == 2:
                                ea_selected = partes[0]
                                symbol_selected = partes[1].upper()
                                
                                # Buscar los datos del resumen mensual para esta EA
                                resumen_ea = next((item for item in resumen_mensual_data if item['EA'] == ea_selected and item['Simbolo'].upper() == symbol_selected), None)
                                
                                if resumen_ea and resumen_ea['Data']:
                                    with st.expander(f"📌 {ea_selected} - {symbol_selected}", expanded=False):
                                        mostrar_tabla_mensual(ea_selected, symbol_selected, resumen_ea)
                                else:
                                    st.info(f"No hay datos disponibles para {estrategia_sel}")
                    
                    # Crear gráfico de beneficio acumulado
                    st.markdown("""
                    <div style="margin-top: 2rem;">
                        <h3 style="margin-bottom: 1rem;">Evolución de Beneficios Acumulados</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    df_grafico = df.copy()
                    df_grafico['Fecha'] = df_grafico['Close'].dt.date
                    # Crear columna combinada EA + Símbolo para distinguir estrategias con mismo nombre
                    df_grafico['EA_Completa'] = df_grafico['EA'] + ' - ' + df_grafico['Símbolo'].str.upper()
                    beneficios_diarios = df_grafico.groupby(['EA_Completa', 'Fecha'])['Beneficio'].sum().reset_index()
                    beneficios_diarios['Beneficio_acumulado'] = beneficios_diarios.groupby('EA_Completa')['Beneficio'].cumsum()

                    if len(beneficios_diarios) > 0:
                        fig = px.line(
                            beneficios_diarios,
                            x="Fecha",
                            y="Beneficio_acumulado",
                            color="EA_Completa",
                            markers=True,
                            labels={"Beneficio_acumulado": "Beneficio acumulado", "Fecha": "Fecha", "EA_Completa": "Estrategia"}
                        )

                        fig.update_traces(
                            hovertemplate=
                            "<b>Estrategia:</b> %{fullData.name}<br>" +
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
                    
                    # Mostrar operaciones individuales por EA y símbolo dentro del analisis_tab2
                    with analisis_tab2:
                        # Crear selector de EA para trades con cálculo de retDD
                        grupos_ordenados = df.groupby(["EA", "Símbolo"]).agg(Beneficio_total=('Beneficio', 'sum')).reset_index()

                        # Calcular retDD para cada grupo
                        ret_dd_list = []
                        for _, row in grupos_ordenados.iterrows():
                            ea = row["EA"]
                            symbol = row["Símbolo"]
                            grupo = df[(df["EA"] == ea) & (df["Símbolo"] == symbol)]
                            ordenado = grupo.sort_values(by='Open')
                            
                            net_profit = ordenado['Beneficio'].sum()
                            max_dd = calcular_max_drawdown(ordenado['Beneficio'])
                            ret_dd = net_profit / abs(max_dd) if max_dd != 0 else 0
                            ret_dd_list.append(ret_dd)
                        
                        grupos_ordenados['retDD'] = ret_dd_list
                        grupos_ordenados = grupos_ordenados.sort_values(by="retDD", ascending=False)
                        
                        # Crear lista de opciones para el selector múltiple
                        opciones_ea_trades = [f"{row['EA']} - {row['Símbolo'].upper()}" for _, row in grupos_ordenados.iterrows()]
                        
                        # Inicializar la key del multiselect solo si no existe
                        if 'multiselect_ea_trades' not in st.session_state:
                            # Filtrar las opciones seleccionadas que existen en este tab
                            seleccion_validas_trades = [s for s in st.session_state.estrategias_seleccionadas_analisis if s in opciones_ea_trades]
                            st.session_state.multiselect_ea_trades = seleccion_validas_trades
                        
                        # Filtrar solo las selecciones válidas del multiselect (respeta eliminaciones del usuario)
                        seleccion_actual_trades = [s for s in st.session_state.multiselect_ea_trades if s in opciones_ea_trades]
                        
                        # Selector múltiple de EAs (compartido entre ambos tabs)
                        # No usar default, dejar que Streamlit use el valor de session_state[key]
                        estrategias_seleccionadas_trades = st.multiselect(
                            "Selecciona una o más estrategias para ver sus trades individuales:",
                            options=opciones_ea_trades,
                            key="multiselect_ea_trades",
                            help="Selecciona una o más estrategias para ver sus trades individuales."
                        )
                        
                        # Actualizar estado compartido: combinar con las del otro tab
                        if 'multiselect_ea_mes' in st.session_state:
                            # Usar las selecciones del otro tab directamente (ya están filtradas por sus opciones)
                            otras_validas = st.session_state.multiselect_ea_mes
                            todas_selecciones = list(set(estrategias_seleccionadas_trades + otras_validas))
                            st.session_state.estrategias_seleccionadas_analisis = todas_selecciones
                        else:
                            st.session_state.estrategias_seleccionadas_analisis = estrategias_seleccionadas_trades
                        
                        # Función para mostrar trades de una EA
                        def mostrar_trades_ea(ea, symbol, grupo):
                            grupo_display = grupo.sort_values(by="Open").copy()
                            
                            def formatear_duracion(minutos):
                                if pd.isna(minutos):
                                    return "-"
                                horas = int(minutos) // 60
                                mins = int(minutos) % 60
                                if horas > 0:
                                    if mins > 0:
                                        return f"{horas}h {mins}m"
                                    else:
                                        return f"{horas}h"
                                else:
                                    return f"{mins}m"
                            
                            grupo_display['Duración'] = grupo_display['Duración'].apply(formatear_duracion)
                            beneficio_original = grupo_display['Beneficio'].copy()
                            
                            # Calcular múltiples niveles de SL para clasificar correctamente los trades
                            perdidas = grupo_display[grupo_display['Beneficio'] < 0]['Beneficio'].abs()
                            if len(perdidas) > 0:
                                niveles_sl = detectar_niveles_sl(perdidas, min_trades_por_nivel=2, tolerancia_agrupacion=0.15)
                                if len(niveles_sl) == 0:
                                    perdida_maxima = perdidas.max()
                                    margen_busqueda = perdida_maxima * 0.10
                                    sl_detected = perdidas[perdidas >= (perdida_maxima - margen_busqueda)].mode()
                                    if len(sl_detected) > 0:
                                        niveles_sl = [sl_detected.iloc[0]]
                                    else:
                                        niveles_sl = [perdida_maxima]
                                riesgo_ea = niveles_sl[0] if len(niveles_sl) > 0 else perdidas.max()
                                margen_tolerancia = riesgo_ea * 0.15
                                
                                def es_sl_directo_display(perdida_abs):
                                    if len(niveles_sl) == 0:
                                        return False
                                    nivel_principal = niveles_sl[0]
                                    if len(niveles_sl) == 1:
                                        umbral_ajustado = max(nivel_principal * 0.975, nivel_principal - 5)
                                        if perdida_abs >= umbral_ajustado:
                                            return True
                                        if abs(perdida_abs - nivel_principal) <= 5:
                                            return True
                                        umbral_redondeado = round(nivel_principal / 10) * 10
                                        if abs(perdida_abs - umbral_redondeado) <= 10:
                                            return True
                                        return False
                                    if perdida_abs >= nivel_principal:
                                        return True
                                    for nivel in niveles_sl:
                                        tolerancia_nivel = nivel * 0.15
                                        if abs(perdida_abs - nivel) <= tolerancia_nivel:
                                            return True
                                    return False
                            else:
                                riesgo_ea = 0
                                margen_tolerancia = 0
                                niveles_sl = []
                                def es_sl_directo_display(perdida_abs):
                                    return False
                            
                            def clasificar_trade(row):
                                idx = row.name
                                beneficio = beneficio_original.loc[idx]
                                tipo_cierre_html = row['TipoCierre'] if 'TipoCierre' in row.index and pd.notna(row['TipoCierre']) else None
                                
                                if tipo_cierre_html == 'SL':
                                    if beneficio < 0:
                                        perdida_abs = abs(beneficio)
                                        if es_sl_directo_display(perdida_abs):
                                            return 'SL'
                                        else:
                                            return 'TSL'
                                    elif beneficio > 0:
                                        return 'TSP'
                                    else:
                                        return 'TSL'
                                elif tipo_cierre_html == 'TP':
                                    return 'TP'
                                else:
                                    if beneficio < 0:
                                        perdida_abs = abs(beneficio)
                                        if es_sl_directo_display(perdida_abs):
                                            return 'SL'
                                        else:
                                            return 'TSL'
                                    elif beneficio > 0:
                                        try:
                                            tp_val = row['TP_val'] if 'TP_val' in row.index and pd.notna(row['TP_val']) else None
                                            close_price = row['PrecioClose'] if 'PrecioClose' in row.index and pd.notna(row['PrecioClose']) else None
                                            if tp_val is not None and close_price is not None:
                                                decs = row['TP_decimales'] if 'TP_decimales' in row.index and pd.notna(row['TP_decimales']) else 0
                                                tol = max(10 ** (-(decs)), 1e-6)
                                                if abs(close_price - tp_val) <= 2 * tol:
                                                    return 'TP'
                                        except:
                                            pass
                                        return 'TSP'
                                    else:
                                        return 'TSL'
                            
                            if 'TipoCierre' in grupo_display.columns:
                                grupo_display['TipoCierre'] = grupo_display.apply(clasificar_trade, axis=1)
                            else:
                                grupo_display['TipoCierre'] = grupo_display.apply(clasificar_trade, axis=1)
                            
                            columnas_a_eliminar = ['EA', 'TP_decimales', 'Símbolo']
                            columnas_disponibles = [col for col in grupo_display.columns if col not in columnas_a_eliminar]
                            
                            columnas_ordenadas = []
                            if 'Tipo' in columnas_disponibles:
                                columnas_ordenadas.append('Tipo')
                            if 'TipoCierre' in columnas_disponibles:
                                columnas_ordenadas.append('TipoCierre')
                            if 'Beneficio' in columnas_disponibles:
                                columnas_ordenadas.append('Beneficio')
                            if 'Duración' in columnas_disponibles:
                                columnas_ordenadas.append('Duración')
                            for col in columnas_disponibles:
                                if col not in columnas_ordenadas:
                                    columnas_ordenadas.append(col)
                            
                            grupo_display = grupo_display[columnas_ordenadas].copy()
                            
                            def estilo_fila(row):
                                beneficio = beneficio_original[row.name]
                                if beneficio > 0:
                                    return ['background-color: #90EE90'] * len(row)
                                elif beneficio < 0:
                                    return ['background-color: #FFB6C1'] * len(row)
                                else:
                                    return ['background-color: white'] * len(row)
                            
                            styled_grupo = grupo_display.style.apply(estilo_fila, axis=1)
                            
                            column_config_display = {}
                            if 'Beneficio' in grupo_display.columns:
                                column_config_display['Beneficio'] = st.column_config.NumberColumn(
                                    "Beneficio",
                                    help="Beneficio de la operación",
                                    format="$%.2f"
                                )
                            if 'Volumen' in grupo_display.columns:
                                volumen_original = grupo_display['Volumen'].copy()
                                grupo_display['Volumen'] = grupo_display['Volumen'].apply(
                                    lambda x: f"{x:.2f}".rstrip('0').rstrip('.') if pd.notna(x) else ''
                                )
                                column_config_display['Volumen'] = st.column_config.TextColumn(
                                    "Volumen",
                                    help="Volumen de la operación"
                                )
                            
                            st.dataframe(styled_grupo, use_container_width=True, column_config=column_config_display)
                        
                        # Mostrar las estrategias seleccionadas
                        for estrategia_sel in estrategias_seleccionadas_trades:
                            # Formato: "EA - SÍMBOLO"
                            partes = estrategia_sel.split(" - ")
                            if len(partes) == 2:
                                ea_selected = partes[0]
                                symbol_selected = partes[1].upper()
                                # Comparar símbolo de forma case-insensitive
                                grupo = df[(df["EA"] == ea_selected) & (df["Símbolo"].str.upper() == symbol_selected)]
                                
                                if len(grupo) > 0:
                                    with st.expander(f"📌 {ea_selected} ({len(grupo)} operaciones) - {symbol_selected}", expanded=False):
                                        mostrar_trades_ea(ea_selected, symbol_selected, grupo)
                                else:
                                    st.info("No hay trades disponibles para esta estrategia")
                
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
        # Tarjeta de análisis de EA
        st.markdown("""
        <div class="card">
            <div class="card-title">Análisis de EA</div>
        """, unsafe_allow_html=True)
        
        # Contenedor para subir archivos CSV
        archivos_csv = st.file_uploader(
            "Elegir archivo(s) CSV", 
            type=["csv"], 
            label_visibility="collapsed", 
            accept_multiple_files=True,
            key="analisis_ea_csv"
        )
        
        if archivos_csv:
            try:
                resultados = []
                datos_estrategias = []  # Guardar datos individuales de cada estrategia
                
                for archivo_csv in archivos_csv:
                    # Leer el CSV - intentar diferentes separadores (priorizar ; para CSVs europeos)
                    df = None
                    for sep in [';', ',', '\t']:
                        try:
                            df = pd.read_csv(archivo_csv, sep=sep, quotechar='"', encoding='utf-8')
                            # Verificar que se leyó correctamente (debe tener más de 0 columnas)
                            if len(df.columns) > 0:
                                break
                        except:
                            continue
                    
                    if df is None or len(df.columns) == 0:
                        st.error(f"❌ No se pudo leer el archivo {archivo_csv.name}. Verifica el formato.")
                        continue
                    
                    # Obtener nombre de la estrategia (nombre del archivo sin extensión)
                    nombre_estrategia = archivo_csv.name.replace('.csv', '').replace('.CSV', '')
                    
                    # Normalizar nombres de columnas (pueden variar, quitar comillas y espacios)
                    df.columns = df.columns.str.strip().str.strip('"').str.strip("'")
                    
                    # Identificar columnas relevantes (buscar variaciones comunes)
                    profit_col = None
                    symbol_col = None
                    balance_col = None
                    close_type_col = None
                    open_time_col = None
                    close_time_col = None
                    
                    for col in df.columns:
                        col_lower = col.lower()
                        col_normalized = col_lower.replace('/', ' ').replace('_', ' ').replace('-', ' ')
                        
                        # Buscar columna de Profit/Loss (más flexible)
                        # Primero buscar coincidencias exactas o muy cercanas
                        if ('profit/loss' in col_lower or 'profit-loss' in col_lower or
                            col_lower == 'profit/loss' or col_lower == 'profit-loss'):
                            profit_col = col
                        # Luego buscar variaciones
                        elif (('profit' in col_normalized and 'loss' in col_normalized) or 
                              'pnl' in col_normalized or 'beneficio' in col_normalized or
                              ('profit' in col_normalized and 'factor' not in col_normalized and 'ratio' not in col_normalized) or
                              ('loss' in col_normalized and 'profit' not in col_normalized and 'drawdown' not in col_normalized)):
                            if profit_col is None:  # Solo asignar si no se encontró una mejor coincidencia
                                profit_col = col
                        elif 'symbol' in col_normalized or 'símbolo' in col_normalized:
                            symbol_col = col
                        elif 'balance' in col_normalized:
                            balance_col = col
                        elif 'close type' in col_normalized or 'tipo cierre' in col_normalized:
                            close_type_col = col
                        elif 'open time' in col_normalized or 'open_time' in col_normalized:
                            open_time_col = col
                        elif 'close time' in col_normalized or 'close_time' in col_normalized:
                            close_time_col = col
                    
                    if profit_col is None:
                        # Mostrar información de debug
                        st.warning(f"⚠️ No se encontró columna de Profit/Loss en {archivo_csv.name}")
                        columnas_lista = df.columns.tolist()
                        if len(columnas_lista) > 20:
                            st.info(f"Columnas encontradas (primeras 20): {', '.join(columnas_lista[:20])}...")
                            st.info(f"Total de columnas: {len(columnas_lista)}")
                        else:
                            st.info(f"Columnas encontradas: {', '.join(columnas_lista)}")
                        # Intentar buscar manualmente columnas que contengan "profit" o "loss"
                        posibles_columnas = [col for col in df.columns if 'profit' in col.lower() or 'loss' in col.lower() or 'pnl' in col.lower()]
                        if posibles_columnas:
                            st.info(f"Columnas que contienen 'profit', 'loss' o 'pnl': {', '.join(posibles_columnas)}")
                        continue
                    
                    # Convertir Profit/Loss a numérico
                    df[profit_col] = pd.to_numeric(df[profit_col], errors='coerce')
                    df = df.dropna(subset=[profit_col])
                    
                    if len(df) == 0:
                        st.warning(f"⚠️ No hay datos válidos en {archivo_csv.name}")
                        continue
                    
                    # Obtener activo (Symbol)
                    activo = df[symbol_col].iloc[0] if symbol_col and symbol_col in df.columns else "N/A"
                    
                    # Calcular métricas básicas
                    net_profit = df[profit_col].sum()
                    
                    # Profit Factor
                    ganancias = df[df[profit_col] > 0][profit_col].sum()
                    perdidas = abs(df[df[profit_col] < 0][profit_col].sum())
                    profit_factor = ganancias / perdidas if perdidas > 0 else float('inf') if ganancias > 0 else 0
                    
                    # Winrate
                    trades_ganadores = len(df[df[profit_col] > 0])
                    trades_perdedores = len(df[df[profit_col] < 0])
                    total_trades = len(df)
                    winrate = (trades_ganadores / total_trades * 100) if total_trades > 0 else 0
                    
                    # AvgWin / AvgLoss
                    avg_win = df[df[profit_col] > 0][profit_col].mean() if trades_ganadores > 0 else 0
                    avg_loss = abs(df[df[profit_col] < 0][profit_col].mean()) if trades_perdedores > 0 else 0
                    
                    # Max Consecutive Loss
                    max_consec_loss = 0
                    current_consec_loss = 0
                    for profit in df[profit_col]:
                        if profit < 0:
                            current_consec_loss += 1
                            max_consec_loss = max(max_consec_loss, current_consec_loss)
                        else:
                            current_consec_loss = 0
                    
                    # Calcular Drawdown
                    # Primero ordenar por tiempo si está disponible
                    if open_time_col and open_time_col in df.columns:
                        try:
                            df[open_time_col] = pd.to_datetime(df[open_time_col], errors='coerce')
                            df = df.sort_values(by=open_time_col).dropna(subset=[open_time_col])
                        except:
                            df = df.sort_index()
                    else:
                        df = df.sort_index()
                    
                    if balance_col and balance_col in df.columns:
                        # Calcular drawdown basado en Balance
                        df[balance_col] = pd.to_numeric(df[balance_col], errors='coerce')
                        df = df.dropna(subset=[balance_col])
                        if len(df) > 0:
                            df['cummax'] = df[balance_col].cummax()
                            # Drawdown en valor absoluto (dólares)
                            df['drawdown'] = df['cummax'] - df[balance_col]
                            max_dd = df['drawdown'].max() if len(df) > 0 else 0
                        else:
                            max_dd = 0
                    else:
                        # Calcular drawdown basado en equity curve de profit
                        df['equity'] = df[profit_col].cumsum()
                        df['cummax'] = df['equity'].cummax()
                        # Drawdown en valor absoluto (dólares)
                        df['drawdown'] = df['cummax'] - df['equity']
                        max_dd = df['drawdown'].max() if len(df) > 0 else 0
                    
                    # retdd (Return/Drawdown ratio)
                    retdd = (net_profit / max_dd) if max_dd > 0 else 0
                    
                    # Ulcer Index
                    # Calcular drawdown porcentual para Ulcer Index usando la equity curve
                    if balance_col and balance_col in df.columns:
                        # Si tenemos balance, usamos balance como equity
                        equity_for_ui = df[balance_col]
                    else:
                        # Si no, usamos la equity curve calculada
                        equity_for_ui = df['equity'] if 'equity' in df.columns else df[profit_col].cumsum()
                    
                    # Calcular máximo acumulado de la equity
                    cummax_equity = equity_for_ui.cummax()
                    # Drawdown porcentual: (cummax - equity) / cummax * 100 (siempre positivo)
                    # Evitar división por cero
                    cummax_safe = cummax_equity.replace(0, 1)
                    dd_pct = ((cummax_equity - equity_for_ui) / cummax_safe * 100).fillna(0)
                    # Solo considerar drawdowns (valores positivos)
                    dd_pct_positive = dd_pct.clip(lower=0)
                    # Ulcer Index = sqrt(sum(dd_pct^2) / n)
                    if len(df) > 0:
                        ulcer_index = np.sqrt((dd_pct_positive ** 2).sum() / len(df))
                    else:
                        ulcer_index = 0
                    
                    # Z-score (fórmula StrategyQuant basada en runs)
                    # Mide la aleatoriedad de las secuencias de ganancias y pérdidas
                    if total_trades > 0:
                        # N = número total de operaciones
                        N = total_trades
                        # W = número de operaciones ganadoras
                        W = trades_ganadores
                        # L = número de operaciones perdedoras
                        L = trades_perdedores
                        # P = probabilidad de ganar
                        P = W / N if N > 0 else 0
                        
                        # Calcular R = número total de runs (rachas)
                        # Un run es una secuencia consecutiva de ganancias o pérdidas
                        if len(df) > 0:
                            # Ordenar por tiempo si está disponible para asegurar orden correcto
                            df_zscore = df.copy()
                            if open_time_col and open_time_col in df_zscore.columns:
                                try:
                                    df_zscore[open_time_col] = pd.to_datetime(df_zscore[open_time_col], errors='coerce')
                                    df_zscore = df_zscore.sort_values(open_time_col).dropna(subset=[open_time_col])
                                except:
                                    pass
                            
                            # Crear secuencia de ganancias (1) y pérdidas (0)
                            profits = df_zscore[profit_col].values
                            secuencia = [1 if p > 0 else 0 for p in profits]
                            
                            # Contar runs: cada cambio en la secuencia es un nuevo run
                            R = 1  # Siempre hay al menos 1 run
                            for i in range(1, len(secuencia)):
                                if secuencia[i] != secuencia[i-1]:
                                    R += 1
                        else:
                            R = 0
                        
                        # Calcular Z-score solo si hay suficientes datos
                        if N > 1 and P > 0 and P < 1:
                            # E(R) = 1 + 2 * (N-1) * P * (1-P)
                            E_R = 1 + 2 * (N - 1) * P * (1 - P)
                            
                            # Var(R) = 2 * (N-1) * P * (1-P) * (2*P*(1-P))
                            Var_R = 2 * (N - 1) * P * (1 - P) * (2 * P * (1 - P))
                            
                            # Z = (R - E(R)) / sqrt(Var(R))
                            if Var_R > 0:
                                z_score = (R - E_R) / np.sqrt(Var_R)
                            else:
                                z_score = 0
                        else:
                            z_score = 0
                    else:
                        z_score = 0
                    
                    # Sharpe Ratio (fórmula StrategyQuant)
                    # Sharpe = (R_p - R_f) / σ_p
                    # Donde R_p = rentabilidad media, R_f = 0 (tasa libre de riesgo), σ_p = desviación estándar
                    if len(df) > 1:
                        # Usar retornos por operación (profit values)
                        returns = df[profit_col].values
                        mean_return = returns.mean()  # R_p
                        std_return = returns.std()    # σ_p
                        
                        # R_f = 0 (tasa libre de riesgo, asumida en 0 para trading algorítmico)
                        R_f = 0
                        
                        if std_return > 0:
                            # Sharpe Ratio = (R_p - R_f) / σ_p
                            sharpe_ratio = (mean_return - R_f) / std_return
                        else:
                            sharpe_ratio = 0
                    else:
                        sharpe_ratio = 0
                    
                    # Avg Trade Mensual (número promedio de trades por mes)
                    total_trades = len(df)
                    if open_time_col and open_time_col in df.columns:
                        try:
                            df[open_time_col] = pd.to_datetime(df[open_time_col], errors='coerce')
                            df = df.dropna(subset=[open_time_col])
                            if len(df) > 0:
                                fecha_inicio = df[open_time_col].min()
                                fecha_fin = df[open_time_col].max()
                                meses = (fecha_fin - fecha_inicio).days / 30.44  # Promedio de días por mes
                                avg_trade_mensual = total_trades / meses if meses > 0 else total_trades
                            else:
                                avg_trade_mensual = total_trades
                        except:
                            avg_trade_mensual = total_trades
                    else:
                        avg_trade_mensual = total_trades
                    
                    # Calmar Ratio (CAGR / Max DD %)
                    # MetaTrader calcula usando CAGR anualizado dividido por max drawdown porcentual
                    if max_dd > 0:
                        # Obtener balance inicial y final si está disponible
                        balance_inicial_calmar = None
                        balance_final_calmar = None
                        if balance_col and balance_col in df.columns:
                            df_balance_calmar = df.copy()
                            df_balance_calmar[balance_col] = pd.to_numeric(df_balance_calmar[balance_col], errors='coerce')
                            df_balance_calmar = df_balance_calmar.dropna(subset=[balance_col])
                            if len(df_balance_calmar) > 0:
                                # Ordenar por tiempo si está disponible
                                if open_time_col and open_time_col in df_balance_calmar.columns:
                                    df_balance_calmar[open_time_col] = pd.to_datetime(df_balance_calmar[open_time_col], errors='coerce')
                                    df_balance_calmar = df_balance_calmar.sort_values(open_time_col).dropna(subset=[open_time_col])
                                
                                balance_inicial_calmar = df_balance_calmar[balance_col].iloc[0] - df_balance_calmar[profit_col].iloc[0]
                                balance_final_calmar = df_balance_calmar[balance_col].iloc[-1]
                                
                                # Calcular período en años
                                if open_time_col and open_time_col in df_balance_calmar.columns:
                                    fecha_inicio_calmar = df_balance_calmar[open_time_col].iloc[0]
                                    fecha_fin_calmar = df_balance_calmar[open_time_col].iloc[-1]
                                    dias_totales_calmar = (fecha_fin_calmar - fecha_inicio_calmar).days
                                    años_calmar = dias_totales_calmar / 365.25
                                else:
                                    años_calmar = 1  # Default a 1 año si no hay fechas
                                
                                if balance_inicial_calmar > 0 and balance_final_calmar > 0 and años_calmar > 0:
                                    # Calcular CAGR (Compound Annual Growth Rate)
                                    # CAGR = ((Valor Final / Valor Inicial) ^ (1 / años)) - 1
                                    ratio_valor = balance_final_calmar / balance_inicial_calmar
                                    if ratio_valor > 0:
                                        cagr = (ratio_valor ** (1 / años_calmar) - 1) * 100  # En porcentaje
                                    else:
                                        cagr = 0
                                    
                                    # Calcular max drawdown porcentual
                                    df_calmar = df_balance_calmar.copy()
                                    df_calmar['cummax'] = df_calmar[balance_col].cummax()
                                    # Drawdown porcentual máximo
                                    df_calmar['dd_pct'] = ((df_calmar['cummax'] - df_calmar[balance_col]) / df_calmar['cummax'] * 100).fillna(0)
                                    max_dd_pct = df_calmar['dd_pct'].max()
                                    
                                    if max_dd_pct > 0:
                                        calmar = cagr / max_dd_pct
                                    else:
                                        calmar = 0
                                else:
                                    # Fallback: usar valores absolutos
                                    calmar = (net_profit / max_dd) if max_dd > 0 else 0
                            else:
                                calmar = 0
                        else:
                            # Si no hay balance, usar cálculo simple
                            calmar = (net_profit / max_dd) if max_dd > 0 else 0
                    else:
                        calmar = 0
                    
                    # R Expectancy
                    # Calcular valor R (usando avg_loss como R) y valor en dólares
                    if total_trades > 0 and avg_loss > 0:
                        # Valor R: (winrate/100 * avg_win/avg_loss) - ((100-winrate)/100 * 1)
                        r_expectancy_r = (winrate / 100 * (avg_win / avg_loss)) - ((100 - winrate) / 100 * 1)
                        # Valor en dólares: (winrate/100 * avg_win) - ((100-winrate)/100 * avg_loss)
                        r_expectancy_dollar = (winrate / 100 * avg_win) - ((100 - winrate) / 100 * avg_loss)
                    else:
                        r_expectancy_r = 0
                        r_expectancy_dollar = 0
                    
                    # R-squared (Smoothness de la curva de capital)
                    # Mide qué tan suave es la curva de equity (qué tan bien se ajusta a una línea recta)
                    if len(df) > 1:
                        # Obtener la equity curve
                        if balance_col and balance_col in df.columns:
                            equity_curve = df[balance_col].values
                        else:
                            equity_curve = df['equity'].values if 'equity' in df.columns else df[profit_col].cumsum().values
                        
                        # Crear variable X (número de trade: 0, 1, 2, ..., n-1)
                        x = np.arange(len(equity_curve))
                        
                        # Calcular regresión lineal: equity = a + b*x
                        # R² = 1 - (SS_res / SS_tot)
                        # Donde SS_res = sum((y - y_pred)^2) y SS_tot = sum((y - y_mean)^2)
                        if len(equity_curve) > 1:
                            # Calcular coeficientes de regresión lineal
                            x_mean = x.mean()
                            y_mean = equity_curve.mean()
                            
                            # Calcular pendiente y ordenada al origen
                            numerator = ((x - x_mean) * (equity_curve - y_mean)).sum()
                            denominator = ((x - x_mean) ** 2).sum()
                            
                            if denominator > 0:
                                slope = numerator / denominator
                                intercept = y_mean - slope * x_mean
                                
                                # Calcular valores predichos
                                y_pred = intercept + slope * x
                                
                                # Calcular R²
                                ss_res = ((equity_curve - y_pred) ** 2).sum()
                                ss_tot = ((equity_curve - y_mean) ** 2).sum()
                                
                                if ss_tot > 0:
                                    r_squared = 1 - (ss_res / ss_tot)
                                else:
                                    r_squared = 0
                            else:
                                r_squared = 0
                        else:
                            r_squared = 0
                    else:
                        r_squared = 0
                    
                    # Holding Time (Media de duración de las operaciones)
                    holding_time_hours = 0
                    if open_time_col and close_time_col and open_time_col in df.columns and close_time_col in df.columns:
                        try:
                            # Intentar parsear las fechas
                            df_time = df.copy()
                            df_time[open_time_col] = pd.to_datetime(df_time[open_time_col], errors='coerce')
                            df_time[close_time_col] = pd.to_datetime(df_time[close_time_col], errors='coerce')
                            df_time = df_time.dropna(subset=[open_time_col, close_time_col])
                            
                            if len(df_time) > 0:
                                # Calcular diferencia de tiempo
                                time_diff = df_time[close_time_col] - df_time[open_time_col]
                                # Convertir a horas
                                holding_time_hours = time_diff.dt.total_seconds().sum() / 3600.0 / len(df_time)
                        except:
                            # Si falla, intentar con columna "Time in trade" si existe
                            time_in_trade_col = None
                            for col in df.columns:
                                if 'time' in col.lower() and 'trade' in col.lower():
                                    time_in_trade_col = col
                                    break
                            
                            if time_in_trade_col and time_in_trade_col in df.columns:
                                try:
                                    # Parsear formato como "1h 30m", "32m 0s", etc.
                                    def parse_time_in_trade(time_str):
                                        if pd.isna(time_str) or time_str == '':
                                            return 0
                                        time_str = str(time_str).strip()
                                        hours = 0
                                        minutes = 0
                                        seconds = 0
                                        
                                        # Buscar horas
                                        if 'h' in time_str:
                                            h_part = time_str.split('h')[0]
                                            hours = float(h_part.strip())
                                            time_str = time_str.split('h', 1)[1]
                                        
                                        # Buscar minutos
                                        if 'm' in time_str:
                                            m_part = time_str.split('m')[0]
                                            minutes = float(m_part.strip())
                                            time_str = time_str.split('m', 1)[1]
                                        
                                        # Buscar segundos
                                        if 's' in time_str:
                                            s_part = time_str.split('s')[0]
                                            seconds = float(s_part.strip())
                                        
                                        return hours + minutes / 60.0 + seconds / 3600.0
                                    
                                    holding_times = df[time_in_trade_col].apply(parse_time_in_trade)
                                    holding_time_hours = holding_times.mean() if len(holding_times) > 0 else 0
                                except:
                                    holding_time_hours = 0
                    
                    # Calcular métricas mensuales
                    pct_meses_rentables = 0
                    
                    if open_time_col and open_time_col in df.columns:
                        try:
                            df_mes = df.copy()
                            df_mes[open_time_col] = pd.to_datetime(df_mes[open_time_col], errors='coerce')
                            df_mes = df_mes.dropna(subset=[open_time_col])
                            
                            if len(df_mes) > 0:
                                df_mes['Mes'] = df_mes[open_time_col].dt.to_period('M')
                                beneficios_mensuales = df_mes.groupby('Mes')[profit_col].sum()
                                
                                total_meses = len(beneficios_mensuales)
                                meses_rentables = len(beneficios_mensuales[beneficios_mensuales >= 0])
                                
                                # Calcular % meses rentables
                                pct_meses_rentables = (meses_rentables / total_meses * 100) if total_meses > 0 else 0
                        except:
                            pass
                    
                    # Agregar resultado
                    resultados.append({
                        "Nombre Estrategia": nombre_estrategia,
                        "Activo": activo,
                        "retdd": retdd,
                        "Profit Factor": profit_factor,
                        "Sharpe Ratio": sharpe_ratio,
                        "Net Profit": net_profit,
                        "Max DD": max_dd,
                        "Avg Trade Mensual": avg_trade_mensual,
                        "Max Consec Loss": max_consec_loss,
                        "Winrate %": winrate,
                        "Ulcer Index": ulcer_index,
                        "Z-score": z_score,
                        "R Expectancy R": r_expectancy_r,  # Guardar valor R
                        "R Expectancy $": r_expectancy_dollar,  # Guardar valor en dólares
                        "Calmar": calmar,
                        "R-squared": r_squared,
                        "Holding Time": holding_time_hours,
                        "% Meses Rentables": round(pct_meses_rentables, 1)
                    })
                    
                    # Guardar datos individuales de la estrategia para combinación
                    datos_estrategias.append({
                        "nombre": nombre_estrategia,
                        "activo": activo,
                        "df": df.copy(),
                        "profit_col": profit_col,
                        "open_time_col": open_time_col
                    })
                
                if resultados:
                    # Crear DataFrame con los resultados
                    df_resultados = pd.DataFrame(resultados)
                    
                    # Formatear R Expectancy para mostrar valor R y valor en $ entre paréntesis
                    df_resultados["R Expectancy Formatted"] = df_resultados.apply(
                        lambda row: f"{row['R Expectancy R']:.2f} (${row['R Expectancy $']:.2f})",
                        axis=1
                    )
                    
                    # Eliminar columnas auxiliares y renombrar
                    df_resultados = df_resultados.drop(columns=["R Expectancy R", "R Expectancy $"])
                    df_resultados = df_resultados.rename(columns={"R Expectancy Formatted": "R Expectancy"})
                    
                    # Definir el orden de las columnas
                    column_order = [
                        "Nombre Estrategia", "Activo", "retdd", 
                        "Profit Factor", "Sharpe Ratio", "Net Profit", "Max DD", 
                        "Avg Trade Mensual", "Max Consec Loss", 
                        "Winrate %", "Ulcer Index", "Z-score", "R Expectancy", "Calmar",
                        "R-squared", "Holding Time", "% Meses Rentables"
                    ]
                    df_resultados = df_resultados[column_order]
                    
                    # Ordenar por retdd de forma descendente (mayor a menor)
                    df_resultados = df_resultados.sort_values(by='retdd', ascending=False)
                    
                    # Función para estilizar Calmar según umbrales (igual que en "estrategias por mes")
                    def estilizar_calmar(valor):
                        if isinstance(valor, (int, float)):
                            if valor >= 3.0:
                                return 'background-color: #28a745'  # Verde (Excelente)
                            elif valor >= 2.0:
                                return 'background-color: #fd7e14'  # Naranja (Muy bueno)
                            elif valor >= 1.0:
                                return 'background-color: #ffc107'  # Amarillo (Aceptable)
                            else:
                                return 'background-color: #dc3545'  # Rojo (Débil)
                        return ''
                    
                    # Aplicar estilo solo a la columna Calmar
                    styled_df = df_resultados.style.applymap(estilizar_calmar, subset=['Calmar'])
                    
                    # Formatear columnas numéricas con tooltips en los títulos
                    column_config = {
                        "Nombre Estrategia": st.column_config.TextColumn("Nombre Estrategia"),
                        "Activo": st.column_config.TextColumn("Activo"),
                        "retdd": st.column_config.NumberColumn("retdd", format="%.2f"),
                        "Profit Factor": st.column_config.NumberColumn("Profit Factor", format="%.2f"),
                        "Sharpe Ratio": st.column_config.NumberColumn("Sharpe Ratio", format="%.2f", help="Sharpe Ratio (StrategyQuant):\n\nMide el rendimiento ajustado al riesgo.\n\nFórmula: Sharpe = (R_p - R_f) / σ_p\n• R_p = rentabilidad media del portafolio\n• R_f = tasa libre de riesgo (0 en trading algorítmico)\n• σ_p = desviación estándar de los retornos\n\nInterpretación:\n• > 2.0: Excelente - Rendimiento muy superior al riesgo\n• 1.0 - 2.0: Buena - Buen rendimiento ajustado al riesgo\n• 0.5 - 1.0: Aceptable - Rendimiento moderado\n• < 0.5: Débil - Rendimiento bajo comparado con el riesgo"),
                        "Net Profit": st.column_config.NumberColumn("Net Profit", format="$%.2f"),
                        "Max DD": st.column_config.NumberColumn("Max DD", format="$%.2f"),
                        "Avg Trade Mensual": st.column_config.NumberColumn("Avg Trade Mensual", format="%.1f", help="Promedio de trades por mes"),
                        "Max Consec Loss": st.column_config.NumberColumn("Max Consec Loss", format="%d"),
                        "Winrate %": st.column_config.NumberColumn("Winrate %", format="%.2f%%"),
                        "Ulcer Index": st.column_config.NumberColumn("Ulcer Index", format="%.2f", help="Ulcer Index:\n\nMide la profundidad y duración de los drawdowns en la equity curve.\n\nCuanto menor sea el valor, mejor. Un valor bajo indica drawdowns pequeños y cortos.\n\nInterpretación:\n• < 5: Excelente - Drawdowns muy controlados\n• 5 - 10: Bueno - Drawdowns moderados\n• 10 - 20: Aceptable - Drawdowns considerables\n• > 20: Alto - Drawdowns profundos y prolongados\n\nSe calcula como la raíz cuadrada del promedio de los drawdowns porcentuales al cuadrado."),
                        "Z-score": st.column_config.NumberColumn("Z-score", format="%.2f", help="Z-score (StrategyQuant):\n\nMide la aleatoriedad de las secuencias de ganancias y pérdidas basado en el número de runs (rachas).\n\nUn Z-score cercano a 0 indica que las secuencias son aleatorias (patrón normal).\nUn Z-score positivo alto indica que hay menos runs de lo esperado (más agrupación de resultados similares).\nUn Z-score negativo indica que hay más runs de lo esperado (más alternancia entre ganancias y pérdidas).\n\nInterpretación:\n• |Z| < 1.0: Secuencia aleatoria normal - Patrón esperado en trading\n• |Z| 1.0 - 2.0: Ligeramente no aleatorio - Puede indicar agrupación o alternancia\n• |Z| > 2.0: Significativamente no aleatorio - Patrón claro de agrupación o alternancia\n\nFórmula: Z = (R - E(R)) / sqrt(Var(R)) donde R es el número de runs observados."),
                        "R Expectancy": st.column_config.TextColumn("R Expectancy", help="R Expectancy (Expectativa en R):\n\nMide la rentabilidad esperada por trade ajustada al riesgo unitario.\n\nEl valor R representa cuántas veces el riesgo esperas ganar por cada trade.\n\nInterpretación:\n• > 1.0 R: Excelente - Ganas más de 1 vez el riesgo por trade\n• 0.5 - 1.0 R: Muy bueno - Estrategias muy rentables\n• 0.25 - 0.5 R: Bueno - Estrategias rentables\n• 0.0 - 0.25 R: Aceptable - Expectativa positiva pero baja\n• < 0.0: Negativo - No rentable a largo plazo\n\nSe muestra en formato R (decimal) y en dólares entre paréntesis."),
                        "Calmar": st.column_config.NumberColumn("Calmar", format="%.2f", help="Calmar Ratio (Return / Max Drawdown):\n\n🟢 > 3.0: Excelente - Ratio excepcional, muy raro en trading real. Ideal para prop firms y fondeos.\n\n🟠 2.0 - 3.0: Muy bueno - Estrategias robustas y consistentes con excelente gestión de riesgo.\n\n🟡 1.0 - 2.0: Aceptable - Nivel estándar para estrategias operativas normales.\n\n🔴 < 1.0: Débil - El riesgo de drawdown es demasiado alto comparado con la rentabilidad obtenida.\n\nMide la relación entre rentabilidad y drawdown máximo. Valores más altos indican mejor eficiencia riesgo-rendimiento."),
                        "R-squared": st.column_config.NumberColumn("R-squared", format="%.4f", help="R-squared (Smoothness de la curva de capital):\n\nMide qué tan suave y consistente es la curva de equity. Indica qué tan bien se ajusta la curva de capital a una línea recta (tendencia).\n\nDiferencia con R Expectancy:\n• R Expectancy: Te dice cuánto podrías ganar por trade asumiendo X riesgo → clave para rentabilidad.\n• R-squared: Te dice qué tan suave es la curva de beneficios → clave para robustez/consistencia.\n\nInterpretación:\n• > 0.95: Excelente - Curva muy suave, alta consistencia\n• 0.90 - 0.95: Muy bueno - Curva suave, buena consistencia\n• 0.80 - 0.90: Bueno - Curva aceptablemente suave\n• 0.70 - 0.80: Aceptable - Curva con algunas variaciones\n• < 0.70: Bajo - Curva con muchas variaciones, menor consistencia\n\nUna estrategia ideal tiene: R Expectancy alta + R² alto."),
                        "Holding Time": st.column_config.NumberColumn("Holding Time", format="%.2f", help="Holding Time (Duración media de las operaciones):\n\nMide el tiempo promedio que las operaciones permanecen abiertas antes de cerrarse.\n\nSe muestra en horas.\n\nInterpretación:\n• Valores bajos (< 1 hora): Operaciones muy cortas, scalping\n• Valores medios (1-24 horas): Operaciones intradía\n• Valores altos (> 24 horas): Operaciones de swing o posición\n\nÚtil para entender el estilo de trading de la estrategia y evaluar la exposición al riesgo temporal."),
                        "% Meses Rentables": st.column_config.NumberColumn("% Meses Rentables", format="%.1f%%", help="Porcentaje de meses con beneficio >= 0")
                    }
                    
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config=column_config
                    )
                    
                    # Sección de Crear Portafolio
                    if len(resultados) > 1:
                        # Inicializar session_state para portafolios
                        if 'portafolios_ea' not in st.session_state:
                            st.session_state.portafolios_ea = {}
                        
                        st.markdown("""
                        <div style="margin-top: 2rem; margin-bottom: 1rem;">
                            <h4>Crear Portafolio</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Lista de estrategias con formato "Nombre Estrategia - Activo" para el multiselect
                        df_resultados['Estrategia_Completa'] = df_resultados['Nombre Estrategia'] + ' - ' + df_resultados['Activo']
                        nombres_estrategias = df_resultados['Estrategia_Completa'].tolist()
                        
                        # Formulario para crear nuevo portafolio
                        nombre_portafolio = st.text_input(
                            "Nombre del portafolio:",
                            key="nombre_portafolio_nuevo",
                            help="Ingresa un nombre único para identificar este portafolio"
                        )
                        
                        estrategias_seleccionadas = st.multiselect(
                            "Selecciona las estrategias que deseas incluir:",
                            options=nombres_estrategias,
                            default=[],
                            key="estrategias_portafolio_nuevo",
                            help="Selecciona una o más estrategias para incluir en el portafolio"
                        )
                        
                        crear_portafolio = st.button("Crear Portafolio", use_container_width=True)
                        
                        if crear_portafolio:
                            if not nombre_portafolio or nombre_portafolio.strip() == "":
                                st.error("⚠️ Por favor, ingresa un nombre para el portafolio.")
                            elif nombre_portafolio in st.session_state.portafolios_ea:
                                st.error(f"⚠️ Ya existe un portafolio con el nombre '{nombre_portafolio}'. Por favor, elige otro nombre.")
                            elif not estrategias_seleccionadas:
                                st.error("⚠️ Por favor, selecciona al menos una estrategia.")
                            else:
                                try:
                                    # Filtrar los datos de las estrategias seleccionadas
                                    estrategias_filtradas = []
                                    for estrategia_sel in estrategias_seleccionadas:
                                        # Formato: "Nombre Estrategia - Activo"
                                        partes = estrategia_sel.split(' - ')
                                        if len(partes) == 2:
                                            nombre_estrategia_sel = partes[0]
                                            activo_sel = partes[1]
                                            estrategias_filtradas.append((nombre_estrategia_sel, activo_sel))
                                    
                                    if not estrategias_filtradas:
                                        st.error("No se pudieron extraer las estrategias seleccionadas. Verifica el formato.")
                                    else:
                                        # Combinar los DataFrames de las estrategias seleccionadas
                                        dfs_combinados = []
                                        for nombre, activo in estrategias_filtradas:
                                            for datos_estrategia in datos_estrategias:
                                                if datos_estrategia['nombre'] == nombre and datos_estrategia['activo'] == activo:
                                                    dfs_combinados.append(datos_estrategia['df'].copy())
                                                    break
                                        
                                        if not dfs_combinados:
                                            st.warning("No se encontraron datos para las estrategias seleccionadas.")
                                        else:
                                            # Concatenar todos los DataFrames
                                            df_combinado = pd.concat(dfs_combinados, ignore_index=True)
                                            
                                            # Obtener las columnas comunes (usar las del primer DataFrame)
                                            profit_col = datos_estrategias[0]['profit_col']
                                            open_time_col = datos_estrategias[0]['open_time_col']
                                            
                                            # Ordenar por tiempo si está disponible
                                            if open_time_col and open_time_col in df_combinado.columns:
                                                try:
                                                    df_combinado[open_time_col] = pd.to_datetime(df_combinado[open_time_col], errors='coerce')
                                                    df_combinado = df_combinado.sort_values(by=open_time_col).dropna(subset=[open_time_col])
                                                except:
                                                    df_combinado = df_combinado.sort_index()
                                            else:
                                                df_combinado = df_combinado.sort_index()
                                            
                                            # Calcular todas las métricas combinadas
                                            net_profit_comb = df_combinado[profit_col].sum()
                                            
                                            # Profit Factor
                                            ganancias_comb = df_combinado[df_combinado[profit_col] > 0][profit_col].sum()
                                            perdidas_comb = abs(df_combinado[df_combinado[profit_col] < 0][profit_col].sum())
                                            profit_factor_comb = ganancias_comb / perdidas_comb if perdidas_comb > 0 else float('inf') if ganancias_comb > 0 else 0
                                            
                                            # Winrate
                                            trades_ganadores_comb = len(df_combinado[df_combinado[profit_col] > 0])
                                            trades_perdedores_comb = len(df_combinado[df_combinado[profit_col] < 0])
                                            total_trades_comb = len(df_combinado)
                                            winrate_comb = (trades_ganadores_comb / total_trades_comb * 100) if total_trades_comb > 0 else 0
                                            
                                            # AvgWin / AvgLoss
                                            avg_win_comb = df_combinado[df_combinado[profit_col] > 0][profit_col].mean() if trades_ganadores_comb > 0 else 0
                                            avg_loss_comb = abs(df_combinado[df_combinado[profit_col] < 0][profit_col].mean()) if trades_perdedores_comb > 0 else 0
                                            
                                            # Max Consecutive Loss
                                            max_consec_loss_comb = 0
                                            current_consec_loss = 0
                                            for profit in df_combinado[profit_col]:
                                                if profit < 0:
                                                    current_consec_loss += 1
                                                    max_consec_loss_comb = max(max_consec_loss_comb, current_consec_loss)
                                                else:
                                                    current_consec_loss = 0
                                            
                                            # Calcular Drawdown
                                            df_combinado['equity'] = df_combinado[profit_col].cumsum()
                                            df_combinado['cummax'] = df_combinado['equity'].cummax()
                                            df_combinado['drawdown'] = df_combinado['cummax'] - df_combinado['equity']
                                            max_dd_comb = df_combinado['drawdown'].max() if len(df_combinado) > 0 else 0
                                            
                                            # retdd
                                            retdd_comb = (net_profit_comb / max_dd_comb) if max_dd_comb > 0 else 0
                                            
                                            # Ulcer Index
                                            equity_for_ui_comb = df_combinado['equity']
                                            cummax_equity_comb = equity_for_ui_comb.cummax()
                                            cummax_safe_comb = cummax_equity_comb.replace(0, 1)
                                            dd_pct_comb = ((cummax_equity_comb - equity_for_ui_comb) / cummax_safe_comb * 100).fillna(0)
                                            dd_pct_positive_comb = dd_pct_comb.clip(lower=0)
                                            if len(df_combinado) > 0:
                                                ulcer_index_comb = np.sqrt((dd_pct_positive_comb ** 2).sum() / len(df_combinado))
                                            else:
                                                ulcer_index_comb = 0
                                            
                                            # Z-score (fórmula StrategyQuant basada en runs)
                                            if total_trades_comb > 0:
                                                # N = número total de operaciones
                                                N_comb = total_trades_comb
                                                # W = número de operaciones ganadoras
                                                W_comb = trades_ganadores_comb
                                                # L = número de operaciones perdedoras
                                                L_comb = trades_perdedores_comb
                                                # P = probabilidad de ganar
                                                P_comb = W_comb / N_comb if N_comb > 0 else 0
                                                
                                                # Calcular R = número total de runs (rachas)
                                                if len(df_combinado) > 0:
                                                    # Ordenar por tiempo si está disponible
                                                    df_zscore_comb = df_combinado.copy()
                                                    if open_time_col and open_time_col in df_zscore_comb.columns:
                                                        try:
                                                            df_zscore_comb[open_time_col] = pd.to_datetime(df_zscore_comb[open_time_col], errors='coerce')
                                                            df_zscore_comb = df_zscore_comb.sort_values(open_time_col).dropna(subset=[open_time_col])
                                                        except:
                                                            pass
                                                    
                                                    # Crear secuencia de ganancias (1) y pérdidas (0)
                                                    profits_comb = df_zscore_comb[profit_col].values
                                                    secuencia_comb = [1 if p > 0 else 0 for p in profits_comb]
                                                    
                                                    # Contar runs: cada cambio en la secuencia es un nuevo run
                                                    R_comb = 1  # Siempre hay al menos 1 run
                                                    for i in range(1, len(secuencia_comb)):
                                                        if secuencia_comb[i] != secuencia_comb[i-1]:
                                                            R_comb += 1
                                                else:
                                                    R_comb = 0
                                                
                                                # Calcular Z-score solo si hay suficientes datos
                                                if N_comb > 1 and P_comb > 0 and P_comb < 1:
                                                    # E(R) = 1 + 2 * (N-1) * P * (1-P)
                                                    E_R_comb = 1 + 2 * (N_comb - 1) * P_comb * (1 - P_comb)
                                                    
                                                    # Var(R) = 2 * (N-1) * P * (1-P) * (2*P*(1-P))
                                                    Var_R_comb = 2 * (N_comb - 1) * P_comb * (1 - P_comb) * (2 * P_comb * (1 - P_comb))
                                                    
                                                    # Z = (R - E(R)) / sqrt(Var(R))
                                                    if Var_R_comb > 0:
                                                        z_score_comb = (R_comb - E_R_comb) / np.sqrt(Var_R_comb)
                                                    else:
                                                        z_score_comb = 0
                                                else:
                                                    z_score_comb = 0
                                            else:
                                                z_score_comb = 0
                                            
                                            # Sharpe Ratio (fórmula StrategyQuant)
                                            # Sharpe = (R_p - R_f) / σ_p
                                            # Donde R_p = rentabilidad media, R_f = 0 (tasa libre de riesgo), σ_p = desviación estándar
                                            if len(df_combinado) > 1:
                                                # Usar retornos por operación (profit values)
                                                returns_comb = df_combinado[profit_col].values
                                                mean_return_comb = returns_comb.mean()  # R_p
                                                std_return_comb = returns_comb.std()     # σ_p
                                                
                                                # R_f = 0 (tasa libre de riesgo, asumida en 0 para trading algorítmico)
                                                R_f_comb = 0
                                                
                                                if std_return_comb > 0:
                                                    # Sharpe Ratio = (R_p - R_f) / σ_p
                                                    sharpe_ratio_comb = (mean_return_comb - R_f_comb) / std_return_comb
                                                else:
                                                    sharpe_ratio_comb = 0
                                            else:
                                                sharpe_ratio_comb = 0
                                            
                                            # Avg Trade Mensual (número promedio de trades por mes)
                                            total_trades_comb = len(df_combinado)
                                            if open_time_col and open_time_col in df_combinado.columns:
                                                try:
                                                    fecha_inicio_comb = df_combinado[open_time_col].min()
                                                    fecha_fin_comb = df_combinado[open_time_col].max()
                                                    meses_comb = (fecha_fin_comb - fecha_inicio_comb).days / 30.44
                                                    avg_trade_mensual_comb = total_trades_comb / meses_comb if meses_comb > 0 else total_trades_comb
                                                except:
                                                    avg_trade_mensual_comb = total_trades_comb
                                            else:
                                                avg_trade_mensual_comb = total_trades_comb
                                            
                                            # Calmar Ratio (CAGR / Max DD %)
                                            # MetaTrader calcula usando CAGR anualizado dividido por max drawdown porcentual
                                            if max_dd_comb > 0:
                                                # Obtener balance inicial y final si está disponible
                                                balance_inicial_calmar_comb = None
                                                balance_final_calmar_comb = None
                                                balance_col_comb_calmar = None
                                                for col in df_combinado.columns:
                                                    if col.lower() == 'balance':
                                                        balance_col_comb_calmar = col
                                                        break
                                                
                                                if balance_col_comb_calmar and balance_col_comb_calmar in df_combinado.columns:
                                                    df_balance_calmar_comb = df_combinado.copy()
                                                    df_balance_calmar_comb[balance_col_comb_calmar] = pd.to_numeric(df_balance_calmar_comb[balance_col_comb_calmar], errors='coerce')
                                                    df_balance_calmar_comb = df_balance_calmar_comb.dropna(subset=[balance_col_comb_calmar])
                                                    if len(df_balance_calmar_comb) > 0:
                                                        # Ordenar por tiempo si está disponible
                                                        if open_time_col and open_time_col in df_balance_calmar_comb.columns:
                                                            df_balance_calmar_comb[open_time_col] = pd.to_datetime(df_balance_calmar_comb[open_time_col], errors='coerce')
                                                            df_balance_calmar_comb = df_balance_calmar_comb.sort_values(open_time_col).dropna(subset=[open_time_col])
                                                        
                                                        balance_inicial_calmar_comb = df_balance_calmar_comb[balance_col_comb_calmar].iloc[0] - df_balance_calmar_comb[profit_col].iloc[0]
                                                        balance_final_calmar_comb = df_balance_calmar_comb[balance_col_comb_calmar].iloc[-1]
                                                        
                                                        # Calcular período en años
                                                        if open_time_col and open_time_col in df_balance_calmar_comb.columns:
                                                            fecha_inicio_calmar_comb = df_balance_calmar_comb[open_time_col].iloc[0]
                                                            fecha_fin_calmar_comb = df_balance_calmar_comb[open_time_col].iloc[-1]
                                                            dias_totales_calmar_comb = (fecha_fin_calmar_comb - fecha_inicio_calmar_comb).days
                                                            años_calmar_comb = dias_totales_calmar_comb / 365.25
                                                        else:
                                                            años_calmar_comb = 1  # Default a 1 año si no hay fechas
                                                        
                                                        if balance_inicial_calmar_comb > 0 and balance_final_calmar_comb > 0 and años_calmar_comb > 0:
                                                            # Calcular CAGR (Compound Annual Growth Rate)
                                                            # CAGR = ((Valor Final / Valor Inicial) ^ (1 / años)) - 1
                                                            ratio_valor_comb = balance_final_calmar_comb / balance_inicial_calmar_comb
                                                            if ratio_valor_comb > 0:
                                                                cagr_comb = (ratio_valor_comb ** (1 / años_calmar_comb) - 1) * 100  # En porcentaje
                                                            else:
                                                                cagr_comb = 0
                                                            
                                                            # Calcular max drawdown porcentual
                                                            df_calmar_comb = df_balance_calmar_comb.copy()
                                                            df_calmar_comb['cummax'] = df_calmar_comb[balance_col_comb_calmar].cummax()
                                                            # Drawdown porcentual máximo
                                                            df_calmar_comb['dd_pct'] = ((df_calmar_comb['cummax'] - df_calmar_comb[balance_col_comb_calmar]) / df_calmar_comb['cummax'] * 100).fillna(0)
                                                            max_dd_pct_comb = df_calmar_comb['dd_pct'].max()
                                                            
                                                            if max_dd_pct_comb > 0:
                                                                calmar_comb = cagr_comb / max_dd_pct_comb
                                                            else:
                                                                calmar_comb = 0
                                                        else:
                                                            # Fallback: usar valores absolutos
                                                            calmar_comb = (net_profit_comb / max_dd_comb) if max_dd_comb > 0 else 0
                                                    else:
                                                        calmar_comb = 0
                                                else:
                                                    # Si no hay balance, usar cálculo simple
                                                    calmar_comb = (net_profit_comb / max_dd_comb) if max_dd_comb > 0 else 0
                                            else:
                                                calmar_comb = 0
                                            
                                            # R Expectancy
                                            if total_trades_comb > 0 and avg_loss_comb > 0:
                                                r_expectancy_r_comb = (winrate_comb / 100 * (avg_win_comb / avg_loss_comb)) - ((100 - winrate_comb) / 100 * 1)
                                                r_expectancy_dollar_comb = (winrate_comb / 100 * avg_win_comb) - ((100 - winrate_comb) / 100 * avg_loss_comb)
                                            else:
                                                r_expectancy_r_comb = 0
                                                r_expectancy_dollar_comb = 0
                                            
                                            # R-squared
                                            if len(df_combinado) > 1:
                                                equity_curve_comb = df_combinado['equity'].values
                                                x_comb = np.arange(len(equity_curve_comb))
                                                if len(equity_curve_comb) > 1:
                                                    x_mean_comb = x_comb.mean()
                                                    y_mean_comb = equity_curve_comb.mean()
                                                    numerator_comb = ((x_comb - x_mean_comb) * (equity_curve_comb - y_mean_comb)).sum()
                                                    denominator_comb = ((x_comb - x_mean_comb) ** 2).sum()
                                                    if denominator_comb > 0:
                                                        slope_comb = numerator_comb / denominator_comb
                                                        intercept_comb = y_mean_comb - slope_comb * x_mean_comb
                                                        y_pred_comb = intercept_comb + slope_comb * x_comb
                                                        ss_res_comb = ((equity_curve_comb - y_pred_comb) ** 2).sum()
                                                        ss_tot_comb = ((equity_curve_comb - y_mean_comb) ** 2).sum()
                                                        if ss_tot_comb > 0:
                                                            r_squared_comb = 1 - (ss_res_comb / ss_tot_comb)
                                                        else:
                                                            r_squared_comb = 0
                                                    else:
                                                        r_squared_comb = 0
                                                else:
                                                    r_squared_comb = 0
                                            else:
                                                r_squared_comb = 0
                                            
                                            # Holding Time (simplificado, usar 0 si no hay datos)
                                            holding_time_hours_comb = 0
                                            
                                            # % Meses Rentables
                                            pct_meses_rentables_comb = 0
                                            if open_time_col and open_time_col in df_combinado.columns:
                                                try:
                                                    df_mes_comb = df_combinado.copy()
                                                    df_mes_comb['Mes'] = df_mes_comb[open_time_col].dt.to_period('M')
                                                    beneficios_mensuales_comb = df_mes_comb.groupby('Mes')[profit_col].sum()
                                                    total_meses_comb = len(beneficios_mensuales_comb)
                                                    meses_rentables_comb = len(beneficios_mensuales_comb[beneficios_mensuales_comb >= 0])
                                                    pct_meses_rentables_comb = (meses_rentables_comb / total_meses_comb * 100) if total_meses_comb > 0 else 0
                                                except:
                                                    pass
                                            
                                            # Crear DataFrame con las métricas combinadas
                                            resultado_comb = {
                                                "Nombre Estrategia": nombre_portafolio,
                                                "Activo": "Múltiples",
                                                "retdd": retdd_comb,
                                                "Profit Factor": profit_factor_comb,
                                                "Sharpe Ratio": sharpe_ratio_comb,
                                                "Net Profit": net_profit_comb,
                                                "Max DD": max_dd_comb,
                                                "Avg Trade Mensual": avg_trade_mensual_comb,
                                                "Max Consec Loss": max_consec_loss_comb,
                                                "Winrate %": winrate_comb,
                                                "Ulcer Index": ulcer_index_comb,
                                                "Z-score": z_score_comb,
                                                "R Expectancy": f"{r_expectancy_r_comb:.2f} (${r_expectancy_dollar_comb:.2f})",
                                                "Calmar": calmar_comb,
                                                "R-squared": r_squared_comb,
                                                "Holding Time": holding_time_hours_comb,
                                                "% Meses Rentables": round(pct_meses_rentables_comb, 1)
                                            }
                                            
                                            df_resultado_comb = pd.DataFrame([resultado_comb])
                                            
                                            # Guardar el portafolio en session_state
                                            st.session_state.portafolios_ea[nombre_portafolio] = {
                                                'resultado': resultado_comb,
                                                'estrategias': estrategias_seleccionadas,
                                                'df_resultado': df_resultado_comb,
                                                'df_combinado': df_combinado.copy(),
                                                'open_time_col': open_time_col,
                                                'profit_col': profit_col
                                            }
                                            
                                            st.success(f"✅ Portafolio '{nombre_portafolio}' creado exitosamente!")
                                            st.rerun()
                                
                                except Exception as e:
                                    st.error(f"❌ Error al crear el portafolio: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc())
                        
                        # Mostrar portafolios creados
                        if st.session_state.portafolios_ea:
                            st.markdown("""
                            <div style="margin-top: 2rem; margin-bottom: 1rem;">
                                <h4>Portafolios Creados</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Crear lista de portafolios para mostrar
                            for nombre_portafolio, datos_portafolio in st.session_state.portafolios_ea.items():
                                with st.expander(f"📊 {nombre_portafolio}", expanded=False):
                                    # Mostrar estrategias incluidas con botón de eliminar a la derecha
                                    col_estrategias, col_eliminar = st.columns([4, 1])
                                    with col_estrategias:
                                        st.markdown(f"**Estrategias:** {', '.join(datos_portafolio['estrategias'])}")
                                    with col_eliminar:
                                        if st.button(f"Eliminar", key=f"eliminar_{nombre_portafolio}", use_container_width=True):
                                            del st.session_state.portafolios_ea[nombre_portafolio]
                                            st.rerun()
                                    
                                    # Mostrar tabla del portafolio
                                    df_resultado_portafolio = datos_portafolio['df_resultado']
                                    styled_df_portafolio = df_resultado_portafolio.style.applymap(estilizar_calmar, subset=['Calmar'])
                                    
                                    st.dataframe(
                                        styled_df_portafolio,
                                        use_container_width=True,
                                        hide_index=True,
                                        column_config=column_config
                                    )
                            
                            # Gráfico de Equity de todos los portafolios
                            st.markdown("""
                            <div style="margin-top: 2rem; margin-bottom: 1rem;">
                                <h4>Evolución del Equity - Portafolios</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Crear gráfico de equity para todos los portafolios
                            fig_equity = go.Figure()
                            
                            for nombre_portafolio, datos_portafolio in st.session_state.portafolios_ea.items():
                                if 'df_combinado' in datos_portafolio and 'open_time_col' in datos_portafolio and 'profit_col' in datos_portafolio:
                                    try:
                                        df_port = datos_portafolio['df_combinado'].copy()
                                        open_time_col_port = datos_portafolio['open_time_col']
                                        profit_col_port = datos_portafolio['profit_col']
                                        
                                        if open_time_col_port and open_time_col_port in df_port.columns and profit_col_port in df_port.columns:
                                            # Ordenar por tiempo
                                            df_port[open_time_col_port] = pd.to_datetime(df_port[open_time_col_port], errors='coerce')
                                            df_port = df_port.sort_values(by=open_time_col_port).dropna(subset=[open_time_col_port])
                                            
                                            if len(df_port) > 0:
                                                # Calcular equity (beneficio acumulado)
                                                df_port['equity'] = df_port[profit_col_port].cumsum()
                                                
                                                # Agregar línea al gráfico
                                                fig_equity.add_trace(go.Scatter(
                                                    x=df_port[open_time_col_port],
                                                    y=df_port['equity'],
                                                    mode='lines+markers',
                                                    name=nombre_portafolio,
                                                    line=dict(width=2),
                                                    marker=dict(size=4),
                                                    hovertemplate=
                                                    "<b>Portafolio:</b> %{fullData.name}<br>" +
                                                    "<b>Fecha:</b> %{x|%d-%m-%Y %H:%M}<br>" +
                                                    "<b>Equity:</b> $%{y:,.2f}<extra></extra>"
                                                ))
                                    except Exception as e:
                                        # Si hay error con un portafolio, continuar con los demás
                                        continue
                            
                            if len(fig_equity.data) > 0:
                                fig_equity.update_layout(
                                    title="Evolución del Equity por Portafolio",
                                    xaxis_title="Fecha",
                                    yaxis_title="Equity ($)",
                                    hovermode='x unified',
                                    showlegend=True,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    ),
                                    height=400,
                                    template="plotly_white",
                                    plot_bgcolor='white',
                                    paper_bgcolor='white'
                                )
                                
                                st.plotly_chart(fig_equity, use_container_width=True)
                            else:
                                st.info("No hay datos suficientes para mostrar el gráfico de equity.")
                else:
                    st.warning("⚠️ No se pudieron procesar los archivos CSV")
                    
            except Exception as e:
                st.error(f"❌ Error al procesar los archivos CSV: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with sub_tab3:
        # Tarjeta de análisis de Prop Firm
        st.markdown("""
        <div class="card">
            <div class="card-title">Evolución Detallada de Challenges</div>
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
                        index=1,
                        format_func=lambda x: f"{x}%",
                        help="Porcentaje de la cuenta que se arriesga por operación"
                    )
                
                with col2:
                    tamaño_cuenta = st.selectbox(
                        "💵 Tamaño de cuenta de fondeo",
                        options=[10000, 25000, 50000, 100000, 200000],
                        index=3,
                        format_func=lambda x: f"${x:,}",
                        help="Tamaño de la cuenta de fondeo"
                    )
                
                # Segunda fila de selectores para objetivos y límites
                st.markdown("---")
                st.markdown("### Objetivos de Ganancia y Límites de Drawdown")
                
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
                        index=2,
                        format_func=lambda x: f"{x}%",
                        help="Drawdown máximo permitido en un solo día. Si pierdes más, suspendes el examen"
                    )
                
                with col6:
                    drawdown_maximo_total = st.selectbox(
                        "📊 Drawdown máximo total",
                        options=[5, 6, 7, 8, 9, 10, 12, 15],
                        index=5,
                        format_func=lambda x: f"{x}%",
                        help="Drawdown máximo total permitido durante toda la evaluación. No se puede superar en ninguna fase"
                    )
                
                # Información sobre el análisis
                st.markdown("---")
                st.markdown("### Configuración Completada")
                
                # Mostrar resumen de la configuración
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #495057;">💰 Configuración de Riesgo</h4>
                    <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">
                    <strong>Tamaño de cuenta:</strong> ${tamaño_cuenta:,}<br>
                    <strong>Riesgo por operación:</strong> {riesgo_por_operacion}%<br>
                    <strong>Factor de escalado:</strong> {riesgo_por_operacion/2:.1f}x
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #495057;">🎯 Objetivos y Límites</h4>
                    <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">
                    <strong>Fase 1:</strong> {porcentaje_fase1}%<br>
                    <strong>Fase 2:</strong> {porcentaje_fase2}%<br>
                    <strong>DD diario:</strong> {drawdown_maximo_diario}%<br>
                    <strong>DD total:</strong> {drawdown_maximo_total}%
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                        
                # Información sobre qué hace el análisis
                st.info("""
                🔍 **¿Qué hace este análisis?**
                
                Este análisis te mostrará la evolución trade por trade de múltiples challenges consecutivos, 
                permitiendo ver exactamente cómo progresa el balance y en qué momento se aprueba o suspende 
                cada examen. Podrás seleccionar diferentes challenges para analizar en detalle.
                """)
                
                # Estilo personalizado para el botón de análisis
                st.markdown("""
                <style>
                .analyze-button {
                    background: linear-gradient(45deg, #28a745, #20c997) !important;
                    color: white !important;
                    border: none !important;
                    border-radius: 8px !important;
                    padding: 12px 24px !important;
                    font-size: 16px !important;
                    font-weight: 600 !important;
                    text-align: center !important;
                    cursor: pointer !important;
                    transition: all 0.3s ease !important;
                    box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3) !important;
                    margin: 20px 0 !important;
                    width: 100% !important;
                }
                
                .analyze-button:hover {
                    background: linear-gradient(45deg, #20c997, #17a2b8) !important;
                    transform: translateY(-2px) !important;
                    box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4) !important;
                }
                
                .analyze-button:active {
                    transform: translateY(0) !important;
                    box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3) !important;
                }
                
                /* Estilo para el botón de Streamlit */
                div[data-testid="stButton"] > button[kind="primary"] {
                    background: linear-gradient(45deg, #28a745, #20c997) !important;
                    color: white !important;
                    border: none !important;
                    border-radius: 8px !important;
                    padding: 12px 24px !important;
                    font-size: 16px !important;
                    font-weight: 600 !important;
                    text-align: center !important;
                    cursor: pointer !important;
                    transition: all 0.3s ease !important;
                    box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3) !important;
                    margin: 20px 0 !important;
                    width: 100% !important;
                }
                
                div[data-testid="stButton"] > button[kind="primary"]:hover {
                    background: linear-gradient(45deg, #20c997, #17a2b8) !important;
                    transform: translateY(-2px) !important;
                    box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4) !important;
                }
                
                div[data-testid="stButton"] > button[kind="primary"]:active {
                    transform: translateY(0) !important;
                    box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3) !important;
                }
                </style>
                        """, unsafe_allow_html=True)
                        
                # Botón para ejecutar el análisis
                if st.button("🚀 Analizar Evolución Detallada", type="primary"):
                    # Configurar parámetros para el análisis
                    config = {
                        'account_size': tamaño_cuenta,
                        'risk_per_trade': riesgo_por_operacion,
                        'phase1_target': porcentaje_fase1,
                        'phase2_target': porcentaje_fase2,
                        'daily_dd_limit': drawdown_maximo_diario,
                        'total_dd_limit': drawdown_maximo_total
                    }
                    
                    # Mostrar la interfaz de evolución detallada
                    display_evolution_interface(df_csv, config)
                        
            except Exception as e:
                st.error(f"❌ Error al procesar el archivo CSV: {str(e)}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #6c757d;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📈</div>
                <h3 style="margin-bottom: 1rem;">Evolución Detallada de Challenges</h3>
                <p style="margin-bottom: 1rem;">
                    Sube tu archivo CSV para ver la evolución trade por trade de cada challenge, 
                    permitiendo analizar exactamente cómo progresa el balance y en qué momento 
                    se aprueba o suspende cada examen.
                </p>
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                    <h4 style="margin: 0 0 0.5rem 0;">🔍 Características del análisis:</h4>
                    <ul style="margin: 0; font-size: 0.9rem; text-align: left;">
                        <li>Seguimiento trade por trade de cada challenge</li>
                        <li>Visualización del balance en tiempo real</li>
                        <li>Detección automática de límites de drawdown</li>
                        <li>Gráficos interactivos con líneas de referencia</li>
                        <li>Resumen completo de todos los challenges</li>
                    </ul>
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
        "codigo": "IMOX20",
        "descripcion": "Accede a hasta $200,000 en capital con condiciones de trading flexibles y sin límite de tiempo. Perfecto para traders principiantes.",
        "url": "https://thetradingpit.com",
        "boton_texto": "Ir a The Trading Pit",
        "tiene_timer": False
    },
    {
        "id": "ttp",
        "nombre": "Darwinex",
        "logo": "https://cdn.document360.io/logo/5f0936e1-6677-439a-8c1d-7fb2b7944358/2c45aeb889744ddf8ade39331d5b2a85-Dzero-logo-hor.svg",
        "badge": "⭐ NUEVO",
        "descuento": "20% DESCUENTO",
        "codigo": "IMOXCODE",
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
    
    return f'<div class="promo-card-full" style="cursor:pointer;"><div class="promo-image-section"><img src="{promo["logo"]}" alt="{promo["nombre"]}" class="promo-main-logo"></div><div class="promo-content-section"><div class="promo-info"><div class="promo-discount">{promo["descuento"]}</div><div class="promo-code"><span>Código: </span><code id="{promo["id"]}-code">{promo["codigo"]}</code><button class="copy-btn" onclick="copyCode(\'{promo["id"]}-code\')">📋</button></div></div><div class="promo-actions">{timer_html}<a href="{promo["url"]}" target="_blank" class="promo-link"></a></div></div></div>'

# Tab de Prop Firms (oculto)
# with tab2:
#     # Generar promociones dinámicamente desde el array
#     for promo in promociones:
#         st.markdown(generar_promo_html(promo), unsafe_allow_html=True)

# Tab de Ranking (oculto)
# with tab3:
#     # Crear tabs anidados dentro del tab 3
#     ranking_sub_tab1, ranking_sub_tab2 = st.tabs(["Ranking", "Añadir Cobro"])
    
    #with ranking_sub_tab1:
        # Selector de mes con navegación usando Streamlit
        #if 'selected_month' not in st.session_state:
            #st.session_state.selected_month = 0  # 0 = mes actual
        
        # Obtener fecha actual
        #from datetime import datetime, timedelta
        #current_date = datetime.now()
        #target_date = current_date + timedelta(days=30 * st.session_state.selected_month)
        
        #month_names = [
            #"Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
            #"Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
        #]
        
        # Crear layout para el selector de mes
        #col_prev, col_month, col_next = st.columns([1, 3, 1])
        
        #with col_prev:
            #st.markdown("""
            #<div style="display: flex; align-items: center; justify-content: flex-end; height: 100%; width: 100%;">
            #""", unsafe_allow_html=True)
            #if st.button("←", key="prev_month", help="Mes anterior"):
                #st.session_state.selected_month -= 1
                #st.rerun()
            #st.markdown("</div>", unsafe_allow_html=True)
        
        #with col_month:
            #month_name = month_names[target_date.month - 1]
            #year = target_date.year
            #st.markdown(f"""
            #<div class="card-month"><h2>{month_name} {year}</h2></div>
            #""", unsafe_allow_html=True)
        
        #with col_next:
            #st.markdown("""
            #<div style="display: flex; align-items: center; justify-content: center; height: 100%;">
            #""", unsafe_allow_html=True)
            #if st.button("→", key="next_month", help="Mes siguiente"):
                #st.session_state.selected_month += 1
                #st.rerun()
            #st.markdown("</div>", unsafe_allow_html=True)
        
        # Obtener datos del backend con indicador de carga
        #with st.spinner("Cargando datos del ranking..."):
            #payouts_data = get_payouts_from_api()
            #users_data = get_users_from_api()
        
        # Validar que los datos estén disponibles
        #if not payouts_data or not isinstance(payouts_data, list):
            #st.warning("⚠️ No se pudieron cargar los datos de payouts. Por favor, verifica la conexión con el backend.")
            #st.stop()
        
        #if not users_data or not isinstance(users_data, list):
            #st.warning("⚠️ No se pudieron cargar los datos de usuarios. Por favor, verifica la conexión con el backend.")
            #st.stop()
        
        # Crear rankings separados por rango
        #global_ranking = create_monthly_payout_ranking(payouts_data, users_data, st.session_state.selected_month)
        #gold_ranking = create_monthly_payout_ranking(payouts_data, users_data, st.session_state.selected_month, 'gold')
        #silver_ranking = create_monthly_payout_ranking(payouts_data, users_data, st.session_state.selected_month, 'silver')
        
        # Crear tres columnas para los paneles de ranking (responsive)
        #col1, col2, col3 = st.columns([1, 1, 1])
        
        #with col1:
            # Crear columnas para título y botón
            #title_col, share_col = st.columns([4, 1])
            
            #with title_col:
                #st.markdown("""
                #<div class="card ranking-panel">
                    #<div class="card-title">🌍 Ranking Global</div>
                #</div>
                #""", unsafe_allow_html=True)
            
            #with share_col:
                # Preparar datos del ranking
                #ranking_data = []
                #if not global_ranking.empty:
                    #for _, row in global_ranking.head(10).iterrows():
                        #ranking_data.append((
                            #row['position'],
                            #row['name'],
                            #row['total_payout_formatted']
                        #))
                
                # Función para generar imagen
                #def generate_ranking_image(ranking_data):
                    #from PIL import Image, ImageDraw, ImageFont
                    #import io
                    
                    #width, height = 1080, 1920
                    #img = Image.new('RGB', (width, height), color='#ffffff')
                    #draw = ImageDraw.Draw(img)
                    
                    #try:
                        #title_font = ImageFont.truetype("Arial Bold", 80)
                        #subtitle_font = ImageFont.truetype("Arial", 50)
                        #name_font = ImageFont.truetype("Arial", 45)
                        #score_font = ImageFont.truetype("Arial Bold", 40)
                        #total_font = ImageFont.truetype("Arial Bold", 60)
                        #medal_font = ImageFont.truetype("Arial Bold", 50)
                    #except:
                        #title_font = ImageFont.load_default()
                        #subtitle_font = ImageFont.load_default()
                        #name_font = ImageFont.load_default()
                        #score_font = ImageFont.load_default()
                        #total_font = ImageFont.load_default()
                        #medal_font = ImageFont.load_default()
                    
                    # Logo y título
                    #try:
                        #logo = Image.open('logo.png')
                        #logo = logo.resize((100, 100), Image.Resampling.LANCZOS)  # Logo más grande
                        
                        # Posicionar logo arriba del título, centrado
                        #logo_x = width//2 - 50  # Centrar el logo (100px de ancho / 2)
                        #logo_y = 60  # Posición más arriba
                        
                        # Pegar logo
                        #if logo.mode == 'RGBA':
                            #img.paste(logo, (logo_x, logo_y), logo)
                        #else:
                            #img.paste(logo, (logo_x, logo_y))
                        
                        # Título centrado debajo del logo
                        #draw.text((width//2, 200), "IMOX CLUB", fill='#1a1a1a', font=title_font, anchor='mm')
                        
                    #except FileNotFoundError:
                        # Si no encuentra el logo, usar solo texto
                        #draw.text((width//2, 150), "IMOX CLUB", fill='#1a1a1a', font=title_font, anchor='mm')
                    
                    # Mes del ranking
                    #target_date = datetime.now() + timedelta(days=30 * st.session_state.selected_month)
                    #month_name = target_date.strftime('%B %Y')
                    
                    # Traducir meses al español
                    #month_translations = {
                        #'January': 'Enero', 'February': 'Febrero', 'March': 'Marzo',
                        #'April': 'Abril', 'May': 'Mayo', 'June': 'Junio',
                        #'July': 'Julio', 'August': 'Agosto', 'September': 'Septiembre',
                        #'October': 'Octubre', 'November': 'Noviembre', 'December': 'Diciembre'
                    #}
                    
                    #month_spanish = month_translations.get(target_date.strftime('%B'), target_date.strftime('%B'))
                    #month_text = f"{month_spanish} {target_date.year}"
                    
                    #draw.text((width//2, 270), month_text, fill='black', font=subtitle_font, anchor='mm')
                    
                    # Ranking con símbolos que PIL puede renderizar
                    #medals = {
                        #1: '1°', 2: '2°', 3: '3°', 4: '4°', 5: '5°', 
                        #6: '6°', 7: '7°', 8: '8°', 9: '9°', 10: '10°'
                    #}
                    
                    #for i, (position, name, score) in enumerate(ranking_data[:10]):
                        #y_pos = 400 + i * 140
                        
                        # Fondo redondeado para cada fila
                        #draw.rounded_rectangle([50, y_pos-50, width-50, y_pos+50], radius=15, fill='#f8f9fa', outline='#dee2e6', width=2)
                        
                        # Medalla con símbolos simples en negro
                        #medal = medals.get(position, f"{position}°")
                        #draw.text((100, y_pos), medal, fill='#000000', font=medal_font, anchor='mm')
                        
                        # Nombre
                        #draw.text((200, y_pos), name, fill='#1a1a1a', font=name_font, anchor='lm')
                        
                        # Score formateado con $ al final y sin centavos, usando punto como separador de miles
                        #score_value = float(score.replace('$', '').replace(',', ''))
                        #score_formatted = f"{score_value:,.0f}$".replace(',', '.')
                        #draw.text((width-100, y_pos), score_formatted, fill='#28a745', font=score_font, anchor='rm')
                    
                    # Total generado al final con $ al final y sin centavos, usando punto como separador de miles
                    #if not global_ranking.empty:
                        #total_amount = sum([float(row['total_payout']) for _, row in global_ranking.head(10).iterrows()])
                        #total_text = f"Total: {total_amount:,.0f}$".replace(',', '.')
                        #draw.text((width//2, height-150), total_text, fill='#1a1a1a', font=total_font, anchor='mm')
                    
                    # Marca de agua
                    #draw.text((width//2, height-50), "@imoxtrading", fill='#999999', font=subtitle_font, anchor='mm')
                    
                    #return img
                
                # Generar imagen
                #img = generate_ranking_image(ranking_data)
                #buffer = io.BytesIO()
                #img.save(buffer, format='PNG')
                #buffer.seek(0)
                
                # UN SOLO BOTÓN que descarga directamente
                #try:
                    # Cargar el icono personalizado
                    #icon = Image.open('downloadIcon.png')
                    #icon = icon.resize((32, 32), Image.Resampling.LANCZOS)
                    
                    # Convertir a base64 para usar en el botón
                    #import base64
                    #buffer_icon = io.BytesIO()
                    #icon.save(buffer_icon, format='PNG')
                    #buffer_icon.seek(0)
                    #icon_base64 = base64.b64encode(buffer_icon.getvalue()).decode()
                    
                    # Usar st.download_button con el icono como texto
                    #st.download_button(
                        #f"![Icon](data:image/png;base64,{icon_base64})",
                        #data=buffer.getvalue(),
                        #file_name=f"ranking_global_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        #mime="image/png",
                        #key="share_global",
                        #help="Compartir ranking en Instagram",
                        #use_container_width=True
                    #)
                    
                #except FileNotFoundError:
                    # Si no encuentra el icono, usar el botón normal
                    #st.download_button(
                        #"📥",
                        #data=buffer.getvalue(),
                        #file_name=f"ranking_global_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        #mime="image/png",
                        #key="share_global",
                        #help="Compartir ranking en Instagram",
                        #use_container_width=True
                    #)
            
            #if not global_ranking.empty:
                # Crear expanders para cada usuario en el ranking
                #for _, row in global_ranking.iterrows():
                    #position = row['position']
                    #name = row['name']
                    #total_payout = row['total_payout_formatted']
                    #num_payouts = row['num_payouts']
                    
                    # Medalla según posición
                    #medal = ('🥇' if position == 1 else 
                             #'🥈' if position == 2 else 
                             #'🥉' if position == 3 else 
                             #'➃' if position == 4 else
                             #'➄' if position == 5 else
                             #'➅' if position == 6 else
                             #'➆' if position == 7 else
                             #'➇' if position == 8 else
                             #'➈' if position == 9 else
                             #'➉' if position == 10 else
                             #f"{position}.")
                    
                    # Obtener payouts detallados del usuario
                    #user_payouts = [p for p in payouts_data if p['nick'] == row['nick'] and p['fecha_payout']]
                    
                    # Filtrar por mes seleccionado
                    #target_date = datetime.now() + timedelta(days=30 * st.session_state.selected_month)
                    #target_year = target_date.year
                    #target_month = target_date.month
                    
                    #monthly_payouts = []
                    #for payout in user_payouts:
                        #payout_date = datetime.fromisoformat(payout['fecha_payout'].replace('Z', '+00:00'))
                        #if payout_date.year == target_year and payout_date.month == target_month:
                            #monthly_payouts.append(payout)
                    
                    # Crear expander para el usuario
                    #with st.expander(f"{medal} {name} - {total_payout} ({num_payouts} payouts)", expanded=False):
                        #if monthly_payouts:
                            # Crear DataFrame con los payouts detallados
                            #payouts_df = pd.DataFrame(monthly_payouts)
                            #payouts_df['fecha_payout'] = pd.to_datetime(payouts_df['fecha_payout'])
                            #payouts_df = payouts_df.sort_values('fecha_payout')
                            
                            # Seleccionar y renombrar columnas
                            #display_df = payouts_df[['fecha_payout', 'payout', 'herramienta']].copy()
                            #display_df.columns = ['Fecha', 'Monto', 'Herramienta']
                            #display_df['Fecha'] = display_df['Fecha'].dt.strftime('%d/%m/%Y')
                            #display_df['Monto'] = display_df['Monto'].apply(lambda x: f"${float(x):,.2f}")
                            
                            #st.dataframe(display_df, use_container_width=True, hide_index=True)
                        #else:
                            #st.info("No hay payouts detallados para este mes")
                
                # Mostrar estadísticas adicionales
                #total_payouts = global_ranking['total_payout'].sum()
                
                #st.markdown(f"""
                #<div class="ranking-total" style="background: white; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    #<div style="display: flex; justify-content: space-between;">
                        #<span style="color: #6c757d;">Total:</span>
                        #<span style="font-weight: bold; color: #28a745;">${total_payouts:,.2f}</span>
                    #</div>
                #</div>
                #""", unsafe_allow_html=True)
            #else:
                #st.markdown("""
                #<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem; text-align: center;">
                    #<p style="color: #6c757d; font-size: 0.9rem; margin: 0;">
                        #No hay payouts para este mes
                    #</p>
                #</div>
                #""", unsafe_allow_html=True)
            
            #st.markdown('</div></div>', unsafe_allow_html=True)
        
        #with col2:
            # Crear columnas para título y botón
            #title_col, share_col = st.columns([4, 1])
            
            #with title_col:
                #st.markdown("""
                #<div class="card ranking-panel">
                    #<div class="card-title">🥇 Ranking Gold</div>
                #</div>
                #""", unsafe_allow_html=True)
            
            #with share_col:
                # Preparar datos del ranking Gold
                #gold_ranking_data = []
                #if not gold_ranking.empty:
                    #for _, row in gold_ranking.head(10).iterrows():
                        #gold_ranking_data.append((
                            #row['position'],
                            #row['name'],
                            #row['total_payout_formatted']
                        #))
                
                # Función para generar imagen Gold
                #def generate_gold_ranking_image(ranking_data):
                    #from PIL import Image, ImageDraw, ImageFont
                    #import io
                    
                    #width, height = 1080, 1920
                    #img = Image.new('RGB', (width, height), color='#ffffff')
                    #draw = ImageDraw.Draw(img)
                    
                    #try:
                        #title_font = ImageFont.truetype("Arial Bold", 80)
                        #subtitle_font = ImageFont.truetype("Arial", 50)
                        #name_font = ImageFont.truetype("Arial", 45)
                        #score_font = ImageFont.truetype("Arial Bold", 40)
                        #total_font = ImageFont.truetype("Arial Bold", 60)
                        #medal_font = ImageFont.truetype("Arial Bold", 50)
                    #except:
                        #title_font = ImageFont.load_default()
                        #subtitle_font = ImageFont.load_default()
                        #name_font = ImageFont.load_default()
                        #score_font = ImageFont.load_default()
                        #total_font = ImageFont.load_default()
                        #medal_font = ImageFont.load_default()
                    
                    # Logo y título
                    #try:
                        #logo = Image.open('logo.png')
                        #logo = logo.resize((100, 100), Image.Resampling.LANCZOS)
                        
                        #logo_x = width//2 - 50
                        #logo_y = 60
                        
                        #if logo.mode == 'RGBA':
                            #img.paste(logo, (logo_x, logo_y), logo)
                        #else:
                            #img.paste(logo, (logo_x, logo_y))
                        
                        #draw.text((width//2, 200), "IMOX CLUB", fill='#1a1a1a', font=title_font, anchor='mm')
                        
                    #except FileNotFoundError:
                        #draw.text((width//2, 150), "IMOX CLUB", fill='#1a1a1a', font=title_font, anchor='mm')
                    
                    # Mes del ranking
                    #target_date = datetime.now() + timedelta(days=30 * st.session_state.selected_month)
                    #month_translations = {
                        #'January': 'Enero', 'February': 'Febrero', 'March': 'Marzo',
                        #'April': 'Abril', 'May': 'Mayo', 'June': 'Junio',
                        #'July': 'Julio', 'August': 'Agosto', 'September': 'Septiembre',
                        #'October': 'Octubre', 'November': 'Noviembre', 'December': 'Diciembre'
                    #}
                    
                    #month_spanish = month_translations.get(target_date.strftime('%B'), target_date.strftime('%B'))
                    #month_text = f"{month_spanish} {target_date.year} - GOLD"
                    
                    #draw.text((width//2, 270), month_text, fill='#ffc107', font=subtitle_font, anchor='mm')
                    
                    # Ranking Gold
                    #medals = {1: '1°', 2: '2°', 3: '3°', 4: '4°', 5: '5°', 6: '6°', 7: '7°', 8: '8°', 9: '9°', 10: '10°'}
                    
                    #for i, (position, name, score) in enumerate(ranking_data[:10]):
                        #y_pos = 400 + i * 140
                        
                        #draw.rectangle([50, y_pos-50, width-50, y_pos+50], fill='#fff3cd', outline='#ffc107', width=2)
                        
                        #medal = medals.get(position, f"{position}°")
                        #draw.text((100, y_pos), medal, fill='#ffc107', font=medal_font, anchor='mm')
                        
                        #draw.text((200, y_pos), name, fill='#1a1a1a', font=name_font, anchor='lm')
                        #draw.text((width-100, y_pos), score, fill='#28a745', font=score_font, anchor='rm')
                        #draw.line([(200, y_pos+30), (width-150, y_pos+30)], fill='#ffc107', width=1)
                    
                    # Total Gold
                    #if not gold_ranking.empty:
                        #total_amount = sum([float(row['total_payout']) for _, row in gold_ranking.head(10).iterrows()])
                        #total_text = f"Total Gold: ${total_amount:,.2f}"
                        #draw.text((width//2, height-200), total_text, fill='#1a1a1a', font=total_font, anchor='mm')
                    
                    #draw.text((width//2, height-100), "@imoxhub", fill='#999999', font=subtitle_font, anchor='mm')
                    
                    #return img
                
                # Generar imagen Gold
                #gold_img = generate_gold_ranking_image(gold_ranking_data)
                #gold_buffer = io.BytesIO()
                #gold_img.save(gold_buffer, format='PNG')
                #gold_buffer.seek(0)
                
                # Botón de descarga Gold
                #try:
                    #icon = Image.open('downloadIcon.png')
                    #icon = icon.resize((32, 32), Image.Resampling.LANCZOS)
                    
                    #import base64
                    #buffer_icon = io.BytesIO()
                    #icon.save(buffer_icon, format='PNG')
                    #buffer_icon.seek(0)
                    #icon_base64 = base64.b64encode(buffer_icon.getvalue()).decode()
                    
                    #st.download_button(
                        #f"![Icon](data:image/png;base64,{icon_base64})",
                        #data=gold_buffer.getvalue(),
                        #file_name=f"ranking_gold_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        #mime="image/png",
                        #key="share_gold",
                        #help="Compartir ranking Gold en Instagram",
                        #use_container_width=True
                    #)
                    
                #except FileNotFoundError:
                    #st.download_button(
                        #"📥",
                        #data=gold_buffer.getvalue(),
                        #file_name=f"ranking_gold_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        #mime="image/png",
                        #key="share_gold",
                        #help="Compartir ranking Gold en Instagram",
                        #use_container_width=True
                    #)
            
            #if not gold_ranking.empty:
                # Crear expanders para cada usuario Gold
                #for _, row in gold_ranking.iterrows():
                    #position = row['position']
                    #name = row['name']
                    #total_payout = row['total_payout_formatted']
                    #num_payouts = row['num_payouts']
                    
                    # Medalla según posición
                    #medal = ('🥇' if position == 1 else 
                             #'🥈' if position == 2 else 
                             #'🥉' if position == 3 else 
                             #'❶' if position == 4 else
                             #'❷' if position == 5 else
                             #'❸' if position == 6 else
                             #'❹' if position == 7 else
                             #'❺' if position == 8 else
                             #'❻' if position == 9 else
                             #'❼' if position == 10 else
                             #f"{position}.")
                    
                    # Obtener payouts detallados del usuario
                    #user_payouts = [p for p in payouts_data if p['nick'] == row['nick'] and p['fecha_payout']]
                    
                    # Filtrar por mes seleccionado
                    #target_date = datetime.now() + timedelta(days=30 * st.session_state.selected_month)
                    #target_year = target_date.year
                    #target_month = target_date.month
                    
                    #monthly_payouts = []
                    #for payout in user_payouts:
                        #payout_date = datetime.fromisoformat(payout['fecha_payout'].replace('Z', '+00:00'))
                        #if payout_date.year == target_year and payout_date.month == target_month:
                            #monthly_payouts.append(payout)
                    
                    # Crear expander para el usuario
                    #with st.expander(f"{medal} {name} - {total_payout} ({num_payouts} payouts)", expanded=False):
                        #if monthly_payouts:
                            # Crear DataFrame con los payouts detallados
                            #payouts_df = pd.DataFrame(monthly_payouts)
                            #payouts_df['fecha_payout'] = pd.to_datetime(payouts_df['fecha_payout'])
                            #payouts_df = payouts_df.sort_values('fecha_payout')
                            
                            # Seleccionar y renombrar columnas
                            #display_df = payouts_df[['fecha_payout', 'payout', 'herramienta']].copy()
                            #display_df.columns = ['Fecha', 'Monto', 'Herramienta']
                            #display_df['Fecha'] = display_df['Fecha'].dt.strftime('%d/%m/%Y')
                            #display_df['Monto'] = display_df['Monto'].apply(lambda x: f"${float(x):,.2f}")
                            
                            #st.dataframe(display_df, use_container_width=True, hide_index=True)
                        #else:
                            #st.info("No hay payouts detallados para este mes")
                
                # Mostrar estadísticas Gold
                #total_gold = gold_ranking['total_payout'].sum()
                
                #st.html(f"""
                #<div class="ranking-total" style="background: #fff3cd; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 7px solid #ffc107;">
                    #<div style="display: flex; justify-content: space-between;">
                        #<span style="color: #856404;">Total Gold:</span>
                        #<span style="font-weight: bold; color: #856404;">${total_gold:,.2f}</span>
                    #</div>
                #</div>
                #""")
            #else:
                #st.markdown("""
                #<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem; text-align: center;">
                    #<p style="color: #6c757d; font-size: 0.9rem; margin: 0;">
                        #No hay usuarios Gold con payouts este mes
                    #</p>
                #</div>
                #""", unsafe_allow_html=True)
            
            #st.markdown('</div></div>', unsafe_allow_html=True)
        
        #with col3:
            # Crear columnas para título y botón
            #title_col, share_col = st.columns([4, 1])
            
            #with title_col:
                #st.markdown("""
                #<div class="card ranking-panel">
                    #<div class="card-title">🥈 Ranking Silver</div>
                #</div>
                #""", unsafe_allow_html=True)
            
            #with share_col:
                # Preparar datos del ranking Silver
                #silver_ranking_data = []
                #if not silver_ranking.empty:
                    #for _, row in silver_ranking.head(10).iterrows():
                        #silver_ranking_data.append((
                            #row['position'],
                            #row['name'],
                            #row['total_payout_formatted']
                        #))
                
                # Función para generar imagen Silver
                #def generate_silver_ranking_image(ranking_data):
                    #from PIL import Image, ImageDraw, ImageFont
                    #import io
                    
                    #width, height = 1080, 1920
                    #img = Image.new('RGB', (width, height), color='#ffffff')
                    #draw = ImageDraw.Draw(img)
                    
                    #try:
                        #title_font = ImageFont.truetype("Arial Bold", 80)
                        #subtitle_font = ImageFont.truetype("Arial", 50)
                        #name_font = ImageFont.truetype("Arial", 45)
                        #score_font = ImageFont.truetype("Arial Bold", 40)
                        #total_font = ImageFont.truetype("Arial Bold", 60)
                        #medal_font = ImageFont.truetype("Arial Bold", 50)
                    #except:
                        #title_font = ImageFont.load_default()
                        #subtitle_font = ImageFont.load_default()
                        #name_font = ImageFont.load_default()
                        #score_font = ImageFont.load_default()
                        #total_font = ImageFont.load_default()
                        #medal_font = ImageFont.load_default()
                    
                    # Logo y título
                    #try:
                        #logo = Image.open('logo.png')
                        #logo = logo.resize((100, 100), Image.Resampling.LANCZOS)
                        
                        #logo_x = width//2 - 50
                        #logo_y = 60
                        
                        #if logo.mode == 'RGBA':
                            #img.paste(logo, (logo_x, logo_y), logo)
                        #else:
                            #img.paste(logo, (logo_x, logo_y))
                        
                        #draw.text((width//2, 200), "IMOX CLUB", fill='#1a1a1a', font=title_font, anchor='mm')
                        
                    #except FileNotFoundError:
                        #draw.text((width//2, 150), "IMOX CLUB", fill='#1a1a1a', font=title_font, anchor='mm')
                    
                    # Mes del ranking
                    #target_date = datetime.now() + timedelta(days=30 * st.session_state.selected_month)
                    #month_translations = {
                        #'January': 'Enero', 'February': 'Febrero', 'March': 'Marzo',
                        #'April': 'Abril', 'May': 'Mayo', 'June': 'Junio',
                        #'July': 'Julio', 'August': 'Agosto', 'September': 'Septiembre',
                        #'October': 'Octubre', 'November': 'Noviembre', 'December': 'Diciembre'
                    #}
                    
                    #month_spanish = month_translations.get(target_date.strftime('%B'), target_date.strftime('%B'))
                    #month_text = f"{month_spanish} {target_date.year} - SILVER"
                    
                    #draw.text((width//2, 270), month_text, fill='#6c757d', font=subtitle_font, anchor='mm')
                    
                    # Ranking Silver
                    #medals = {1: '1°', 2: '2°', 3: '3°', 4: '4°', 5: '5°', 6: '6°', 7: '7°', 8: '8°', 9: '9°', 10: '10°'}
                    
                    #for i, (position, name, score) in enumerate(ranking_data[:10]):
                        #y_pos = 400 + i * 140
                        
                        #draw.rectangle([50, y_pos-50, width-50, y_pos+50], fill='#f8f9fa', outline='#6c757d', width=2)
                        
                        #medal = medals.get(position, f"{position}°")
                        #draw.text((100, y_pos), medal, fill='#6c757d', font=medal_font, anchor='mm')
                        
                        #draw.text((200, y_pos), name, fill='#1a1a1a', font=name_font, anchor='lm')
                        #draw.text((width-100, y_pos), score, fill='#28a745', font=score_font, anchor='rm')
                        #draw.line([(200, y_pos+30), (width-150, y_pos+30)], fill='#6c757d', width=1)
                    
                    # Total Silver
                    #if not silver_ranking.empty:
                        #total_amount = sum([float(row['total_payout']) for _, row in silver_ranking.head(10).iterrows()])
                        #total_text = f"Total Silver: ${total_amount:,.2f}"
                        #draw.text((width//2, height-200), total_text, fill='#1a1a1a', font=total_font, anchor='mm')
                    
                    #draw.text((width//2, height-100), "@imoxhub", fill='#999999', font=subtitle_font, anchor='mm')
                    
                    #return img
                
                # Generar imagen Silver
                #silver_img = generate_silver_ranking_image(silver_ranking_data)
                #silver_buffer = io.BytesIO()
                #silver_img.save(silver_buffer, format='PNG')
                #silver_buffer.seek(0)
                
                # Botón de descarga Silver
                #try:
                    #icon = Image.open('downloadIcon.png')
                    #icon = icon.resize((32, 32), Image.Resampling.LANCZOS)
                    
                    #import base64
                    #buffer_icon = io.BytesIO()
                    #icon.save(buffer_icon, format='PNG')
                    #buffer_icon.seek(0)
                    #icon_base64 = base64.b64encode(buffer_icon.getvalue()).decode()
                    
                    #st.download_button(
                        #f"![Icon](data:image/png;base64,{icon_base64})",
                        #data=silver_buffer.getvalue(),
                        #file_name=f"ranking_silver_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        #mime="image/png",
                        #key="share_silver",
                        #help="Compartir ranking Silver en Instagram",
                        #use_container_width=True
                    #)
                    
                #except FileNotFoundError:
                    #st.download_button(
                        #"📥",
                        #data=silver_buffer.getvalue(),
                        #file_name=f"ranking_silver_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        #mime="image/png",
                        #key="share_silver",
                        #help="Compartir ranking Silver en Instagram",
                        #use_container_width=True
                    #)
            
            #if not silver_ranking.empty:
                # Crear expanders para cada usuario Silver
                #for _, row in silver_ranking.iterrows():
                    #position = row['position']
                    #name = row['name']
                    #total_payout = row['total_payout_formatted']
                    #num_payouts = row['num_payouts']
                    
                    # Medalla según posición
                    #medal = ('🥇' if position == 1 else 
                             #'🥈' if position == 2 else 
                             #'🥉' if position == 3 else 
                             #'❶' if position == 4 else
                             #'❷' if position == 5 else
                             #'❸' if position == 6 else
                             #'❹' if position == 7 else
                             #'❺' if position == 8 else
                             #'❻' if position == 9 else
                             #'❼' if position == 10 else
                             #f"{position}.")
                    
                    # Obtener payouts detallados del usuario
                    #user_payouts = [p for p in payouts_data if p['nick'] == row['nick'] and p['fecha_payout']]
                    
                    # Filtrar por mes seleccionado
                    #target_date = datetime.now() + timedelta(days=30 * st.session_state.selected_month)
                    #target_year = target_date.year
                    #target_month = target_date.month
                    
                    #monthly_payouts = []
                    #for payout in user_payouts:
                        #payout_date = datetime.fromisoformat(payout['fecha_payout'].replace('Z', '+00:00'))
                        #if payout_date.year == target_year and payout_date.month == target_month:
                            #monthly_payouts.append(payout)
                    
                    # Crear expander para el usuario
                    #with st.expander(f"{medal} {name} - {total_payout} ({num_payouts} payouts)", expanded=False):
                        #if monthly_payouts:
                            # Crear DataFrame con los payouts detallados
                            #payouts_df = pd.DataFrame(monthly_payouts)
                            #payouts_df['fecha_payout'] = pd.to_datetime(payouts_df['fecha_payout'])
                            #payouts_df = payouts_df.sort_values('fecha_payout')
                            
                            # Seleccionar y renombrar columnas
                            #display_df = payouts_df[['fecha_payout', 'payout', 'herramienta']].copy()
                            #display_df.columns = ['Fecha', 'Monto', 'Herramienta']
                            #display_df['Fecha'] = display_df['Fecha'].dt.strftime('%d/%m/%Y')
                            #display_df['Monto'] = display_df['Monto'].apply(lambda x: f"${float(x):,.2f}")
                            
                            #st.dataframe(display_df, use_container_width=True, hide_index=True)
                        #else:
                            #st.info("No hay payouts detallados para este mes")
                
                # Mostrar estadísticas Silver
                #total_silver = silver_ranking['total_payout'].sum()
                
                #st.markdown(f"""
                #<div class="ranking-total" style="background: white; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 7px solid #6c757d;">
                    #<div style="display: flex; justify-content: space-between;">
                        #<span style="color: #6c757d;">Total Silver:</span>
                        #<span style="font-weight: bold; color: #6c757d;">${total_silver:,.2f}</span>
                    #</div>
                #</div>
                #""", unsafe_allow_html=True)
            #else:
                #st.markdown("""
                #<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem; text-align: center;">
                    #<p style="color: #6c757d; font-size: 0.9rem; margin: 0;">
                        #No hay usuarios Silver con payouts este mes
                    #</p>
                #</div>
                #""", unsafe_allow_html=True)
            
            #st.markdown('</div></div>', unsafe_allow_html=True)
    
    # Gráfico de evolución mensual por categoría (después de todos los rankings)
    #st.markdown("""
    #<div class="card" style="margin-top: 2rem;">
        #<div class="card-title">Evolución Mensual por Categoría</div>
    #""", unsafe_allow_html=True)
    
    # Preparar datos para el gráfico mensual de payouts
    #if not payouts_data:
        #st.info("No hay datos de payouts para mostrar")
        #st.markdown('</div>', unsafe_allow_html=True)
    #else:
        #df_payouts = pd.DataFrame(payouts_data)
        
        # Verificar que la columna fecha_payout existe antes de usarla
        #if 'fecha_payout' not in df_payouts.columns:
            #st.info("No hay datos de fechas de payout disponibles")
            #st.markdown('</div>', unsafe_allow_html=True)
        #else:
            #df_payouts['fecha_payout'] = pd.to_datetime(df_payouts['fecha_payout'], errors='coerce')
    
    # Filtrar solo payouts válidos
    #df_payouts_valid = df_payouts.dropna(subset=['fecha_payout']).copy()
    
    #if not df_payouts_valid.empty:
        # Convertir payout a float
        #df_payouts_valid['payout_amount'] = pd.to_numeric(df_payouts_valid['payout'], errors='coerce')
        #df_payouts_valid = df_payouts_valid.dropna(subset=['payout_amount'])
        
        # Crear diccionario de usuarios para obtener rangos
        #users_dict = {user['nick']: {'name': user['name'], 'rank': user['rank']} for user in users_data}
        
        # Agregar información de usuario
        #df_payouts_valid['rank'] = df_payouts_valid['nick'].map(lambda x: users_dict.get(x, {}).get('rank', 'unknown'))
        
        # Agrupar por mes y categoría
        #df_payouts_valid['year_month'] = df_payouts_valid['fecha_payout'].dt.to_period('M')
        
        # Calcular totales mensuales por categoría
        #datos_mensuales = []
        
        #for mes in df_payouts_valid['year_month'].unique():
            #df_mes = df_payouts_valid[df_payouts_valid['year_month'] == mes]
            
            # Global (todos los datos)
            #total_global = df_mes['payout_amount'].sum()
            
            # Gold (usuarios con rank 'gold')
            #df_gold = df_mes[df_mes['rank'] == 'gold']
            #total_gold = df_gold['payout_amount'].sum()
            
            # Silver (usuarios con rank 'silver')
            #df_silver = df_mes[df_mes['rank'] == 'silver']
            #total_silver = df_silver['payout_amount'].sum()
            
            #datos_mensuales.append({
                #'Mes': mes,
                #'Global': total_global,
                #'Gold': total_gold,
                #'Silver': total_silver
            #})
        
        # Crear DataFrame para el gráfico
        #df_mensual = pd.DataFrame(datos_mensuales)
        #df_mensual = df_mensual.sort_values('Mes')
        #df_mensual['Mes_Str'] = df_mensual['Mes'].astype(str)
        
        # Crear gráfico de líneas
        #import plotly.graph_objects as go
        
        #fig_mensual = go.Figure()
        
        # Línea Global
        #fig_mensual.add_trace(go.Scatter(
            #x=df_mensual['Mes_Str'],
            #y=df_mensual['Global'],
            #mode='lines+markers',
            #name='Global',
            #line=dict(color='#1f77b4', width=3),
            #marker=dict(size=8),
            #hovertemplate='<b>Global</b><br>Mes: %{x}<br>Total: $%{y:,.2f}<extra></extra>'
        #))
        
        # Línea Gold
        #fig_mensual.add_trace(go.Scatter(
            #x=df_mensual['Mes_Str'],
            #y=df_mensual['Gold'],
            #mode='lines+markers',
            #name='Gold',
            #line=dict(color='#ffd700', width=3),
            #marker=dict(size=8),
            #hovertemplate='<b>Gold</b><br>Mes: %{x}<br>Total: $%{y:,.2f}<extra></extra>'
        #))
        
        # Línea Silver
        #fig_mensual.add_trace(go.Scatter(
            #x=df_mensual['Mes_Str'],
            #y=df_mensual['Silver'],
            #mode='lines+markers',
            #name='Silver',
            #line=dict(color='#c0c0c0', width=3),
            #marker=dict(size=8),
            #hovertemplate='<b>Silver</b><br>Mes: %{x}<br>Total: $%{y:,.2f}<extra></extra>'
        #))
        
        # Configurar layout
        #fig_mensual.update_layout(
            #title="Evolución Mensual de Payouts por Categoría",
            #xaxis_title="Mes",
            #yaxis_title="Payout Total ($)",
            #template="plotly_white",
            #height=500,
            #margin=dict(l=20, r=20, t=40, b=20),
            #plot_bgcolor='white',
            #paper_bgcolor='white',
            #font=dict(color='#495057'),
            #xaxis=dict(
                #gridcolor='#e9ecef',
                #linecolor='#e9ecef',
                #tickcolor='#495057',
                #tickfont=dict(color='#495057')
            #),
            #yaxis=dict(
                #gridcolor='#e9ecef',
                #linecolor='#e9ecef',
                #tickcolor='#495057',
                #tickfont=dict(color='#495057'),
                #tickformat='$,.0f'
            #),
            #legend=dict(
                #orientation="h",
                #yanchor="bottom",
                #y=1.02,
                #xanchor="right",
                #x=1
            #)
        #)
        
        #st.plotly_chart(fig_mensual, use_container_width=True)
        
        # Mostrar estadísticas del gráfico
        #col1, col2, col3 = st.columns(3)
        
        #with col1:
            #total_global_periodo = df_mensual['Global'].sum()
            #st.metric(
                #"Total Global",
                #f"${total_global_periodo:,.2f}",
                #help="Suma total de todos los payouts en el período"
            #)
        
        #with col2:
            #total_gold_periodo = df_mensual['Gold'].sum()
            #porcentaje_gold = (total_gold_periodo / total_global_periodo * 100) if total_global_periodo > 0 else 0
            #st.metric(
                #"Total Gold",
                #f"${total_gold_periodo:,.2f}",
                #f"{porcentaje_gold:.1f}% del total",
                #help="Suma de usuarios Gold"
            #)
        
        #with col3:
            #total_silver_periodo = df_mensual['Silver'].sum()
            #porcentaje_silver = (total_silver_periodo / total_global_periodo * 100) if total_global_periodo > 0 else 0
            #st.metric(
                #"Total Silver",
                #f"${total_silver_periodo:,.2f}",
                #f"{porcentaje_silver:.1f}% del total",
                #help="Suma de usuarios Silver"
            #)
    #else:
        #st.info("No hay datos de payouts suficientes para generar el gráfico")
    
    #st.markdown('</div>', unsafe_allow_html=True)
    
    #with ranking_sub_tab2:
        #st.markdown("""
        #<div class="card">
            #<div class="card-title">Añadir Cobro</div>
            #<div style="text-align: center; padding: 2rem;">
                #<div style="font-size: 4rem; margin-bottom: 1rem;" class="construction-icon">💰</div>
                #<h2 style="color: #6c757d; margin-bottom: 1rem;">¡En Desarrollo!</h2>
                #<p style="color: #6c757d; font-size: 1.1rem; margin-bottom: 2rem;">
                    #Próximamente disponible.
                #</p>
            #</div>
        #</div>
        #""", unsafe_allow_html=True)

# Tab de Administración (solo para admin)
if is_admin:
    with tab4:
        st.markdown("""
        <div class="card">
                <div class="card-title">Administración de Usuarios</div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Crear Nuevo Usuario")
        
        with st.form("create_user_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                nick = st.text_input("Nick de Usuario *", placeholder="ejemplo: usuario123", help="Nombre de usuario único")
                name = st.text_input("Nombre Completo *", placeholder="ejemplo: Juan Pérez", help="Nombre completo del usuario")
            
            with col2:
                correo = st.text_input("Correo Electrónico *", placeholder="ejemplo: usuario@email.com", help="Correo electrónico válido")
                rank = st.selectbox(
                    "Rango *",
                    options=["silver", "gold", "admin"],
                    index=0,
                    help="Rango del usuario: Silver (por defecto), Gold o Admin"
                )
            
            st.markdown("<small>* Campos obligatorios</small>", unsafe_allow_html=True)
            
            submitted = st.form_submit_button("Crear Usuario", use_container_width=True)
            
            if submitted:
                if not nick or not name or not correo:
                    st.error("Por favor completa todos los campos obligatorios")
                else:
                    success, message = create_user_api(nick, name, correo, rank)
                    if success:
                        st.success(message)
                        # Invalidar caché para refrescar los datos
                        get_users_from_api.clear()
                        # Limpiar el formulario recargando la página
                        st.rerun()
                    else:
                        st.error(f"Error al crear usuario: {message}")
        
        st.markdown("---")
        st.markdown("### Resetear Contraseña de un Usuario Específico")
        
        with st.form("reset_password_form"):
            st.markdown("**Resetear contraseña de un usuario**")
            st.caption("💡 Al resetear la contraseña, el usuario deberá establecer una nueva en su próximo login")
            
            reset_nick = st.text_input("Nick del Usuario", placeholder="ejemplo: usuario123", help="Nick del usuario cuya contraseña quieres resetear")
            
            reset_submitted = st.form_submit_button("Resetear Contraseña", use_container_width=True)
            
            if reset_submitted:
                if not reset_nick or not reset_nick.strip():
                    st.error("Por favor ingresa el nick del usuario")
                else:
                    with st.spinner("Reseteando contraseña..."):
                        success, message = reset_password_api(reset_nick.strip(), st.session_state.token)
                    
                    if success:
                        st.success(message)
                        # Invalidar caché de usuarios
                        get_users_from_api.clear()
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
        
        st.markdown("---")
        st.markdown("### Usuarios Existentes")
        
        # Mostrar lista de usuarios con indicador de carga
        with st.spinner("Cargando lista de usuarios..."):
            users_data = get_users_from_api()
        
        if users_data and isinstance(users_data, list) and len(users_data) > 0:
            try:
                df_users = pd.DataFrame(users_data)
                # Verificar que las columnas necesarias existan
                required_cols = ['nick', 'name', 'correo', 'rank']
                missing_cols = [col for col in required_cols if col not in df_users.columns]
                
                if missing_cols:
                    st.warning(f"⚠️ Faltan columnas en los datos: {', '.join(missing_cols)}")
                elif not df_users.empty:
                    df_users_display = df_users[required_cols].copy()
                    df_users_display.columns = ['Nick', 'Nombre', 'Correo', 'Rango']
                    st.dataframe(df_users_display, use_container_width=True, hide_index=True)
                else:
                    st.info("No hay usuarios registrados")
            except Exception as e:
                st.error(f"Error al procesar los datos de usuarios: {str(e)}")
        else:
            st.warning("⚠️ No se pudieron cargar los usuarios. Por favor, verifica la conexión con el backend.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Cerrar contenedor principal
st.markdown('</div>', unsafe_allow_html=True)