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

# Configurar tema claro por defecto para Plotly
pio.templates.default = "plotly_white"

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

def get_payouts_from_api():
    """
    Obtiene todos los payouts desde la API del backend
    """
    try:
        response = requests.get("http://localhost:8000/payouts/")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error al obtener payouts: {response.status_code}")
            return []
    except requests.exceptions.ConnectionError:
        st.error("No se puede conectar con el backend. Asegúrate de que esté ejecutándose en http://localhost:8000")
        return []
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")
        return []

def get_users_from_api():
    """
    Obtiene todos los usuarios desde la API del backend
    """
    try:
        response = requests.get("http://localhost:8000/users/")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error al obtener usuarios: {response.status_code}")
            return []
    except requests.exceptions.ConnectionError:
        st.error("No se puede conectar con el backend. Asegúrate de que esté ejecutándose en http://localhost:8000")
        return []
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")
        return []

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
    if not payouts_data:
        return pd.DataFrame()
    
    # Convertir a DataFrame
    df_payouts = pd.DataFrame(payouts_data)
    
    # Convertir fechas
    df_payouts['fecha_payout'] = pd.to_datetime(df_payouts['fecha_payout'], errors='coerce')
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
    users_dict = {user['nick']: {'name': user['name'], 'rank': user['rank']} for user in users_data}
    
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
    
    st.markdown("""
    <div class="card">
        <div class="card-title">📈 Evolución Detallada de Challenges</div>
    """, unsafe_allow_html=True)
    
    # Preparar datos
    df_processed = df_csv.copy()
    df_processed['Open time'] = pd.to_datetime(df_processed['Open time'])
    df_processed['Close time'] = pd.to_datetime(df_processed['Close time'])
    df_processed['profit_loss'] = pd.to_numeric(df_processed['Profit/Loss'])
    df_processed['date'] = df_processed['Close time'].dt.date
    df_processed = df_processed.sort_values('Close time').reset_index(drop=True)
    
    # Procesar tipo de cierre
    df_processed['close_type'] = df_processed['Close type'].fillna('Unknown')
    
    # Información sobre la funcionalidad
    st.info("Esta tabla muestra la evolución trade por trade de cada challenge, permitiendo ver exactamente cómo progresa el balance y en qué momento se aprueba o suspende cada examen.")
    
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
    
    # Mostrar todos los challenges sin paginación
    available_challenges = sorted(evolution_df['challenge_num'].unique())
    total_challenges = len(available_challenges)
    
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
            st.markdown("#### 📊 Gráfico de Evolución del Balance")
            
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
    st.markdown("#### 📋 Resumen de Todos los Challenges")
    
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
    
    /* Alinear las tarjetas de ranking horizontalmente */
    .ranking-panel {
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;
    }
    
    .ranking-panel .card-title {
        flex-shrink: 0 !important;
        margin-bottom: 0 !important;
        height: 60px !important;
        display: flex !important;
        align-items: center !important;
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
        st.markdown('<div class="login-form">', unsafe_allow_html=True)
        
        usuario = st.text_input("Usuario", placeholder="Ingresa tu usuario")
        contraseña = st.text_input("Contraseña", type="password", placeholder="Ingresa tu contraseña")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            login_button = st.form_submit_button("Entrar", use_container_width=True)
        
        if login_button:
            if usuario and contraseña:
                st.session_state.logged_in = True
                st.session_state.username = usuario
                st.rerun()
            else:
                st.error("Por favor ingresa usuario y contraseña")

# Verificar si el usuario está logueado
if not st.session_state.logged_in:
    show_login_page()
    st.stop()

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
        align-items: center !important;
        justify-content: center !important;
    }

    .st-emotion-cache-wfksaw:last-child {
        align-items: center !important;
        justify-content: normal !important;
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
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 0.5rem; margin-left: 0.5rem;">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="height: 30px; margin-right: 10px;">
        <h1>IMOXHUB</h1>
        <div style="flex-grow: 1;"></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    if st.button("Cerrar Sesión", key="logout"):
        st.session_state.logged_in = False
        st.rerun()

# Usar tabs de Streamlit para el contenido
tab1, tab2, tab3, tab4 = st.tabs(["📊 Análisis", "% Descuentos", "🏆 Ranking", "⚙️ Configuración"])

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
                        close_time = datetime.strptime(fila_op[8].text.strip(), "%Y.%m.%d %H:%M:%S")
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

                        datos.append({
                            "EA": ea_name,
                            "Símbolo": symbol,
                            "Tipo": tipo,
                            "Beneficio": profit,
                            "Open": open_time,
                            "Close": close_time,
                            "Duración": (close_time - open_time).total_seconds() / 60  # en minutos
                        })
                    except Exception as e:
                        # Log del error para debugging (opcional)
                        # print(f"Error procesando fila {i}: {e}")
                        continue

                if datos:
                    df = pd.DataFrame(datos)
                    
                    # Mostrar información sobre el filtrado
                    if eas_filtradas > 0:
                        st.info(f"ℹ️ Se eliminaron {eas_filtradas} operaciones que no pertenecen a ninguna EA de un total de {total_operaciones} operaciones. Se procesaron {len(datos)} operaciones válidas.")

                    # Crear gráfico de beneficio acumulado (incluyendo "Sin EA")
                    df_filtrado_grafico = df.copy()
                    df_filtrado_grafico['Fecha'] = df_filtrado_grafico['Close'].dt.date
                    beneficios_diarios = df_filtrado_grafico.groupby(['EA', 'Fecha'])['Beneficio'].sum().reset_index()
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
                
                    # Filtrar "Sin EA" para el resto del análisis
                    df = df[df['EA'] != 'Sin EA']
                    
                    if df.empty:
                        st.warning("No hay operaciones válidas para mostrar en las tablas.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
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
                        <div style="margin-top: -1rem; padding: 1rem; background-color: #f8f9fa; border-left: 7px solid #6c757d;">
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
            <div class="card-title">📈 Evolución Detallada de Challenges</div>
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
                st.markdown("### 📊 Configuración Completada")
                
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

# Tab de Ranking
with tab3:
    # Crear tabs anidados dentro del tab 3
    ranking_sub_tab1, ranking_sub_tab2 = st.tabs(["🏆 Ranking", "💰 Añadir Cobro"])
    
    with ranking_sub_tab1:
        # Selector de mes con navegación usando Streamlit
        if 'selected_month' not in st.session_state:
            st.session_state.selected_month = 0  # 0 = mes actual
        
        # Obtener fecha actual
        from datetime import datetime, timedelta
        current_date = datetime.now()
        target_date = current_date + timedelta(days=30 * st.session_state.selected_month)
        
        month_names = [
            "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
            "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
        ]
        
        # Crear layout para el selector de mes
        col_prev, col_month, col_next = st.columns([1, 3, 1])
        
        with col_prev:
            st.markdown("""
            <div style="display: flex; align-items: center; justify-content: flex-end; height: 100%; width: 100%;">
            """, unsafe_allow_html=True)
            if st.button("←", key="prev_month", help="Mes anterior"):
                st.session_state.selected_month -= 1
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_month:
            month_name = month_names[target_date.month - 1]
            year = target_date.year
            st.markdown(f"""
            <div style="
                background: #f8f9fa; 
                padding: 0.5rem 1.5rem; 
                border-radius: 25px; 
                border: 2px solid #dee2e6;
                font-size: 1.2rem;
                font-weight: 600;
                color: #495057;
                text-align: center;
                margin: 0.5rem 0;
            ">{month_name} {year}</div>
            """, unsafe_allow_html=True)
        
        with col_next:
            st.markdown("""
            <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
            """, unsafe_allow_html=True)
            if st.button("→", key="next_month", help="Mes siguiente"):
                st.session_state.selected_month += 1
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Obtener datos del backend
        payouts_data = get_payouts_from_api()
        users_data = get_users_from_api()
        
        # Crear rankings separados por rango
        global_ranking = create_monthly_payout_ranking(payouts_data, users_data, st.session_state.selected_month)
        gold_ranking = create_monthly_payout_ranking(payouts_data, users_data, st.session_state.selected_month, 'gold')
        silver_ranking = create_monthly_payout_ranking(payouts_data, users_data, st.session_state.selected_month, 'silver')
        
        # Crear tres columnas para los paneles de ranking (responsive)
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("""
            <div class="card ranking-panel">
                <div class="card-title">🌍 Ranking Global</div>
                <div class="ranking-content">
            """, unsafe_allow_html=True)
            
            if not global_ranking.empty:
                # Crear expanders para cada usuario en el ranking
                for _, row in global_ranking.iterrows():
                    position = row['position']
                    name = row['name']
                    total_payout = row['total_payout_formatted']
                    num_payouts = row['num_payouts']
                    
                    # Medalla según posición
                    medal = '🥇' if position == 1 else '🥈' if position == 2 else '🥉' if position == 3 else f"{position}."
                    
                    # Obtener payouts detallados del usuario
                    user_payouts = [p for p in payouts_data if p['nick'] == row['nick'] and p['fecha_payout']]
                    
                    # Filtrar por mes seleccionado
                    target_date = datetime.now() + timedelta(days=30 * st.session_state.selected_month)
                    target_year = target_date.year
                    target_month = target_date.month
                    
                    monthly_payouts = []
                    for payout in user_payouts:
                        payout_date = datetime.fromisoformat(payout['fecha_payout'].replace('Z', '+00:00'))
                        if payout_date.year == target_year and payout_date.month == target_month:
                            monthly_payouts.append(payout)
                    
                    # Crear expander para el usuario
                    with st.expander(f"{medal} {name} - {total_payout} ({num_payouts} payouts)", expanded=False):
                        if monthly_payouts:
                            # Crear DataFrame con los payouts detallados
                            payouts_df = pd.DataFrame(monthly_payouts)
                            payouts_df['fecha_payout'] = pd.to_datetime(payouts_df['fecha_payout'])
                            payouts_df = payouts_df.sort_values('fecha_payout')
                            
                            # Seleccionar y renombrar columnas
                            display_df = payouts_df[['fecha_payout', 'payout', 'herramienta']].copy()
                            display_df.columns = ['Fecha', 'Monto', 'Herramienta']
                            display_df['Fecha'] = display_df['Fecha'].dt.strftime('%d/%m/%Y')
                            display_df['Monto'] = display_df['Monto'].apply(lambda x: f"${float(x):,.2f}")
                            
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No hay payouts detallados para este mes")
                
                # Mostrar estadísticas adicionales
                total_payouts = global_ranking['total_payout'].sum()
                
                st.markdown(f"""
                <div class="ranking-total" style="background: white; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #6c757d;">Total:</span>
                        <span style="font-weight: bold; color: #28a745;">${total_payouts:,.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem; text-align: center;">
                    <p style="color: #6c757d; font-size: 0.9rem; margin: 0;">
                        No hay payouts para este mes
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card ranking-panel">
                <div class="card-title">🥇 Ranking Gold</div>
                <div class="ranking-content">
            """, unsafe_allow_html=True)
            
            if not gold_ranking.empty:
                # Crear expanders para cada usuario Gold
                for _, row in gold_ranking.iterrows():
                    position = row['position']
                    name = row['name']
                    total_payout = row['total_payout_formatted']
                    num_payouts = row['num_payouts']
                    
                    # Medalla según posición
                    medal = '🥇' if position == 1 else '🥈' if position == 2 else '🥉' if position == 3 else f"{position}."
                    
                    # Obtener payouts detallados del usuario
                    user_payouts = [p for p in payouts_data if p['nick'] == row['nick'] and p['fecha_payout']]
                    
                    # Filtrar por mes seleccionado
                    target_date = datetime.now() + timedelta(days=30 * st.session_state.selected_month)
                    target_year = target_date.year
                    target_month = target_date.month
                    
                    monthly_payouts = []
                    for payout in user_payouts:
                        payout_date = datetime.fromisoformat(payout['fecha_payout'].replace('Z', '+00:00'))
                        if payout_date.year == target_year and payout_date.month == target_month:
                            monthly_payouts.append(payout)
                    
                    # Crear expander para el usuario
                    with st.expander(f"{medal} {name} - {total_payout} ({num_payouts} payouts)", expanded=False):
                        if monthly_payouts:
                            # Crear DataFrame con los payouts detallados
                            payouts_df = pd.DataFrame(monthly_payouts)
                            payouts_df['fecha_payout'] = pd.to_datetime(payouts_df['fecha_payout'])
                            payouts_df = payouts_df.sort_values('fecha_payout')
                            
                            # Seleccionar y renombrar columnas
                            display_df = payouts_df[['fecha_payout', 'payout', 'herramienta']].copy()
                            display_df.columns = ['Fecha', 'Monto', 'Herramienta']
                            display_df['Fecha'] = display_df['Fecha'].dt.strftime('%d/%m/%Y')
                            display_df['Monto'] = display_df['Monto'].apply(lambda x: f"${float(x):,.2f}")
                            
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No hay payouts detallados para este mes")
                
                # Mostrar estadísticas Gold
                total_gold = gold_ranking['total_payout'].sum()
                
                st.html(f"""
                <div class="ranking-total" style="background: #fff3cd; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 7px solid #ffc107;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #856404;">Total Gold:</span>
                        <span style="font-weight: bold; color: #856404;">${total_gold:,.2f}</span>
                    </div>
                </div>
                """)
            else:
                st.markdown("""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem; text-align: center;">
                    <p style="color: #6c757d; font-size: 0.9rem; margin: 0;">
                        No hay usuarios Gold con payouts este mes
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="card ranking-panel">
                <div class="card-title">🥈 Ranking Silver</div>
                <div class="ranking-content">
            """, unsafe_allow_html=True)
            
            if not silver_ranking.empty:
                # Crear expanders para cada usuario Silver
                for _, row in silver_ranking.iterrows():
                    position = row['position']
                    name = row['name']
                    total_payout = row['total_payout_formatted']
                    num_payouts = row['num_payouts']
                    
                    # Medalla según posición
                    medal = '🥇' if position == 1 else '🥈' if position == 2 else '🥉' if position == 3 else f"{position}."
                    
                    # Obtener payouts detallados del usuario
                    user_payouts = [p for p in payouts_data if p['nick'] == row['nick'] and p['fecha_payout']]
                    
                    # Filtrar por mes seleccionado
                    target_date = datetime.now() + timedelta(days=30 * st.session_state.selected_month)
                    target_year = target_date.year
                    target_month = target_date.month
                    
                    monthly_payouts = []
                    for payout in user_payouts:
                        payout_date = datetime.fromisoformat(payout['fecha_payout'].replace('Z', '+00:00'))
                        if payout_date.year == target_year and payout_date.month == target_month:
                            monthly_payouts.append(payout)
                    
                    # Crear expander para el usuario
                    with st.expander(f"{medal} {name} - {total_payout} ({num_payouts} payouts)", expanded=False):
                        if monthly_payouts:
                            # Crear DataFrame con los payouts detallados
                            payouts_df = pd.DataFrame(monthly_payouts)
                            payouts_df['fecha_payout'] = pd.to_datetime(payouts_df['fecha_payout'])
                            payouts_df = payouts_df.sort_values('fecha_payout')
                            
                            # Seleccionar y renombrar columnas
                            display_df = payouts_df[['fecha_payout', 'payout', 'herramienta']].copy()
                            display_df.columns = ['Fecha', 'Monto', 'Herramienta']
                            display_df['Fecha'] = display_df['Fecha'].dt.strftime('%d/%m/%Y')
                            display_df['Monto'] = display_df['Monto'].apply(lambda x: f"${float(x):,.2f}")
                            
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No hay payouts detallados para este mes")
                
                # Mostrar estadísticas Silver
                total_silver = silver_ranking['total_payout'].sum()
                
                st.markdown(f"""
                <div class="ranking-total" style="background: white; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 7px solid #6c757d;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #6c757d;">Total Silver:</span>
                        <span style="font-weight: bold; color: #6c757d;">${total_silver:,.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem; text-align: center;">
                    <p style="color: #6c757d; font-size: 0.9rem; margin: 0;">
                        No hay usuarios Silver con payouts este mes
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div></div>', unsafe_allow_html=True)
    
    with ranking_sub_tab2:
        st.markdown("""
        <div class="card">
            <div class="card-title">💰 Añadir Cobro</div>
            <div style="text-align: center; padding: 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;" class="construction-icon">💰</div>
                <h2 style="color: #6c757d; margin-bottom: 1rem;">¡En Desarrollo!</h2>
                <p style="color: #6c757d; font-size: 1.1rem; margin-bottom: 2rem;">
                    Próximamente disponible.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Tab de Configuración
with tab4:
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