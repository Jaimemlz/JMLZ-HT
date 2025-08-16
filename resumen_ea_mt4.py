import streamlit as st
import plotly.express as px
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Resumen de EAs MT4", layout="wide")
st.title("📈 Comparativa de estrategias por EA (MetaTrader 4)")

archivo = st.file_uploader("📤 Sube tu archivo HTML exportado de MetaTrader 4", type=["htm", "html"])

if archivo:
    soup = BeautifulSoup(archivo, 'html.parser')
    rows = soup.find_all('tr', align='right')

    datos = []

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

        # Ordenar
          # Ordenar y preparar resumen
        resumen = resumen.sort_values(by="Beneficio_total_raw", ascending=False)

        def resaltar_beneficio(val):
            color = '#b6f3c0' if val > 0 else '#f3b6b6'
            return f'background-color: {color}; color: black;'

        styler = resumen.style.applymap(resaltar_beneficio, subset=['Beneficio_total_raw'])

        styled_df = styler.format(None, subset=['Beneficio_total_raw']).set_table_styles([{
            'selector': f'.col_heading.level0:nth-child({resumen.columns.get_loc("Beneficio_total_raw")+1})',
            'props': [('display', 'none')]
        }, {
            'selector': f'.rowheading:nth-child({resumen.columns.get_loc("Beneficio_total_raw")+1})',
            'props': [('display', 'none')]
        }, {
            'selector': f'.data:nth-child({resumen.columns.get_loc("Beneficio_total_raw")+1})',
            'props': [('display', 'none')]
        }])

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

        # Mostrar tabla resumen
        st.subheader("📊 Comparativa de EAs")
        st.dataframe(
            resumen_filtrado.style.applymap(resaltar_beneficio, subset=['Beneficio_total_raw']).format(None, subset=['Beneficio_total_raw']),
            use_container_width=True,
            column_config={"Beneficio_total_raw": None}
        )

        # Crear gráfico de beneficio acumulado
        df_filtrado['Fecha'] = df_filtrado['Close'].dt.date
        beneficios_diarios = df_filtrado.groupby(['EA', 'Fecha'])['Beneficio'].sum().reset_index()
        beneficios_diarios['Beneficio_acumulado'] = beneficios_diarios.groupby('EA')['Beneficio'].cumsum()

        fig = px.line(
            beneficios_diarios,
            x="Fecha",
            y="Beneficio_acumulado",
            color="EA",
            markers=True,
            title="📈 Evolución del beneficio acumulado",
            labels={"Beneficio_acumulado": "Beneficio acumulado", "Fecha": "Fecha"}
        )

        # Añadir tooltip personalizado
        fig.update_traces(
            hovertemplate=
            "<b> %{customdata[0]} </b><br>" +
            "<b>Fecha:</b> %{x|%d-%m-%Y}<br>" +
            "<b>Beneficio acumulado:</b> $%{y:.2f}<extra></extra>",
            customdata=beneficios_diarios[["EA"]]
        )

        fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))

        st.subheader("📉 Beneficio acumulado por EA")
        st.plotly_chart(fig, use_container_width=True)

        # Mostrar operaciones individuales por EA y símbolo
        grupos_ordenados = df_filtrado.groupby(["EA", "Símbolo"]).agg(Beneficio_total=('Beneficio', 'sum')).reset_index()
        grupos_ordenados = grupos_ordenados.sort_values(by="Beneficio_total", ascending=False)

        for _, row in grupos_ordenados.iterrows():
            ea = row["EA"]
            symbol = row["Símbolo"]
            grupo = df_filtrado[(df_filtrado["EA"] == ea) & (df_filtrado["Símbolo"] == symbol)]
            with st.expander(f"📌 {ea} - {symbol} ({len(grupo)} operaciones)"):
                st.dataframe(grupo.sort_values(by="Open"), use_container_width=True)

        st.subheader("📎 Detalle completo de operaciones")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No se encontraron operaciones válidas en el archivo.")