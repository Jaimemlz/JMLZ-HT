# Resumen de EAs MT4

Esta aplicación de Streamlit analiza archivos HTML exportados de MetaTrader 4 para generar un resumen comparativo de estrategias por Expert Advisor (EA).

## Instalación y Ejecución

### 1. Crear y activar el entorno virtual
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar la aplicación
```bash
streamlit run resumen_ea_mt4.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## Características

- 📤 Carga archivos HTML exportados de MT4
- 📊 Análisis comparativo por EA y símbolo
- 📈 Gráficos de evolución del beneficio acumulado
- 🎯 Filtrado por EA específica
- 📋 Detalle completo de operaciones

## Solución de Problemas

Si encuentras el error de MIME type:
1. Asegúrate de usar un entorno virtual
2. Verifica que todas las dependencias estén instaladas
3. Reinicia el navegador y limpia la caché
4. Si persiste, ejecuta: `streamlit cache clear`

## Dependencias

- streamlit
- plotly
- beautifulsoup4
- pandas
