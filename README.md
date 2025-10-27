# IMOXHUB 🏆

Sistema de gestión y visualización de payouts y rankings para la comunidad IMOXHUB.

## 🏗️ Arquitectura

Esta aplicación consta de dos servicios:

- **Frontend**: Aplicación Streamlit con visualizaciones interactivas
- **Backend**: API REST desarrollada con FastAPI
- **Base de datos**: PostgreSQL (en producción) / SQLite (en desarrollo)

## 🚀 Deployment en Render.com

Para desplegar esta aplicación en Render.com, consulta la [Guía de Deployment](./DEPLOY.md).

### Resumen rápido:

1. Asegúrate de tener tu código en GitHub
2. Conecta tu repositorio en [Render.com](https://render.com)
3. Render usará automáticamente el archivo `render.yaml` para configurar todo
4. ¡Espera a que se complete el build!

## 💻 Desarrollo Local

### Requisitos previos

- Python 3.11+
- Git

### Configuración del Backend

```bash
# 1. Navegar al directorio backend
cd backend

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar el servidor
python main.py
```

El backend estará disponible en `http://localhost:8000`
- Documentación: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Configuración del Frontend

```bash
# En la raíz del proyecto
# 1. Crear entorno virtual (si no existe)
python3 -m venv venv
source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar Streamlit
streamlit run imoxhub.py
```

La aplicación se abrirá en `http://localhost:8501`

## 📁 Estructura del Proyecto

```
JMLZ-HT/
├── imoxhub.py              # Frontend Streamlit
├── requirements.txt         # Dependencias del frontend
├── backend/
│   ├── main.py             # Backend FastAPI
│   ├── init_db.py          # Script de inicialización DB
│   ├── requirements.txt    # Dependencias del backend
│   └── imoxhub.db          # Base de datos SQLite (local)
├── render.yaml             # Configuración para Render.com
└── DEPLOY.md               # Guía completa de deployment
```

## 🔧 Variables de Entorno

### Desarrollo Local

Crea un archivo `.env` en la raíz del proyecto:

```env
API_URL=http://localhost:8000
```

### Producción (Render)

Las variables de entorno se configuran automáticamente en Render:
- `API_URL`: URL del backend (configurado automáticamente)
- `DATABASE_URL`: URL de PostgreSQL (configurado automáticamente)

## 📊 Características

### Backend (FastAPI)
- ✅ API RESTful completa
- ✅ Manejo de usuarios y rangos (admin, gold, silver)
- ✅ Sistema de payouts con diferentes herramientas
- ✅ Base de datos con SQLAlchemy
- ✅ Documentación automática con Swagger
- ✅ Health check endpoint

### Frontend (Streamlit)
- ✅ Visualizaciones interactivas con Plotly
- ✅ Rankings y estadísticas
- ✅ Generación de imágenes para Instagram Stories
- ✅ Exportación de datos
- ✅ Dashboard completo

## 🔒 Seguridad

- La aplicación utiliza variables de entorno para configuraciones sensibles
- Base de datos PostgreSQL en producción
- HTTPS automático en Render.com

## 📝 Notas

- El plan gratuito de Render puede poner los servicios en "sleep mode" después de 15 minutos de inactividad
- El primer request tras el sleep puede tardar hasta 60 segundos en responder

## 📚 Recursos

- [Documentación de FastAPI](https://fastapi.tiangolo.com/)
- [Documentación de Streamlit](https://docs.streamlit.io/)
- [Documentación de Render.com](https://render.com/docs)
- [Documentación de Plotly](https://plotly.com/python/)
