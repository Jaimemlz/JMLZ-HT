#!/bin/bash

# Script para iniciar el backend de IMOXHUB

echo "🚀 Iniciando IMOXHUB Backend..."

# Verificar si existe el entorno virtual
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Instalar dependencias
echo "📚 Instalando dependencias..."
pip install -r requirements.txt
pip install faker

# Ejecutar migración de base de datos (agregar columna password_hash si no existe)
echo "🔧 Ejecutando migración de base de datos..."
python migrate_add_password.py

# Crear base de datos y poblar con datos de ejemplo (si no existe)
if [ ! -f "imoxhub.db" ]; then
    echo "🗄️ Creando base de datos..."
    python populate_db.py
fi

# Iniciar servidor
echo "🌐 Iniciando servidor en http://localhost:8000"
echo "📖 Documentación disponible en http://localhost:8000/docs"
python main.py
