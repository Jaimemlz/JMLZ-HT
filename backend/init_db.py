"""
Script para inicializar la base de datos en Render.com
Este script crea las tablas necesarias en la base de datos PostgreSQL
"""
import os
import sys
from sqlalchemy import create_engine
from main import Base

# Obtener la URL de la base de datos desde las variables de entorno
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ ERROR: DATABASE_URL no está configurado")
    sys.exit(1)

print(f"📦 Conectando a la base de datos...")
print(f"🔗 DATABASE_URL: {DATABASE_URL[:20]}...")  # Mostrar solo los primeros caracteres por seguridad

try:
    # Crear el engine
    if DATABASE_URL.startswith("postgresql://"):
        engine = create_engine(DATABASE_URL)
    else:
        print("❌ ERROR: La URL de la base de datos no es PostgreSQL")
        sys.exit(1)
    
    # Crear todas las tablas
    print("🗄️  Creando tablas en la base de datos...")
    Base.metadata.create_all(bind=engine)
    print("✅ Base de datos inicializada correctamente!")
    
except Exception as e:
    print(f"❌ ERROR al inicializar la base de datos: {str(e)}")
    sys.exit(1)

