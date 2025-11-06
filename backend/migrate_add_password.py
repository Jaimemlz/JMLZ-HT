"""
Script de migración para agregar la columna password_hash a la tabla users
Ejecuta: python migrate_add_password.py
"""
import os
import sys
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError

# Configuración de la base de datos
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./imoxhub.db")

def check_column_exists(engine, table_name, column_name):
    """Verifica si una columna existe en una tabla"""
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns

def migrate_sqlite(engine):
    """Migración para SQLite"""
    print("🔍 Detectada base de datos SQLite")
    with engine.begin() as conn:  # begin() hace commit automático
        # Verificar si la columna ya existe
        if check_column_exists(engine, 'users', 'password_hash'):
            print("✅ La columna 'password_hash' ya existe. No se requiere migración.")
            return True
        
        print("📝 Agregando columna 'password_hash' a la tabla 'users'...")
        try:
            # SQLite no soporta ALTER TABLE ADD COLUMN IF NOT EXISTS directamente
            # pero podemos intentar agregarla
            conn.execute(text("ALTER TABLE users ADD COLUMN password_hash VARCHAR(255)"))
            print("✅ Columna 'password_hash' agregada exitosamente.")
            return True
        except OperationalError as e:
            error_msg = str(e).lower()
            if "duplicate column name" in error_msg or "already exists" in error_msg or "duplicate" in error_msg:
                print("✅ La columna 'password_hash' ya existe.")
                return True
            else:
                print(f"❌ Error al agregar la columna: {e}")
                return False

def migrate_postgresql(engine):
    """Migración para PostgreSQL"""
    print("🔍 Detectada base de datos PostgreSQL")
    
    # Verificar si la columna ya existe ANTES de abrir la transacción
    if check_column_exists(engine, 'users', 'password_hash'):
        print("✅ La columna 'password_hash' ya existe. No se requiere migración.")
        return True
    
    print("📝 Agregando columna 'password_hash' a la tabla 'users'...")
    with engine.begin() as conn:  # begin() hace commit automático
        try:
            # PostgreSQL NO soporta IF NOT EXISTS en ALTER TABLE ADD COLUMN
            # Por eso verificamos antes con check_column_exists
            conn.execute(text("ALTER TABLE users ADD COLUMN password_hash VARCHAR(255)"))
            print("✅ Columna 'password_hash' agregada exitosamente.")
            return True
        except OperationalError as e:
            error_msg = str(e).lower()
            # Verificar si el error es porque la columna ya existe (por si acaso)
            if "already exists" in error_msg or "duplicate" in error_msg:
                print("✅ La columna 'password_hash' ya existe.")
                return True
            else:
                print(f"❌ Error al agregar la columna: {e}")
                return False

def main():
    print("🚀 Iniciando migración de base de datos...")
    print(f"📊 Base de datos: {DATABASE_URL}")
    
    # Configurar engine según el tipo de base de datos
    if DATABASE_URL.startswith("postgresql://"):
        engine = create_engine(DATABASE_URL)
        success = migrate_postgresql(engine)
    else:
        # SQLite
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
        success = migrate_sqlite(engine)
    
    if success:
        print("\n✅ Migración completada exitosamente!")
        print("💡 Los usuarios existentes podrán establecer su contraseña en su primer login.")
    else:
        print("\n❌ La migración falló. Por favor revisa los errores arriba.")
        sys.exit(1)

if __name__ == "__main__":
    main()

