"""
Script para inicializar la base de datos en Render.com
Este script crea las tablas necesarias en la base de datos PostgreSQL
"""
import os
import sys
import traceback

print("🔄 Iniciando inicialización de base de datos...")

try:
    from sqlalchemy import create_engine, Column, Integer, String, DateTime, Enum
    from sqlalchemy.ext.declarative import declarative_base
    from enum import Enum as PyEnum
    from datetime import datetime, timezone
    
    print("✅ Imports exitosos")
    
    # Definir Enums
    class UserRank(PyEnum):
        ADMIN = "admin"
        GOLD = "gold"
        SILVER = "silver"

    class Herramienta(PyEnum):
        AXI_SELECT = "AXI_SELECT"
        FTMO = "FTMO"
        DARWINEX_ZERO = "DARWINEX_ZERO"
        TTP = "TTP"
        CUENTA_PERSONAL = "CUENTA_PERSONAL"
        OTROS = "OTROS"
    
    # Definir Base
    Base = declarative_base()
    
    # Modelos de base de datos
    class User(Base):
        __tablename__ = "users"
        
        nick = Column(String(50), primary_key=True, index=True)
        name = Column(String(100), nullable=False)
        correo = Column(String(100), unique=True, index=True, nullable=False)
        rank = Column(Enum(UserRank), nullable=False, default=UserRank.SILVER)
        created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
        updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    class Payout(Base):
        __tablename__ = "payouts"
        
        id = Column(Integer, primary_key=True, index=True)
        nick = Column(String(50), nullable=False, index=True)
        payout = Column(String(20), nullable=False)
        fecha_creacion = Column(DateTime, default=lambda: datetime.now(timezone.utc))
        fecha_payout = Column(DateTime, nullable=True)
        herramienta = Column(Enum(Herramienta), nullable=False)
        created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
        updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    print("✅ Modelos definidos")
    
    # Obtener la URL de la base de datos desde las variables de entorno
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    if not DATABASE_URL:
        print("❌ ERROR: DATABASE_URL no está configurado")
        print("Variables de entorno disponibles:", list(os.environ.keys()))
        sys.exit(1)
    
    print(f"📦 Conectando a la base de datos...")
    print(f"🔗 DATABASE_URL: {DATABASE_URL[:20]}...")  # Mostrar solo los primeros caracteres por seguridad
    
    # Crear el engine
    if DATABASE_URL.startswith("postgresql://"):
        engine = create_engine(DATABASE_URL)
    else:
        print(f"⚠️  WARNING: La URL no es PostgreSQL: {DATABASE_URL[:30]}...")
        print("Intentando continuar de todos modos...")
        engine = create_engine(DATABASE_URL)
    
    print("🔧 Creando tablas en la base de datos...")
    Base.metadata.create_all(bind=engine)
    print("✅ Base de datos inicializada correctamente!")
    
except Exception as e:
    print(f"❌ ERROR al inicializar la base de datos: {str(e)}")
    print("Traceback completo:")
    traceback.print_exc()
    sys.exit(1)

