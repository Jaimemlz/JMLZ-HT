from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, EmailStr
from datetime import datetime, timezone
from enum import Enum as PyEnum
import os
from typing import List, Optional

# Configuración de la base de datos
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./imoxhub.db")

# Configurar engine según el tipo de base de datos
if DATABASE_URL.startswith("postgresql://"):
    engine = create_engine(DATABASE_URL)
else:
    # SQLite
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Enums
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
    payout = Column(String(20), nullable=False)  # Monto en dólares como string para manejar decimales
    fecha_creacion = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    fecha_payout = Column(DateTime, nullable=True)  # Puede ser null si aún no se ha pagado
    herramienta = Column(Enum(Herramienta), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

# Esquemas Pydantic
class UserBase(BaseModel):
    nick: str
    name: str
    correo: EmailStr
    rank: UserRank = UserRank.SILVER

class UserCreate(UserBase):
    pass

class UserResponse(UserBase):
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    name: Optional[str] = None
    correo: Optional[EmailStr] = None
    rank: Optional[UserRank] = None

# Esquemas Pydantic para Payout
class PayoutBase(BaseModel):
    nick: str
    payout: str
    fecha_payout: Optional[datetime] = None
    herramienta: Herramienta

class PayoutCreate(PayoutBase):
    pass

class PayoutResponse(PayoutBase):
    id: int
    fecha_creacion: datetime
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class PayoutUpdate(BaseModel):
    payout: Optional[str] = None
    fecha_payout: Optional[datetime] = None
    herramienta: Optional[Herramienta] = None

# Crear las tablas
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI(title="IMOXHUB API", version="1.0.0")

# Dependency para obtener la sesión de la base de datos
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Rutas de usuarios
@app.post("/users/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # Verificar si el nick ya existe
    db_user = db.query(User).filter(User.nick == user.nick).first()
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="El nick ya está registrado"
        )
    
    # Verificar si el correo ya existe
    db_user = db.query(User).filter(User.correo == user.correo).first()
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="El correo electrónico ya está registrado"
        )
    
    db_user = User(**user.model_dump())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/", response_model=List[UserResponse])
def get_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@app.get("/users/{nick}", response_model=UserResponse)
def get_user(nick: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.nick == nick).first()
    if user is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return user

@app.put("/users/{nick}", response_model=UserResponse)
def update_user(nick: str, user_update: UserUpdate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.nick == nick).first()
    if user is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Verificar si el nuevo correo ya existe (si se está actualizando)
    if user_update.correo and user_update.correo != user.correo:
        existing_user = db.query(User).filter(User.correo == user_update.correo).first()
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="El correo electrónico ya está registrado"
            )
    
    # Actualizar solo los campos proporcionados
    update_data = user_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(user, field, value)
    
    user.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)
    return user

@app.delete("/users/{nick}")
def delete_user(nick: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.nick == nick).first()
    if user is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    db.delete(user)
    db.commit()
    return {"message": "Usuario eliminado correctamente"}

@app.get("/users/by-email/{email}", response_model=UserResponse)
def get_user_by_email(email: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.correo == email).first()
    if user is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return user

# Rutas de Payout
@app.post("/payouts/", response_model=PayoutResponse, status_code=status.HTTP_201_CREATED)
def create_payout(payout: PayoutCreate, db: Session = Depends(get_db)):
    # Verificar que el usuario existe
    user = db.query(User).filter(User.nick == payout.nick).first()
    if not user:
        raise HTTPException(
            status_code=404,
            detail="Usuario no encontrado"
        )
    
    db_payout = Payout(**payout.model_dump())
    db.add(db_payout)
    db.commit()
    db.refresh(db_payout)
    return db_payout

@app.get("/payouts/", response_model=List[PayoutResponse])
def get_payouts(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    payouts = db.query(Payout).offset(skip).limit(limit).all()
    return payouts

@app.get("/payouts/{payout_id}", response_model=PayoutResponse)
def get_payout(payout_id: int, db: Session = Depends(get_db)):
    payout = db.query(Payout).filter(Payout.id == payout_id).first()
    if payout is None:
        raise HTTPException(status_code=404, detail="Payout no encontrado")
    return payout

@app.get("/payouts/by-nick/{nick}", response_model=List[PayoutResponse])
def get_payouts_by_nick(nick: str, db: Session = Depends(get_db)):
    payouts = db.query(Payout).filter(Payout.nick == nick).all()
    return payouts

@app.put("/payouts/{payout_id}", response_model=PayoutResponse)
def update_payout(payout_id: int, payout_update: PayoutUpdate, db: Session = Depends(get_db)):
    payout = db.query(Payout).filter(Payout.id == payout_id).first()
    if payout is None:
        raise HTTPException(status_code=404, detail="Payout no encontrado")
    
    # Actualizar solo los campos proporcionados
    update_data = payout_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(payout, field, value)
    
    payout.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(payout)
    return payout

@app.delete("/payouts/{payout_id}")
def delete_payout(payout_id: int, db: Session = Depends(get_db)):
    payout = db.query(Payout).filter(Payout.id == payout_id).first()
    if payout is None:
        raise HTTPException(status_code=404, detail="Payout no encontrado")
    
    db.delete(payout)
    db.commit()
    return {"message": "Payout eliminado correctamente"}

# Ruta de salud
@app.get("/")
def root():
    return {"status": "ok", "message": "IMOXHUB API está funcionando", "docs": "/docs"}

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "IMOXHUB API está funcionando"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
