from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Enum, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, EmailStr
from datetime import datetime, timezone, timedelta
from enum import Enum as PyEnum
import os
from typing import List, Optional
from jose import JWTError, jwt
import bcrypt
from collections import defaultdict
from datetime import datetime as dt

# Configuración de la base de datos
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./imoxhub.db")

# Configuración de seguridad
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production-min-32-chars")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 horas

# Contexto para hashing de contraseñas
# Usar bcrypt directamente para evitar problemas de compatibilidad
import bcrypt

# Protección contra fuerza bruta (intentos de login por nick)
login_attempts = defaultdict(list)
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 15

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
    password_hash = Column(String(255), nullable=True)  # Nullable para usuarios nuevos sin contraseña
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
    has_password: bool  # Indica si el usuario tiene contraseña establecida
    
    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    name: Optional[str] = None
    correo: Optional[EmailStr] = None
    rank: Optional[UserRank] = None

# Esquemas para autenticación
class LoginRequest(BaseModel):
    nick: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse
    first_login: bool  # True si es el primer login (sin contraseña previa)

class SetPasswordRequest(BaseModel):
    password: str
    confirm_password: str

class ResetPasswordRequest(BaseModel):
    target_nick: str  # Nick del usuario cuya contraseña se va a resetear

class TokenData(BaseModel):
    nick: Optional[str] = None

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

# Funciones de utilidad para autenticación
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifica si una contraseña coincide con el hash"""
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception as e:
        return False

def get_password_hash(password: str) -> str:
    """Genera un hash de la contraseña"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Crea un token JWT"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def check_login_attempts(nick: str) -> bool:
    """Verifica si el usuario está bloqueado por intentos fallidos"""
    now = dt.now()
    # Limpiar intentos antiguos
    login_attempts[nick] = [
        attempt_time for attempt_time in login_attempts[nick]
        if (now - attempt_time).total_seconds() < LOCKOUT_DURATION_MINUTES * 60
    ]
    
    # Verificar si hay demasiados intentos
    if len(login_attempts[nick]) >= MAX_LOGIN_ATTEMPTS:
        return False
    return True

def record_failed_login(nick: str):
    """Registra un intento de login fallido"""
    login_attempts[nick].append(dt.now())

def clear_login_attempts(nick: str):
    """Limpia los intentos de login para un usuario"""
    if nick in login_attempts:
        del login_attempts[nick]

# Security scheme
security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Obtiene el usuario actual desde el token JWT"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudo validar el token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        nick: str = payload.get("sub")
        if nick is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Buscar usuario (case-insensitive para mantener compatibilidad)
    user = db.query(User).filter(func.lower(User.nick) == nick.lower()).first()
    if user is None:
        raise credentials_exception
    return user

def get_user_response(user: User) -> UserResponse:
    """Convierte un User a UserResponse"""
    # Considerar None o cadena vacía como sin contraseña
    has_password = user.password_hash is not None and (
        isinstance(user.password_hash, str) and user.password_hash.strip() != ""
    )
    return UserResponse(
        nick=user.nick,
        name=user.name,
        correo=user.correo,
        rank=user.rank,
        created_at=user.created_at,
        updated_at=user.updated_at,
        has_password=has_password
    )

# Rutas de autenticación
@app.post("/auth/login", response_model=LoginResponse)
def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """Endpoint de login"""
    # Verificar bloqueo por intentos fallidos
    if not check_login_attempts(login_data.nick):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Demasiados intentos fallidos. Intenta de nuevo en {LOCKOUT_DURATION_MINUTES} minutos."
        )
    
    # Buscar usuario (case-insensitive)
    # Normalizar el nick a minúsculas para la búsqueda
    nick_normalized = login_data.nick.lower().strip()
    user = db.query(User).filter(func.lower(User.nick) == nick_normalized).first()
    if not user:
        record_failed_login(login_data.nick)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales inválidas"
        )
    
    # Verificar si es primer login (sin contraseña)
    # Considerar None o cadena vacía como sin contraseña
    first_login = user.password_hash is None or (isinstance(user.password_hash, str) and user.password_hash.strip() == "")
    
    if first_login:
        # Si es primer login, permitir acceso sin contraseña
        # El frontend deberá pedirle que establezca su contraseña
        clear_login_attempts(login_data.nick)
    else:
        # Verificar contraseña solo si tiene contraseña establecida
        if not login_data.password:
            record_failed_login(login_data.nick)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credenciales inválidas"
            )
        if not verify_password(login_data.password, user.password_hash):
            record_failed_login(login_data.nick)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credenciales inválidas"
            )
        clear_login_attempts(login_data.nick)
    
    # Crear token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.nick},
        expires_delta=access_token_expires
    )
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user=get_user_response(user),
        first_login=first_login
    )

@app.post("/auth/set-password")
def set_password(
    password_data: SetPasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Endpoint para establecer contraseña en primer login"""
    # Validar que las contraseñas coincidan
    if password_data.password != password_data.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Las contraseñas no coinciden"
        )
    
    # Validar longitud mínima
    if len(password_data.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La contraseña debe tener al menos 6 caracteres"
        )
    
    # Actualizar contraseña
    try:
        current_user.password_hash = get_password_hash(password_data.password)
        current_user.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(current_user)
        
        return {
            "message": "Contraseña establecida correctamente",
            "user": get_user_response(current_user)
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al establecer la contraseña: {str(e)}"
        )

@app.get("/auth/me", response_model=UserResponse)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Obtiene la información del usuario actual autenticado"""
    return get_user_response(current_user)

@app.post("/auth/reset-password")
def reset_password(
    reset_data: ResetPasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Endpoint para que el admin resetee la contraseña de un usuario"""
    # Verificar que el usuario actual es admin
    if current_user.rank != UserRank.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Solo los administradores pueden resetear contraseñas"
        )
    
    # Buscar el usuario objetivo (case-insensitive)
    target_nick_normalized = reset_data.target_nick.lower().strip()
    target_user = db.query(User).filter(func.lower(User.nick) == target_nick_normalized).first()
    
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuario no encontrado"
        )
    
    # Resetear la contraseña (establecer a None para que tenga que establecer una nueva)
    target_user.password_hash = None
    target_user.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(target_user)
    
    return {
        "message": f"Contraseña reseteada para el usuario {target_user.nick}. Deberá establecer una nueva contraseña en su próximo login.",
        "user": get_user_response(target_user)
    }

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
    return get_user_response(db_user)

@app.get("/users/", response_model=List[UserResponse])
def get_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = db.query(User).offset(skip).limit(limit).all()
    return [get_user_response(user) for user in users]

@app.get("/users/{nick}", response_model=UserResponse)
def get_user(nick: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.nick == nick).first()
    if user is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return get_user_response(user)

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
    return get_user_response(user)

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
    return get_user_response(user)

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
