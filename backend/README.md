# IMOXHUB Backend API

Backend desarrollado con FastAPI para la aplicación IMOXHUB.

## Características

- **FastAPI**: Framework moderno y rápido para APIs
- **SQLAlchemy**: ORM para manejo de base de datos
- **SQLite**: Base de datos ligera para desarrollo
- **Pydantic**: Validación de datos
- **Autenticación**: Sistema de rangos (admin, gold, silver)

## Estructura de la Base de Datos

### Tabla Users
- `id`: Identificador único (Primary Key)
- `nombre`: Nombre del usuario
- `correo`: Email único del usuario
- `rango`: Rango del usuario (admin, gold, silver)
- `created_at`: Fecha de creación
- `updated_at`: Fecha de última actualización

## Instalación

1. Crear entorno virtual:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
pip install faker  # Para el script de población
```

3. Ejecutar el servidor:
```bash
python main.py
```

4. Poblar la base de datos (opcional):
```bash
python populate_db.py
```

## Endpoints Disponibles

### Usuarios
- `POST /users/` - Crear usuario
- `GET /users/` - Listar usuarios (con paginación)
- `GET /users/{user_id}` - Obtener usuario por ID
- `PUT /users/{user_id}` - Actualizar usuario
- `DELETE /users/{user_id}` - Eliminar usuario
- `GET /users/by-email/{email}` - Obtener usuario por email

### Utilidades
- `GET /health` - Verificar estado del servidor

## Documentación

Una vez ejecutado el servidor, puedes acceder a:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Próximas Funcionalidades

- [ ] Tabla de Cobros
- [ ] Tabla de Descuentos
- [ ] Sistema de autenticación JWT
- [ ] Middleware de autorización
- [ ] Logs y monitoreo
