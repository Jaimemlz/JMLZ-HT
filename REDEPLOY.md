# 🔄 Cómo Hacer Redeploy en Render

## Problemas Corregidos

Se han hecho las siguientes correcciones:

1. ✅ **backend/init_db.py**: Ahora define todos los modelos directamente sin depender de main.py
2. ✅ **render.yaml**: Corregido el comando de build para cambiar al directorio correcto
3. ✅ **render.yaml**: Agregado urlPrefix para que API_URL use HTTPS
4. ✅ **backend/main.py**: Agregada ruta raíz "/" para verificar que el servidor funciona

## Pasos para Redeploy

### Opción 1: Auto-redeploy desde GitHub (Recomendado)

1. Sube los cambios a GitHub:
```bash
git add .
git commit -m "Fix: Corregir configuración de deployment para backend"
git push origin main
```

2. Render detectará el push automáticamente
3. Render iniciará un nuevo build automáticamente

### Opción 2: Redeploy Manual desde Render Dashboard

1. Ve a tu dashboard en https://dashboard.render.com
2. Encuentra el servicio `imoxhub-backend`
3. Haz clic en "Manual Deploy" → "Deploy latest commit"

## Verificación del Backend

Después del redeploy, verifica:

1. **Health Check**: https://imoxhub-backend.onrender.com/health
   - Debe devolver: `{"status": "ok", "message": "IMOXHUB API está funcionando"}`

2. **Raíz**: https://imoxhub-backend.onrender.com/
   - Debe devolver: `{"status": "ok", "message": "IMOXHUB API está funcionando", "docs": "/docs"}`

3. **Documentación**: https://imoxhub-backend.onrender.com/docs
   - Debe mostrar la documentación interactiva de FastAPI

## Verificación del Frontend

Después de que el backend esté funcionando, verifica el frontend:

1. Ve a https://imoxhub-frontend.onrender.com
2. Verifica que no aparezcan errores de conexión con el backend

## Diagnóstico de Problemas

Si el backend sigue sin funcionar:

### 1. Verifica los logs

En Render Dashboard → tu servicio backend → Logs

Busca errores relacionados con:
- `DATABASE_URL`
- `psycopg2`
- `init_db`

### 2. Verifica las variables de entorno

En Render Dashboard → tu servicio backend → Environment

Asegúrate de que:
- `DATABASE_URL` esté configurado
- Apunte a la base de datos `imoxhub-db`

### 3. Verifica la base de datos

En Render Dashboard → Databases → `imoxhub-db`

Asegúrate de que:
- El estado sea "Available"
- La conexión esté activa

## Comandos Útiles

### Ver logs en tiempo real
```bash
# En Render Dashboard → Logs
# O usa la API de Render para obtener logs
```

### Verificar variables de entorno en el build
El script `init_db.py` ahora imprime información de debug durante el build.

### Conectar directamente a la base de datos (local)
```bash
psql <connection_string_from_render>
```

## Problemas Comunes

### "DATABASE_URL no está configurado"
- Verifica que la base de datos se haya creado
- Verifica que el servicio backend esté vinculado a la base de datos
- Espera unos minutos si acabas de crear la base de datos

### "No se puede conectar con el backend"
- Verifica que el backend esté en estado "Live"
- Espera a que salga del "Sleep mode" (puede tardar 60 segundos)
- Verifica que API_URL en el frontend apunte a la URL correcta

### "Tabla no existe"
- El script init_db.py debe ejecutarse automáticamente durante el build
- Revisa los logs del build para ver si hubo errores
- Puedes ejecutar init_db manualmente si es necesario

## Siguientes Pasos

Una vez que el backend esté funcionando:

1. Pobla la base de datos con datos iniciales (opcional)
2. Verifica todos los endpoints en /docs
3. Prueba el frontend completo
4. Considera configurar un servicio de ping para mantener el servicio activo

## Notas Importantes

⚠️ **Plan Gratuito**: El servicio entra en "sleep mode" después de 15 minutos de inactividad

💡 **Soluciones**:
- El primer request después del sleep puede tardar hasta 60 segundos
- Considera usar un servicio externo de ping/health check
- Considera actualizar a un plan de pago si necesitas disponibilidad 24/7

✅ **Todas las configuraciones están listas**: Una vez que hagas el redeploy, todo debería funcionar correctamente.

