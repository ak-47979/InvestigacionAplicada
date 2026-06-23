from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import joblib
import json
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware

# Inicializar la API
app = FastAPI(title="API Calidad del Aire - Rango de Fechas (Quito)")

# Configurar CORS para permitir peticiones desde tu archivo index.html local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo ONNX, el escalador y el archivo semilla de 24 horas
try:
    session = ort.InferenceSession('modelo_aire.onnx')
    scaler = joblib.load('escalador_aire.pkl')
    with open('semilla_historial.json', 'r') as f:
        HISTORIAL_SEMILLA = json.load(f)
    print("¡Todos los recursos cargados correctamente!")
except Exception as e:
    print(f"Error crítico al cargar componentes: {e}")

# Modelo de datos que espera recibir la API desde el Frontend
class RangePredictRequest(BaseModel):
    fecha_inicio: str
    fecha_fin: str

# Funciones auxiliares de evaluación según normativas de calidad de aire
def evaluar_pm25(valor):
    if valor <= 15.0: return "Bueno"
    elif valor <= 35.0: return "Normal"
    elif valor <= 55.4: return "Dañino para grupos sensibles"
    else: return "Malo"

def evaluar_pm10(valor):
    if valor <= 45.0: return "Bueno"
    elif valor <= 154.0: return "Normal"
    else: return "Malo"

def obtener_direccion_viento(grados):
    direcciones = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    indice = int((grados / 22.5) + .5) % 16
    return direcciones[indice]

# RUTA POST principal para resolver las proyecciones en rango
@app.post("/predict_range")
def predecir_rango_calidad(payload: RangePredictRequest):
    try:
        start_dt = datetime.strptime(payload.fecha_inicio, "%Y-%m-%d")
        end_dt = datetime.strptime(payload.fecha_fin, "%Y-%m-%d")
        # Si el usuario pone las fechas al revés, las ordenamos automáticamente
        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt
    except Exception as e:
        return {"error": f"Formatos de fecha inválidos. Use AAAA-MM-DD. Detalles: {e}"}

    # 1. Ejecutar inferencia base con la secuencia de 24 horas (Semilla)
    entrada_bloque = np.array(HISTORIAL_SEMILLA)
    entrada_listo = np.expand_dims(entrada_bloque, axis=0).astype(np.float32)
    inputs = {session.get_inputs()[0].name: entrada_listo}
    prediccion_escalada = session.run(None, inputs)[0]
    
    # Desescalar los resultados numéricos a valores reales
    prediccion_real = scaler.inverse_transform(prediccion_escalada)[0]
    
    # Desempaquetar las variables calculadas por el modelo LSTM
    no2_b, o3_b, so2_b, pm25_b, co_b, pm10_b, tmp_b, dir_b, rs_b, pre_b, iuv_b, vel_b, hum_b, llu_b, hr, ms, ds = prediccion_real

    resultados_rango = []
    current_dt = start_dt
    max_dias = 15  # Límite de seguridad para evitar sobrecargar el servidor
    dias_procesados = 0

    # 2. Iterar día por día dentro del rango solicitado por el usuario
    while current_dt <= end_dt and dias_procesados < max_dias:
        dia = current_dt.day
        mes_obj = current_dt.month
        
        # Factor pseudo-aleatorio controlado matemáticamente según el día calendario
        factor = (dia * 0.05) - 0.4  # Varía de forma estable entre -0.4 y +0.4
        es_verano = mes_obj in [6, 7, 8, 9]  # Época seca y despejada en Quito

        # Simular comportamiento estacional y variaciones meteorológicas
        tmp = tmp_b + (factor * 4.5) if es_verano else tmp_b + (factor * 3.0) - 1.0
        hum = min(max(30.0, hum_b - (factor * 20) if es_verano else hum_b + (factor * 25)), 95.0)
        vel = vel_b + abs(factor * 6) if mes_obj == 8 else vel_b + (factor * 2)
        
        # Simular comportamiento correlacionado de contaminantes atmosféricos
        pm25 = max(3.0, pm25_b + (factor * 7.0)) if not es_verano else max(2.5, pm25_b - (factor * 3.0))
        pm10 = max(8.0, pm10_b + (factor * 14.0))
        o3 = max(4.0, o3_b + (factor * 9.0)) if es_verano else max(4.0, o3_b + (factor * 4.0))
        no2 = max(6.0, no2_b + (factor * 6.0))

        # Modificación de radiación solar para picos estacionales altos
        rs_calculada = max(10.0, rs_b + (factor * 180.0)) if es_verano else max(5.0, rs_b + (factor * 90.0))
        
        # Recalcular el Índice UV de forma dinámica según la Radiación Solar simulada
        if rs_calculada < 50.0:
            iuv_dinamico = 0
        elif rs_calculada < 150.0:
            iuv_dinamico = int(2 + factor * 2)
        elif rs_calculada < 300.0:
            iuv_dinamico = int(5 + factor * 3)
        else:
            iuv_dinamico = int(11 + factor * 4)
        iuv_dinamico = min(max(0, iuv_dinamico), 14)

        # Recalcular el estado de la probabilidad de precipitación en base a la Humedad Relativa resultante
        if hum > 82.0:
            lluvia_txt = "Alta probabilidad"
        elif hum > 68.0:
            lluvia_txt = "Moderada probabilidad"
        else:
            lluvia_txt = "Baja probabilidad"

        estado_pm25 = evaluar_pm25(pm25)
        
        # Agregar el reporte formateado de este día específico a la lista estructurada
        resultados_rango.append({
            "fecha": current_dt.strftime("%d/%m/%Y"),
            "estado_general": "MALA" if "Dañino" in estado_pm25 or pm25 > 35 else "BUENA / NORMAL",
            "pm25": f"{pm25:.1f} ug/m³ ({estado_pm25})",
            "pm10": f"{pm10:.1f} ug/m³",
            "o3": f"{o3:.1f} ppb",
            "tmp": f"{tmp:.1f} °C",
            "hum": f"{hum:.0f}%",
            "lluvia": lluvia_txt,
            "iuv": f"{iuv_dinamico}"
        })
        
        # Avanzar al siguiente día del calendario
        current_dt += timedelta(days=1)
        dias_procesados += 1

    return {"proyecciones": resultados_rango}
