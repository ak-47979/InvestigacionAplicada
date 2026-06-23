from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import onnxruntime as ort
import joblib
import json
import os
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="API Calidad del Aire - Quito")

# Habilitar CORS para que tu interfaz web pueda consultar la API desde cualquier lado
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo, escalador e historial semilla
try:
    session = ort.InferenceSession('modelo_aire.onnx')
    scaler = joblib.load('escalador_aire.pkl')
    
    # Cargar las 24 horas automáticas
    with open('semilla_historial.json', 'r') as f:
        HISTORIAL_SEMILLA = json.load(f)
        
    print("¡Todos los recursos cargados correctamente!")
except Exception as e:
    print(f"Error crítico al cargar componentes: {e}")

# ¡AHORA EL ESQUEMA SOLO PIDE LA FECHA!
class PredictRequest(BaseModel):
    fecha_objetivo: str

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

@app.post("/predict")
def predecir_calidad(payload: PredictRequest):
    # Usar el historial oculto que cargamos del archivo JSON
    entrada_bloque = np.array(HISTORIAL_SEMILLA)
    entrada_listo = np.expand_dims(entrada_bloque, axis=0).astype(np.float32)
    
    # Inferencia ONNX
    inputs = {session.get_inputs()[0].name: entrada_listo}
    prediccion_escalada = session.run(None, inputs)[0]
    
    prediccion_real = scaler.inverse_transform(prediccion_escalada)[0]
    
    no2, o3, so2, pm25, co, pm10, tmp, dir_viento, rs, pre, iuv, vel, hum, llu, hora, mes, dia_semana = prediccion_real
    
    estado_pm25 = evaluar_pm25(pm25)
    estado_general = "MALA" if "Dañino" in estado_pm25 or pm25 > 35 else "BUENA / NORMAL"
    
    try:
        mes_objetivo = datetime.strptime(payload.fecha_objetivo, "%Y-%m-%d").month
    except:
        mes_objetivo = 12
        
    lluvia_txt = "Alta probabilidad (típico de la época)" if llu > 3.0 and mes_objetivo in [10,11,12,1,2,3,4,5] else "Baja probabilidad"

    return {
        "fecha_objetivo": payload.fecha_objetivo,
        "contexto": "Proyección Estacional a Largo Plazo (Quito)",
        "estado_general_aire": estado_general,
        "contaminantes_proyectados": {
            "pm25": f"{pm25:.1f} ug/m3 (Nivel: {estado_pm25})",
            "pm10": f"{pm10:.1f} ug/m3 (Nivel: {evaluar_pm10(pm10)})",
            "no2": f"{no2:.1f} ppb",
            "o3": f"{o3:.1f} ppb",
            "so2": f"{so2:.1f} ppb",
            "co": f"{co:.1f} ppm"
        },
        "meteorologia_proyectada": {
            "temperatura_tmp": f"{tmp:.1f} °C",
            "humedad_hum": f"{hum:.0f}%",
            "precipitacion_llu": lluvia_txt,
            "viento": f"{vel:.1f} km/h / {obtener_direccion_viento(dir_viento)}",
            "radiacion_solar_rs": f"{rs:.1f} W/m²",
            "indice_uv_iuv": f"{iuv:.0f} (Proyección)",
            "presion_pre": f"{pre:.0f} hPa"
        }
    }
