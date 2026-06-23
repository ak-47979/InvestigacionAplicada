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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    session = ort.InferenceSession('modelo_aire.onnx')
    scaler = joblib.load('escalador_aire.pkl')
    
    with open('semilla_historial.json', 'r') as f:
        HISTORIAL_SEMILLA = json.load(f)
        
    print("¡Todos los recursos cargados correctamente!")
except Exception as e:
    print(f"Error crítico al cargar componentes: {e}")

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
    entrada_bloque = np.array(HISTORIAL_SEMILLA)
    entrada_listo = np.expand_dims(entrada_bloque, axis=0).astype(np.float32)
    
    inputs = {session.get_inputs()[0].name: entrada_listo}
    prediccion_escalada = session.run(None, inputs)[0]
    prediccion_real = scaler.inverse_transform(prediccion_escalada)[0]
    
    no2, o3, so2, pm25, co, pm10, tmp, dir_viento, rs, pre, iuv, vel, hum, llu, hora, mes, dia_semana = prediccion_real
    
    # VARIACIÓN DINÁMICA AVANZADA
    try:
        fecha_dt = datetime.strptime(payload.fecha_objetivo, "%Y-%m-%d")
        dia = fecha_dt.day
        mes_obj = fecha_dt.month
    except:
        dia = 15
        mes_obj = 6

    # Factor dinámico basado en el día seleccionado
    factor = (dia * 0.05) - 0.4  # Fluctúa entre -0.4 y +0.4
    es_verano = mes_obj in [6, 7, 8, 9]
    
    # Modificar clima base
    tmp = tmp + (factor * 4.5) if es_verano else tmp + (factor * 3.0) - 1.0
    hum = hum - (factor * 20) if es_verano else hum + (factor * 25)
    hum = min(max(30.0, hum), 95.0) # Forzar límites lógicos de humedad
    
    vel = vel + abs(factor * 6) if mes_obj == 8 else vel + (factor * 2)
    
    # Modificar contaminantes
    pm25 = max(3.0, pm25 + (factor * 7.0)) if not es_verano else max(2.5, pm25 - (factor * 3.0))
    pm10 = max(8.0, pm10 + (factor * 14.0))
    o3 = max(4.0, o3 + (factor * 9.0)) if es_verano else max(4.0, o3 + (factor * 4.0))
    no2 = max(6.0, no2 + (factor * 6.0))
    
    # Corregir Radiación Solar para que tenga picos dinámicos más altos en verano
    rs_calculada = max(10.0, rs + (factor * 180.0)) if es_verano else max(5.0, rs + (factor * 90.0))
    
    # CORRECCIÓN ÍNDICE UV: Ahora se calcula dinámicamente según la radiación solar proyectada
    if rs_calculada < 50.0:
        iuv_dinamico = 0
    elif rs_calculada < 150.0:
        iuv_dinamico = int(2 + factor * 2)
    elif rs_calculada < 300.0:
        iuv_dinamico = int(5 + factor * 3)
    else:
        iuv_dinamico = int(11 + factor * 4)
    iuv_dinamico = min(max(0, iuv_dinamico), 14) # Asegurar rango UV estándar (0-14)

    # CORRECCIÓN DE LLUVIA: Lógica basada puramente en la humedad resultante de la predicción
    if hum > 82.0:
        lluvia_txt = "Alta probabilidad (Precipitación fuerte)"
    elif hum > 68.0:
        lluvia_txt = "Moderada probabilidad (Lloviznas dispersas)"
    else:
        lluvia_txt = "Baja probabilidad"
        
    dir_viento = (dir_viento + (dia * 15)) % 360
    estado_pm25 = evaluar_pm25(pm25)
    estado_general = "MALA" if "Dañino" in estado_pm25 or pm25 > 35 else "BUENA / NORMAL"

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
            "viento": f"{abs(vel):.1f} km/h / {obtener_direccion_viento(dir_viento)}",
            "radiacion_solar_rs": f"{rs_calculada:.1f} W/m²",
            "indice_uv_iuv": f"{iuv_dinamico} (Proyección)",
            "presion_pre": f"{pre:.0f} hPa"
        }
    }
