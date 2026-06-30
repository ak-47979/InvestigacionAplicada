from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import joblib
import json
from datetime import datetime, timedelta

app = FastAPI(title="Predicción de Calidad del Aire - Quito")

# Configuración de CORS obligatoria para conectar tu index.html
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga de recursos reales del proyecto
try:
    session = ort.InferenceSession('modelo_aire.onnx')
    scaler = joblib.load('escalador_aire.pkl')
    with open('semilla_historial.json', 'r') as f:
        HISTORIAL_SEMILLA = json.load(f)
    print("¡Recursos cargados con éxito!")
except Exception as e:
    print(f"Error crítico al cargar datos: {e}")

# --- FUNCIONES DE EVALUACIÓN ORIGINALES ---
def evaluar_pm25(valor):
    if valor <= 15.0: return "Bueno"
    elif valor <= 35.0: return "Normal"
    elif valor <= 55.4: return "Dañino para grupos sensibles"
    else: return "Malo"

def obtener_direccion_viento(grados):
    direcciones = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    indice = int((grados / 22.5) + .5) % 16
    return direcciones[indice]

# --- ENDPOINTS ADAPTADOS AL FRONTEND ---

@app.get("/")
def read_root():
    return {"status": "online", "message": "API de Investigación Aplicada activa"}

@app.get("/predict")
def predecir_individual(fecha: str = Query(..., description="Fecha YYYY-MM-DD")):
    try:
        target_dt = datetime.strptime(fecha, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato inválido. Use YYYY-MM-DD")
    
    # Reutilizamos tu algoritmo de inferencia para un solo día
    entrada_bloque = np.array(HISTORIAL_SEMILLA)
    entrada_listo = np.expand_dims(entrada_bloque, axis=0).astype(np.float32)
    inputs = {session.get_inputs()[0].name: entrada_listo}
    prediccion_escalada = session.run(None, inputs)[0]
    prediccion_real = scaler.inverse_transform(prediccion_escalada)[0]
    
    no2_b, o3_b, so2_b, pm25_b, co_b, pm10_b, tmp_b, dir_b, rs_b, pre_b, iuv_b, vel_b, hum_b, llu_b, _, _, _ = prediccion_real

    dia = target_dt.day
    mes_obj = target_dt.month
    factor = (dia * 0.05) - 0.4
    es_verano = mes_obj in [6, 7, 8, 9]

    # Tus fórmulas matemáticas de alteración dinámica
    tmp = tmp_b + (factor * 4.5) if es_verano else tmp_b + (factor * 3.0) - 1.0
    hum = min(max(30.0, hum_b - (factor * 20) if es_verano else hum_b + (factor * 25)), 95.0)
    vel = vel_b + abs(factor * 6) if mes_obj == 8 else vel_b + (factor * 2)
    pm25 = max(3.0, pm25_b + (factor * 7.0)) if not es_verano else max(2.5, pm25_b - (factor * 3.0))
    pm10 = max(8.0, pm10_b + (factor * 14.0))
    o3 = max(4.0, o3_b + (factor * 9.0)) if es_verano else max(4.0, o3_b + (factor * 4.0))
    no2 = max(6.0, no2_b + (factor * 6.0))
    so2 = max(1.5, so2_b + (factor * 1.5))
    co = max(0.1, co_b + (factor * 0.2))
    rs_calculada = max(10.0, rs_b + (factor * 180.0)) if es_verano else max(5.0, rs_b + (factor * 90.0))
    
    if rs_calculada < 50.0: iuv_dinamico = 0
    elif rs_calculada < 150.0: iuv_dinamico = int(2 + factor * 2)
    elif rs_calculada < 300.0: iuv_dinamico = int(5 + factor * 3)
    else: iuv_dinamico = int(11 + factor * 4)
    iuv_dinamico = min(max(0, iuv_dinamico), 14)

    lluvia_txt = "Alta" if hum > 82.0 else ("Moderada" if hum > 68.0 else "Baja probabilidad")
    dir_viento = (dir_b + (dia * 15)) % 360

    return {
        "fecha": target_dt.strftime("%d/%m/%Y"),
        "estado_general": "Moderado/Malo" if "Dañino" in evaluar_pm25(pm25) or pm25 > 35 else "Bueno",
        "pm25": round(float(pm25), 1), "pm10": round(float(pm10), 1), "no2": round(float(no2), 1),
        "o3": round(float(o3), 1), "so2": round(float(so2), 1), "co": round(float(co), 2),
        "temperatura": round(float(tmp), 1), "humedad": int(hum), "lluvia": lluvia_txt,
        "viento": f"{abs(vel):.1f} km/h ({obtener_direccion_viento(dir_viento)})",
        "radiacion": round(float(rs_calculada), 1), "uv": int(iuv_dinamico), "presion": int(pre_b)
    }

@app.get("/predict_range")
def predecir_rango(
    start_date: str = Query(..., description="Fecha inicio YYYY-MM-DD"),
    end_date: str = Query(..., description="Fecha fin YYYY-MM-DD")
):
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt
    except ValueError:
        raise HTTPException(status_code=400, detail="Fechas inválidas. Use YYYY-MM-DD")

    # Limitar la consulta a un máximo de 31 días por estabilidad de la interfaz
    dias_solicitados = (end_dt - start_dt).days + 1
    if dias_solicitados > 31:
        end_dt = start_dt + timedelta(days=30)

    # Inferencia base ONNX
    entrada_bloque = np.array(HISTORIAL_SEMILLA)
    entrada_listo = np.expand_dims(entrada_bloque, axis=0).astype(np.float32)
    inputs = {session.get_inputs()[0].name: entrada_listo}
    prediccion_escalada = session.run(None, inputs)[0]
    prediccion_real = scaler.inverse_transform(prediccion_escalada)[0]
    
    no2_b, o3_b, so2_b, pm25_b, co_b, pm10_b, tmp_b, dir_b, rs_b, pre_b, iuv_b, vel_b, hum_b, llu_b, _, _, _ = prediccion_real

    resultados_rango = []
    current_dt = start_dt

    while current_dt <= end_dt:
        dia = current_dt.day
        mes_obj = current_dt.month
        factor = (dia * 0.05) - 0.4
        es_verano = mes_obj in [6, 7, 8, 9]

        tmp = tmp_b + (factor * 4.5) if es_verano else tmp_b + (factor * 3.0) - 1.0
        hum = min(max(30.0, hum_b - (factor * 20) if es_verano else hum_b + (factor * 25)), 95.0)
        vel = vel_b + abs(factor * 6) if mes_obj == 8 else vel_b + (factor * 2)
        
        pm25 = max(3.0, pm25_b + (factor * 7.0)) if not es_verano else max(2.5, pm25_b - (factor * 3.0))
        pm10 = max(8.0, pm10_b + (factor * 14.0))
        o3 = max(4.0, o3_b + (factor * 9.0)) if es_verano else max(4.0, o3_b + (factor * 4.0))
        no2 = max(6.0, no2_b + (factor * 6.0))
        so2 = max(1.5, so2_b + (factor * 1.5))
        co = max(0.1, co_b + (factor * 0.2))
        rs_calculada = max(10.0, rs_b + (factor * 180.0)) if es_verano else max(5.0, rs_b + (factor * 90.0))
        
        if rs_calculada < 50.0: iuv_dinamico = 0
        elif rs_calculada < 150.0: iuv_dinamico = int(2 + factor * 2)
        elif rs_calculada < 300.0: iuv_dinamico = int(5 + factor * 3)
        else: iuv_dinamico = int(11 + factor * 4)
        iuv_dinamico = min(max(0, iuv_dinamico), 14)

        lluvia_txt = "Alta" if hum > 82.0 else ("Moderada" if hum > 68.0 else "Baja probabilidad")
        dir_viento = (dir_b + (dia * 15)) % 360
        
        # Mapeo limpio y compatible con los bucles del index.html
        resultados_rango.append({
            "fecha": current_dt.strftime("%d/%m/%Y"),
            "estado_general": "Malo" if "Dañino" in evaluar_pm25(pm25) or pm25 > 35 else "Bueno",
            "pm25": round(float(pm25), 1), "pm10": round(float(pm10), 1), "no2": round(float(no2), 1),
            "o3": round(float(o3), 1), "so2": round(float(so2), 1), "co": round(float(co), 2),
            "temperatura": round(float(tmp), 1), "humedad": int(hum), "lluvia": lluvia_txt,
            "viento": f"{abs(vel):.1f} km/h ({obtener_direccion_viento(dir_viento)})",
            "radiacion": round(float(rs_calculada), 1), "uv": int(iuv_dinamico), "presion": int(pre_b)
        })
        
        current_dt += timedelta(days=1)

    return {
        "rango_dias_procesados": len(resultados_rango),
        "limite_aplicado": dias_solicitados > 31,
        "datos": resultados_rango
    }
