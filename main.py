from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime

# Inicializar FastAPI
app = FastAPI(title="API Calidad del Aire - Quito (Producción)")

# Cargar la red neuronal y el escalador exportados desde Colab
try:
    modelo = tf.keras.models.load_model('modelo_aire.h5')
    scaler = joblib.load('escalador_aire.pkl')
    print("¡Modelo y escalador cargados correctamente!")
except Exception as e:
    print(f"Error crítico al cargar los archivos: {e}")

# Definir la estructura estricta del JSON que debe enviar el cliente (App/Web)
class InputData(BaseModel):
    fecha_objetivo: str            # Formato: "YYYY/MM/DD"
    historial: List[List[float]]    # Matriz obligatoria de 24 filas x 17 columnas

# Lógica de categorización bajo estándares de la OMS
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

# ENDPOINT PRINCIPAL: Procesa la petición POST de la aplicación móvil
@app.post("/predict")
def predecir_calidad(payload: InputData):
    # 1. Convertir el historial a matriz NumPy y aplicar la misma escala de Colab
    entrada_bloque = np.array(payload.historial)
    entrada_escalada = scaler.transform(entrada_bloque)
    
    # Reestructurar dimensiones para que la capa LSTM lo acepte: (1, 24, 17)
    entrada_listo = np.expand_dims(entrada_escalada, axis=0)
    
    # 2. Ejecutar la predicción matemática en la red neuronal
    prediccion_escalada = modelo.predict(entrada_listo)
    
    # 3. Deshacer la escala (0-1) para recuperar las unidades de medida originales del Excel
    prediccion_real = scaler.inverse_transform(prediccion_escalada)[0]
    
    # Desempaquetado siguiendo EL ORDEN EXACTO de las columnas de tu dataset
    no2, o3, so2, pm25, co, pm10, tmp, dir_viento, rs, pre, iuv, vel, hum, llu, hora, mes, dia_semana = prediccion_real
    
    # 4. Procesamiento de alertas y lógicas del contexto de Quito
    estado_pm25 = evaluar_pm25(pm25)
    estado_general = "MALA" if "Dañino" in estado_pm25 or pm25 > 35 else "BUENA / NORMAL"
    
    # Extraer el mes de la fecha solicitada para evaluar estacionalidad de lluvias
    try:
        mes_objetivo = datetime.strptime(payload.fecha_objetivo, "%Y/%m/%d").month
    except:
        mes_objetivo = 12  # Mes por defecto si el formato falla
        
    lluvia_txt = "Alta probabilidad (típico de la época)" if llu > 3.0 and mes_objetivo in [10,11,12,1,2,3,4,5] else "Baja probabilidad"

    # 5. Retornar el desglose estructurado exacto solicitado
    return {
        "fecha_objetivo": payload.fecha_objetivo,
        "contexto": "Proyección Estacional a Largo Plazo (Quito)",
        "estado_general_aire": estado_general,
        "detalles_estado": "Condicionado por proyecciones de Material Particulado",
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