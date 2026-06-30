from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import joblib
from datetime import datetime, timedelta

app = FastAPI(title="Predicción de Calidad del Aire y Clima - Quito Centro Histórico")

# Configuración de CORS obligatoria para conectar tu index.html
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Matriz base real del Centro Histórico (24 horas x 17 variables)
# Importante: Esta matriz ya debe estar escalada (entre 0 y 1) ya que sirve de look-back para el modelo
MATRIZ_BASE_CENTRO = np.array([
    [0.2097706, 0.0983454, 0.2571556, 0.0234269, 0.1045340, 0.1669693, 0.3719951, 0.1187862, 0.0, 0.5496264, 0.0, 0.1638795, 0.5733741, 0.0, 0.0, 1.0, 0.3333333],
    [0.2097706, 0.1851376, 0.1788908, 0.1053600, 0.0662468, 0.1010552, 0.3707932, 0.1149795, 0.0, 0.5037353, 0.0, 0.3076923, 0.5690456, 0.0, 0.0434782, 1.0, 0.3333333],
    [0.2097706, 0.0794465, 0.1627907, 0.1604317, 0.0851385, 0.1159263, 0.3257211, 0.1686349, 0.0, 0.4781216, 0.0, 0.2123746, 0.6782094, 0.0, 0.0869565, 1.0, 0.3333333],
    [0.2097706, 0.0550563, 0.2110912, 0.1071998, 0.0831234, 0.2723926, 0.3076923, 0.2266804, 0.0, 0.4621131, 0.0, 0.1789298, 0.6884501, 0.0, 0.1304348, 1.0, 0.3333333],
    [0.2097706, 0.0216089, 0.4136852, 0.1163989, 0.0743073, 0.2396564, 0.2776442, 0.2521048, 0.0, 0.4866596, 0.0, 0.1655518, 0.7279350, 0.0, 0.1739130, 1.0, 0.3333333],
    [0.2097706, 0.0219655, 0.4396243, 0.1486569, 0.1224181, 0.3008589, 0.2998798, 0.1997277, 0.0, 0.5112060, 0.0, 0.1287625, 0.7037584, 0.0, 0.2173913, 1.0, 0.3333333],
    [0.2097706, 0.0358009, 0.4651163, 0.1605544, 0.2327456, 0.1560245, 0.2728365, 0.4544166, 0.0248057, 0.5699039, 0.0040533, 0.0668896, 0.7263514, 0.0, 0.2608696, 1.0, 0.3333333],
    [0.2097706, 0.1156041, 0.4499106, 0.1151723, 0.2045340, 0.2638528, 0.4879808, 0.0743838, 0.2143810, 0.5976521, 0.0428489, 0.0484950, 0.5040118, 0.0, 0.3043478, 1.0, 0.3333333],
    [0.2097706, 0.2190130, 0.2365832, 0.0857353, 0.1579345, 0.0968344, 0.6592548, 0.3684181, 0.4098438, 0.6264674, 0.1638680, 0.1036789, 0.3815456, 0.0, 0.3478261, 1.0, 0.3333333],
    [0.2097706, 0.3023820, 0.1350626, 0.1955109, 0.0743073, 0.2308712, 0.6586538, 0.2794187, 0.5651935, 0.6478122, 0.3763752, 0.3327759, 0.3628590, 0.0, 0.3913043, 1.0, 0.3333333],
    [0.2097706, 0.3180003, 0.1650268, 0.0402306, 0.0637280, 0.2858405, 0.6850962, 0.2527161, 0.7403250, 0.6296692, 0.6189925, 0.3327759, 0.3722551, 0.0, 0.4347826, 1.0, 0.3333333],
    [0.2097706, 0.3516617, 0.0621646, 0.1480437, 0.0627204, 0.2185521, 0.7265625, 0.2143433, 0.7224272, 0.5837780, 0.7527504, 0.4765886, 0.3412162, 0.0, 0.4782609, 1.0, 0.3333333],
    [0.2097706, 0.3598631, 0.0237030, 0.0512695, 0.0574307, 0.4369571, 0.7770433, 0.2586068, 0.8801319, 0.5250800, 0.6027794, 0.4832776, 0.3132390, 0.0, 0.5217391, 1.0, 0.3333333],
    [0.2097706, 0.3749822, 0.0353309, 0.1181160, 0.0521411, 0.4107485, 0.7800481, 0.1876685, 0.6469111, 0.4535752, 0.3173133, 0.4331104, 0.3146115, 0.0, 0.5652174, 1.0, 0.3333333],
    [0.2097706, 0.3903152, 0.0313059, 0.1306268, 0.0607053, 0.4186503, 0.7427885, 0.1985329, 0.3140749, 0.3767343, 0.2159815, 0.4314381, 0.3364654, 0.0, 0.6086957, 1.0, 0.3333333],
    [0.2097706, 0.3443161, 0.0456172, 0.1184840, 0.0750630, 0.1544049, 0.6820913, 0.1053655, 0.2336918, 0.3436500, 0.1546034, 0.4481605, 0.4110008, 0.0, 0.6521739, 1.0, 0.3333333],
    [0.2097706, 0.3519469, 0.0375671, 0.0585061, 0.0798489, 0.3098405, 0.6544471, 0.0677151, 0.3110919, 0.3553895, 0.0492183, 0.5351171, 0.4582981, 0.0, 0.6956522, 1.0, 0.3333333],
    [0.2097706, 0.2734988, 0.0603757, 0.1514780, 0.0838790, 0.0961963, 0.5853365, 0.0414849, 0.0620928, 0.3874066, 0.0191083, 0.4515050, 0.4764569, 0.0, 0.7391304, 1.0, 0.3333333],
    [0.2097706, 0.1926259, 0.0693202, 0.1511100, 0.1128463, 0.1422822, 0.5414663, 0.0918058, 0.0055734, 0.4514408, 0.0005790, 0.3026756, 0.4921875, 0.0, 0.7826087, 1.0, 0.3333333],
    [0.2097706, 0.1089003, 0.0885510, 0.0978781, 0.1460957, 0.2850552, 0.4855769, 0.1838062, 0.0, 0.5400213, 0.0, 0.3494983, 0.5842483, 0.0, 0.8260870, 1.0, 0.3333333],
    [0.2097706, 0.1162459, 0.0881038, 0.1398258, 0.1241814, 0.1213251, 0.4681490, 0.1697741, 0.0, 0.6296692, 0.0, 0.3829431, 0.5894215, 0.0, 0.8695652, 1.0, 0.3333333],
    [0.2097706, 0.1966909, 0.1319320, 0.1200785, 0.0790932, 0.0901104, 0.4368990, 0.1035038, 0.0, 0.6894344, 0.0, 0.4665552, 0.5934333, 0.0, 0.9130435, 1.0, 0.3333333],
    [0.2097706, 0.1977607, 0.1677102, 0.1448547, 0.0702771, 0.2633620, 0.4188702, 0.0788574, 0.0, 0.7075774, 0.0, 0.4147157, 0.5935389, 0.0, 0.9565217, 1.0, 0.3333333],
    [0.2097706, 0.1637427, 0.1976744, 0.1186066, 0.0924433, 0.0763190, 0.3966346, 0.0373447, 0.0, 0.6851654, 0.0, 0.1404682, 0.5933277, 0.0, 1.0, 1.0, 0.3333333]
], dtype=np.float32)

try:
    # Carga de recursos (Modelo entrenado con Sigmoid y Escalador guardado en Colab)
    session = ort.InferenceSession('modelo_aire.onnx')
    scaler = joblib.load('escalador_aire.pkl')
    print("¡Recursos ONNX y de escalamiento acoplados correctamente!")
except Exception as e:
    print(f"Error crítico al cargar componentes: {e}")

def generar_entrada_dinamica_del_dia() -> np.ndarray:
    hoy = datetime.now()
    seed_del_dia = int(hoy.strftime("%Y%m%d"))
    rng = np.random.default_rng(seed_del_dia)
    ruido = rng.uniform(-0.01, 0.01, size=MATRIZ_BASE_CENTRO.shape)
    matriz_dinamica = np.clip(MATRIZ_BASE_CENTRO + ruido, 0.0, 1.0)
    matriz_dinamica[:, -1] = MATRIZ_BASE_CENTRO[:, -1] # Mantener constante el día de la semana
    return matriz_dinamica

def evaluar_pm25(valor):
    if valor <= 15.0: return "Bueno"
    elif valor <= 35.0: return "Normal"
    elif valor <= 55.4: return "Dañino para grupos sensibles"
    else: return "Malo"

def obtener_direccion_viento(grados):
    direcciones = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    indice = int((grados / 22.5) + .5) % 16
    return direcciones[indice]

# --- NUEVA LOGICA DE TEXTO 100% DINÁMICA ---
def generar_texto_explicativo(pm25, pm10, o3, no2, tmp, hum, uv) -> str:
    estado = evaluar_pm25(pm25)
    
    if estado == "Bueno":
        consejo_aire = "El aire es ideal para actividades al aire libre y deportes en las plazas coloniales."
    elif estado == "Normal":
        consejo_aire = "La calidad del aire es aceptable; sin embargo, personas extremadamente sensibles podrían experimentar síntomas leves."
    elif estado == "Dañino para grupos sensibles":
        consejo_aire = "Se recomienda que niños, adultos mayores y personas con afecciones respiratorias crónicas reduzcan esfuerzos prolongados en el exterior."
    else:
        consejo_aire = "¡Alerta ambiental! Se sugiere usar mascarilla en calles de alto flujo vehicular y suspender deportes al aire libre."

    # Diagnóstico dinámico de radiación calibrado para la altitud de Quito
    if uv >= 11:
        consejo_uv = f"El índice UV se sitúa en un nivel {uv} (Extremadamente Alto). Es imperativo evitar la exposición directa y usar protección total."
    elif uv >= 6:
        consejo_uv = f"El índice UV registrará un nivel {uv} (Muy Alto). El uso de protector solar, sombrero y gafas es altamente recomendable."
    elif uv >= 3:
        consejo_uv = f"El índice UV estará en un rango de {uv} (Moderado). Se aconseja usar protector solar si pasa períodos prolongados bajo el sol."
    else:
        consejo_uv = f"El índice UV se mantendrá bajo ({uv}), requiriendo precauciones estándar mínimas."

    texto = (
        f"Para la fecha consultada en el Centro Histórico de Quito, se registra un índice de material particulado "
        f"fino PM2.5 de {pm25} µg/m³, lo que clasifica la calidad del aire como '{estado}'. {consejo_aire} "
        f"En el aspecto climático, se estima una temperatura promedio de {tmp}°C con una humedad relativa del {hum}%. "
        f"{consejo_uv}"
    )
    return texto

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "online", "message": "API de Calidad del Aire - Centro Histórico de Quito lista."}

@app.get("/predict")
def predecir_individual(fecha: str = Query(..., description="Fecha YYYY-MM-DD")):
    try:
        target_dt = datetime.strptime(fecha, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato inválido. Use YYYY-MM-DD")
    
    entrada_bloque = generar_entrada_dinamica_del_dia()
    entrada_listo = np.expand_dims(entrada_bloque, axis=0).astype(np.float32)
    inputs = {session.get_inputs()[0].name: entrada_listo}
    
    # El modelo ONNX con Sigmoid devuelve valores estrictamente acotados entre 0 y 1
    prediccion_escalada = session.run(None, inputs)[0]
    
    # Recuperamos los valores físicos reales gracias al escalador guardado
    prediccion_real = scaler.inverse_transform(prediccion_escalada)[0]
    
    no2_b, o3_b, so2_b, pm25_b, co_b, pm10_b, tmp_b, dir_b, rs_b, pre_b, iuv_b, vel_b, hum_b, llu_b, _, _, _ = prediccion_real

    # Ajustes estacionales de simulación según el día consultado
    dia = target_dt.day
    mes_obj = target_dt.month
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
    
    # Calibración de Radiación Solar e Índice UV acoplado para Quito (2800m de altura)
    rs_calculada = max(10.0, rs_b + (factor * 180.0)) if es_verano else max(5.0, rs_b + (factor * 90.0))
    
    if rs_calculada < 40.0: iuv_dinamico = 0
    elif rs_calculada < 140.0: iuv_dinamico = int(1 + (rs_calculada / 70))
    elif rs_calculada < 380.0: iuv_dinamico = int(3 + (rs_calculada - 140) / 80)
    elif rs_calculada < 720.0: iuv_dinamico = int(6 + (rs_calculada - 380) / 85)
    else: iuv_dinamico = int(11 + (rs_calculada - 720) / 90)
    
    iuv_dinamico = min(max(0, iuv_dinamico), 15)
    lluvia_txt = "Alta" if hum > 82.0 else ("Moderada" if hum > 68.0 else "Baja probabilidad")
    dir_viento = (dir_b + (dia * 15)) % 360

    pm25_r = round(float(pm25), 1)
    pm10_r = round(float(pm10), 1)
    o3_r = round(float(o3), 1)
    no2_r = round(float(no2), 1)
    tmp_r = round(float(tmp), 1)
    hum_r = int(hum)

    # Generación del párrafo dinámico explicativo
    analisis_ciudadano = generar_texto_explicativo(pm25_r, pm10_r, o3_r, no2_r, tmp_r, hum_r, iuv_dinamico)

    return {
        "fecha": target_dt.strftime("%d/%m/%Y"),
        "estado_general": "Moderado/Malo" if "Dañino" in evaluar_pm25(pm25) or pm25 > 35 else "Bueno",
        "pm25": pm25_r, "pm10": pm10_r, "no2": no2_r, "o3": o3_r, "so2": round(float(so2), 1), "co": round(float(co), 2),
        "temperatura": tmp_r, "humedad": hum_r, "lluvia": lluvia_txt,
        "viento": f"{abs(vel):.1f} km/h ({obtener_direccion_viento(dir_viento)})",
        "radiacion": round(float(rs_calculada), 1), "uv": int(iuv_dinamico), "presion": int(pre_b),
        "analisis_texto": analisis_ciudadano
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

    dias_solicitados = (end_dt - start_dt).days + 1
    if dias_solicitados > 31:
        end_dt = start_dt + timedelta(days=30)

    entrada_bloque = generar_entrada_dinamica_del_dia()
    entrada_listo = np.expand_dims(entrada_bloque, axis=0).astype(np.float32)
    inputs = {session.get_inputs()[0].name: entrada_listo}
    prediccion_escalada = session.run(None, inputs)[0]
    prediccion_real = scaler.inverse_transform(prediccion_escalada)[0]
    
    no2_b, o3_b, so2_b, pm25_b, co_b, pm10_b, tmp_b, dir_b, rs_b, pre_b, iuv_b, vel_b, hum_b, llu_b, _, _, _ = prediccion_real

    resultados_rango = []
    current_dt = start_dt

    sum_pm25, sum_pm10, sum_no2, sum_o3, sum_tmp, sum_hum, sum_uv = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

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
        
        if rs_calculada < 40.0: iuv_dinamico = 0
        elif rs_calculada < 140.0: iuv_dinamico = int(1 + (rs_calculada / 70))
        elif rs_calculada < 380.0: iuv_dinamico = int(3 + (rs_calculada - 140) / 80)
        elif rs_calculada < 720.0: iuv_dinamico = int(6 + (rs_calculada - 380) / 85)
        else: iuv_dinamico = int(11 + (rs_calculada - 720) / 90)
        iuv_dinamico = min(max(0, iuv_dinamico), 15)

        lluvia_txt = "Alta" if hum > 82.0 else ("Moderada" if hum > 68.0 else "Baja probabilidad")
        dir_viento = (dir_b + (dia * 15)) % 360
        
        sum_pm25 += pm25
        sum_pm10 += pm10
        sum_no2 += no2
        sum_o3 += o3
        sum_tmp += tmp
        sum_hum += hum
        sum_uv += iuv_dinamico

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

    total_dias = len(resultados_rango)
    prom_pm25 = round(sum_pm25 / total_dias, 1)
    prom_pm10 = round(sum_pm10 / total_dias, 1)
    prom_no2 = round(sum_no2 / total_dias, 1)
    prom_o3 = round(sum_o3 / total_dias, 1)
    prom_tmp = round(sum_tmp / total_dias, 1)
    prom_hum = int(sum_hum / total_dias)
    prom_uv = round(sum_uv / total_dias, 1)

    estado_promedio = evaluar_pm25(prom_pm25)
    if estado_promedio in ["Bueno", "Normal"]:
        conclusion = (
            f"Conclusión general: Durante este rango de fechas, el Centro Histórico de Quito mantendrá un patrón atmosférico "
            f"saludable y estable con un promedio de PM2.5 de {prom_pm25} µg/m³. Ideal para la afluencia turística habitual y actividades al aire libre."
        )
    else:
        conclusion = (
            f"Conclusión general: El periodo evaluado muestra tendencias críticas de acumulación de material particulado fino, "
            f"registrando un promedio de {prom_pm25} µg/m³ que califica como '{estado_promedio}'. Se aconseja tomar precauciones en el casco colonial, especialmente en niños y adultos mayores."
        )

    return {
        "rango_dias_processed": total_dias,
        "limite_aplicado": dias_solicitados > 31,
        "resumen_rango": {
            "promedio_pm25": prom_pm25,
            "promedio_pm10": prom_pm10,
            "promedio_no2": prom_no2,
            "promedio_o3": prom_o3,
            "promedio_temperatura": prom_tmp,
            "promedio_humedad": prom_hum,
            "promedio_indice_uv": prom_uv,
            "estado_promedio_periodo": estado_promedio,
            "conclusion_texto": conclusion
        },
        "datos": resultados_rango
    }
