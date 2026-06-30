from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import joblib
import json
from datetime import datetime, timedelta

app = FastAPI(title="Predicción de Calidad del Aire - Quito Centro Histórico")

# Configuración de CORS obligatoria para conectar tu index.html
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Matriz base real del Centro Histórico (24 horas x 17 variables)
MATRIZ_BASE_CENTRO = np.array([
    [0.20977061810541064, 0.09834545713878191, 0.2571556350626118, 0.02342695940144732, 0.10453400503778337, 0.16696932515337426, 0.3719951923076923, 0.11878629581260941, 0.0, 0.5496264674493005, 0.0, 0.16387959866220736, 0.5733741554054054, 0.0, 0.0, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.18513764085009274, 0.17889087656529518, 0.10535999018766098, 0.06624685138539044, 0.10105521472392638, 0.37079326923076916, 0.11497957709300065, 0.0, 0.5037353255069377, 0.0, 0.3076923076923077, 0.5690456081081081, 0.0, 0.043478260869565216, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.07944658393952361, 0.16279069767441862, 0.16043174291671777, 0.08513853904282116, 0.11592638036809817, 0.32572115384615385, 0.16863486065186584, 0.0, 0.4781216648879507, 0.0, 0.21237458193979933, 0.6782094594594594, 0.0, 0.08695652173913043, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.05505634003708458, 0.2110912343470483, 0.10719980375321968, 0.08312342569269522, 0.27239263803680985, 0.3076923076923076, 0.226680374558893, 0.0, 0.462113127001075, 0.0, 0.17892976588628765, 0.6884501689189189, 0.0, 0.13043478260869565, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.02160890029952931, 0.4136851520572451, 0.11639887158101313, 0.0743073047858942, 0.2396564417177914, 0.2776442307692307, 0.2521048098029953, 0.0, 0.4866595517609511, 0.0, 0.16555183946488294, 0.7279349662162161, 0.0, 0.17391304347826086, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.021965482812722866, 0.4396243291592129, 0.14865693609714214, 0.12241813602015113, 0.3008588957055215, 0.2998798076923077, 0.19972769457334186, 0.0, 0.5112059765208272, 0.0, 0.12876254180602006, 0.7037584459459459, 0.0, 0.21739130434782608, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.03580088432463272, 0.4651162790697675, 0.16055439715442169, 0.2327455919395466, 0.1560245398773006, 0.27283653846153844, 0.454416627302787, 0.024805714734280556, 0.5699039487726907, 0.004053271569195137, 0.06688963210702341, 0.7263513513513514, 0.0, 0.2608695652173913, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.11560405077734989, 0.4499105545617174, 0.11517232920397401, 0.20453400503778338, 0.26385276073619635, 0.4879807692307693, 0.07438383950651588, 0.2143810346181019, 0.5976520811099277, 0.042848870874348584, 0.04849498327759197, 0.5040118243243243, 0.0, 0.30434782608695654, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.21901297960348026, 0.23658318425760289, 0.08573531215503497, 0.15793450881612092, 0.09683435582822086, 0.6592548076923075, 0.3684181277612603, 0.40984378679645184, 0.626467449306304, 0.16386797915460335, 0.10367892976588629, 0.3815456081081081, 0.0, 0.34782608695652173, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.3023819711881329, 0.13506261180679785, 0.1955108549000368, 0.0743073047858942, 0.2308711656441718, 0.6586538461538463, 0.2794187112728889, 0.5651935002747468, 0.647812166488805, 0.37637521713954836, 0.33277591973244147, 0.3628589527027027, 0.0, 0.3913043478260869, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.3180002852660106, 0.1650268336314848, 0.040230589966883355, 0.063727959697733, 0.28584049079754603, 0.6850961538461537, 0.2527161076995748, 0.740324986262658, 0.6296691568836792, 0.6189924724956573, 0.33277591973244147, 0.37225506756756754, 0.0, 0.43478260869565216, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.351661674511482, 0.06216457960644007, 0.1480436649086226, 0.06272040302267003, 0.21855214723926383, 0.7265625, 0.2143432715551974, 0.7224271920872909, 0.5837780149413021, 0.7527504342790967, 0.4765886287625418, 0.34121621621621623, 0.0, 0.4782608695652174, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.3598630723149337, 0.023703041144901613, 0.05126947136023549, 0.057430730478589424, 0.43695705521472394, 0.7770432692307692, 0.2586067965211592, 0.8801318784833975, 0.5250800426894386, 0.6027793862188767, 0.48327759197324416, 0.3132390202702703, 0.0, 0.5217391304347826, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.3749821708743403, 0.0353309481216458, 0.11811603090886791, 0.052141057934508815, 0.4107484662576687, 0.7800480769230769, 0.1876684542499097, 0.646911060522804, 0.4535752401280746, 0.31731325998841925, 0.4331103678929766, 0.31461148648648646, 0.0, 0.5652173913043478, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.3903152189416631, 0.03130590339892665, 0.130626763154667, 0.06070528967254408, 0.41865030674846626, 0.7427884615384615, 0.19853288504820915, 0.3140748881387864, 0.3767342582710853, 0.21598147075854082, 0.431438127090301, 0.3364653716216216, 0.0, 0.6086956521739131, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.3443160747396948, 0.04561717352415027, 0.11848399362197964, 0.07506297229219143, 0.15440490797546014, 0.6820913461538463, 0.10536552835588653, 0.23369181254415572, 0.3436499466382088, 0.15460335842501446, 0.4481605351170569, 0.41100084459459457, 0.0, 0.6521739130434783, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.35194694052203684, 0.03756708407871199, 0.058506071384766344, 0.07984886649874055, 0.30984049079754605, 0.6544471153846154, 0.06771513518019395, 0.3110919224428919, 0.35538954108858434, 0.04921829762594094, 0.5351170568561873, 0.4582981418918919, 0.0, 0.6956521739130435, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.2734987876194552, 0.06037567084078713, 0.15147798356433215, 0.08387909319899245, 0.09619631901840492, 0.5853365384615383, 0.041484898163327684, 0.06209278593296177, 0.3874066168623358, 0.019108280254777073, 0.45150501672240806, 0.47645692567567566, 0.0, 0.7391304347826086, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.19262587362715736, 0.06932021466905189, 0.1511100208512204, 0.11284634760705291, 0.14228220858895704, 0.5414663461538463, 0.09180582955903192, 0.005573435905487086, 0.4514407684098245, 0.0005790387955993052, 0.30267558528428096, 0.4921875, 0.0, 0.7826086956521738, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.10890029952931109, 0.0885509838998211, 0.09787808168772232, 0.14609571788413098, 0.28505521472392636, 0.485576923076923, 0.18380616299424823, 0.0, 0.5400213447171893, 0.0, 0.3494983277591973, 0.5842483108108109, 0.0, 0.8260869565217391, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.11624589930109828, 0.08810375670840788, 0.13982583098246046, 0.12418136020151134, 0.12132515337423314, 0.46814903846153844, 0.16977409764094584, 0.0, 0.6296691568836792, 0.0, 0.38294314381270905, 0.5894214527027026, 0.0, 0.8695652173913043, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.19669091427756383, 0.1319320214669052, 0.1200784987121305, 0.07909319899244333, 0.09011042944785276, 0.43689903846153844, 0.10350384839812166, 0.0, 0.6894343649946677, 0.0, 0.4665551839464883, 0.593433277027027, 0.0, 0.9130434782608695, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.1977606618171445, 0.16771019677996424, 0.14485465472832088, 0.07027707808564232, 0.26336196319018407, 0.4188701923076923, 0.07885742865875685, 0.0, 0.7075773745997935, 0.0, 0.41471571906354515, 0.5935388513513513, 0.0, 0.9565217391304348, 0.9999999999999999, 0.3333333333333333],
    [0.20977061810541064, 0.16374269005847955, 0.19767441860465115, 0.11860664785968356, 0.09244332493702771, 0.07631901840490798, 0.3966346153846153, 0.03734474422740282, 0.0, 0.6851654215581817, 0.0, 0.14046822742474915, 0.5933277027027027, 0.0, 1.0, 0.9999999999999999, 0.3333333333333333]
], dtype=np.float32)

try:
    session = ort.InferenceSession('modelo_aire.onnx')
    scaler = joblib.load('escalador_aire.pkl')
    print("¡Recursos cargados con éxito!")
except Exception as e:
    print(f"Error crítico al cargar datos: {e}")

def generar_entrada_dinamica_del_dia() -> np.ndarray:
    hoy = datetime.now()
    seed_del_dia = int(hoy.strftime("%Y%m%d"))
    rng = np.random.default_rng(seed_del_dia)
    ruido = rng.uniform(-0.015, 0.015, size=MATRIZ_BASE_CENTRO.shape)
    matriz_dinamica = np.clip(MATRIZ_BASE_CENTRO + ruido, 0.0, 1.0)
    matriz_dinamica[:, -1] = MATRIZ_BASE_CENTRO[:, -1]
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

# --- NUEVA FUNCIÓN PARA REDACTAR EL ANÁLISIS CIUDADANO ---
def generar_texto_explicativo(pm25, pm10, o3, no2, tmp, hum, uv) -> str:
    estado = evaluar_pm25(pm25)
    consejo = ""
    if estado == "Bueno":
        consejo = "El aire es ideal para actividades al aire libre y deportes en las plazas coloniales."
    elif estado == "Normal":
        consejo = "La calidad del aire es aceptable; sin embargo, personas extremadamente sensibles podrían experimentar síntomas leves."
    elif estado == "Dañino para grupos sensibles":
        consejo = "Se recomienda que niños, adultos mayores y personas con problemas respiratorios (como asma) reduzcan los esfuerzos prolongados al aire libre."
    else:
        consejo = "¡Alerta ambiental! Se sugiere usar mascarilla en calles de alto flujo vehicular y evitar deportes al aire libre."

    texto = (
        f"Para la fecha consultada en el Centro Histórico de Quito, se registra un índice de material particulado "
        f"fino PM2.5 de {pm25} µg/m³, lo que clasifica la calidad del aire como '{estado}'. {consejo} "
        f"En el aspecto climático, se estima una temperatura de {tmp}°C con una humedad relativa del {hum}%. "
        f"El índice de Radiación Ultravioleta (UV) se situará en un nivel {uv}, por lo que el uso de protector solar es altamente recomendable."
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
    prediccion_escalada = session.run(None, inputs)[0]
    prediccion_real = scaler.inverse_transform(prediccion_escalada)[0]
    
    no2_b, o3_b, so2_b, pm25_b, co_b, pm10_b, tmp_b, dir_b, rs_b, pre_b, iuv_b, vel_b, hum_b, llu_b, _, _, _ = prediccion_real

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
    rs_calculada = max(10.0, rs_b + (factor * 180.0)) if es_verano else max(5.0, rs_b + (factor * 90.0))
    
    if rs_calculada < 50.0: iuv_dinamico = 0
    elif rs_calculada < 150.0: iuv_dinamico = int(2 + factor * 2)
    elif rs_calculada < 300.0: iuv_dinamico = int(5 + factor * 3)
    else: iuv_dinamico = int(11 + factor * 4)
    iuv_dinamico = min(max(0, iuv_dinamico), 14)

    lluvia_txt = "Alta" if hum > 82.0 else ("Moderada" if hum > 68.0 else "Baja probabilidad")
    dir_viento = (dir_b + (dia * 15)) % 360

    pm25_r = round(float(pm25), 1)
    pm10_r = round(float(pm10), 1)
    o3_r = round(float(o3), 1)
    no2_r = round(float(no2), 1)
    tmp_r = round(float(tmp), 1)
    hum_r = int(hum)

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

    # Acumuladores para promedios
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
        
        if rs_calculada < 50.0: iuv_dinamico = 0
        elif rs_calculada < 150.0: iuv_dinamico = int(2 + factor * 2)
        elif rs_calculada < 300.0: iuv_dinamico = int(5 + factor * 3)
        else: iuv_dinamico = int(11 + factor * 4)
        iuv_dinamico = min(max(0, iuv_dinamico), 14)

        lluvia_txt = "Alta" if hum > 82.0 else ("Moderada" if hum > 68.0 else "Baja probabilidad")
        dir_viento = (dir_b + (dia * 15)) % 360
        
        # Sumar a acumuladores
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

    # Cálculo de promedios del rango
    total_dias = len(resultados_rango)
    prom_pm25 = round(sum_pm25 / total_dias, 1)
    prom_pm10 = round(sum_pm10 / total_dias, 1)
    prom_no2 = round(sum_no2 / total_dias, 1)
    prom_o3 = round(sum_o3 / total_dias, 1)
    prom_tmp = round(sum_tmp / total_dias, 1)
    prom_hum = int(sum_hum / total_dias)
    prom_uv = round(sum_uv / total_dias, 1)

    # Redacción de la conclusión basada en el promedio general del rango
    estado_promedio = evaluar_pm25(prom_pm25)
    if estado_promedio in ["Bueno", "Normal"]:
        conclusion = (
            f"Conclusión general: Durante este rango de fechas, el Centro Histórico de Quito mantendrá un patrón atmosférico "
            f"saludable y estable, ideal para la afluencia turística habitual. El promedio general de contaminación se conserva dentro de los límites admisibles."
        )
    else:
        conclusion = (
            f"Conclusión general: El periodo evaluado muestra tendencias críticas de acumulación de contaminantes, "
            f"con promedios que superan la norma técnica recomendada. Esto es propicio de días secos o con alta congestión vehicular en el casco colonial, "
            f"ameritando atención preventiva en grupos vulnerables."
        )

    return {
        "rango_dias_procesados": total_dias,
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
