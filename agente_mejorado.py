"""
API Graiman – versión endurecida contra DoS
-------------------------------------------
•  Límite de tamaño de imagen: 10 MB
•  Lectura «chunked» + 413 si excede el límite
•  Pool CPU → tantos hilos como ½ núcleos (≥2)
•  Semáforo para no lanzar > N tareas LBP simultáneas
•  Middleware que rechaza peticiones grandes vía Content-Length
"""

from __future__ import annotations

# ──────────────────────────────── stdlib ────────────────────────────────
import asyncio, json, logging, time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import List

# ────────────────────────────── terceros ───────────────────────────────
import cx_Oracle, joblib, numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, conint
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse

# ────────────────────────────── locales ────────────────────────────────
from config import settings
from features import compute_lbp

# ───────────────────────── Logger ─────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)
logger = logging.getLogger("ceramica-api")

# ───────────────────────── Constantes ─────────────────────────
MAX_IMAGE_SIZE = 10 * 1024 * 1024                        # 10 MB
CPU_WORKERS    = max(2, cpu_count() // 2)                # ≥2 hilos
CHUNK_SIZE     = 8192                                    # 8 KB por lectura

# ───────────────────────── App ─────────────────────────
app = FastAPI(
    title="Agente Cerámicas Graiman",
    description="Clasificador de imágenes LBP + RandomForest",
    version="1.1",
)

# ──────────────────── Middleware Content-Length ────────────────────
class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit: int):
        super().__init__(app)
        self.limit = limit

    async def dispatch(self, request, call_next):
        cl = request.headers.get("content-length")
        if cl and int(cl) > self.limit:
            return PlainTextResponse("Archivo demasiado grande", status_code=413)
        return await call_next(request)


app.add_middleware(MaxBodySizeMiddleware, limit=MAX_IMAGE_SIZE)

# ───────────────────────── Globals ─────────────────────────
clf: joblib.BaseEstimator | None = None
classes: List[str] = []
executor        = ThreadPoolExecutor(max_workers=CPU_WORKERS)
lbp_semaphore   = asyncio.Semaphore(CPU_WORKERS)

# ──────────────────── Inicializar Oracle client ────────────────────
cx_Oracle.init_oracle_client(
    lib_dir=str(settings.oracle_lib_dir),
    config_dir=str(settings.oracle_config_dir),
)

# ──────────────────── Modelos Pydantic ────────────────────
class ProductItem(BaseModel):
    COD_PRODUCTO_NK2: str
    LINEA_NEGOCIO: str
    ESTADO_INVENTARIO: str
    CALIDAD: str
    MARCA: str
    FORMATO: str
    APLICACION: str
    COLOR: str
    COLECCION: str
    SUBMARCA: str


class ProductsResponse(BaseModel):
    resumen: str
    productos: list[ProductItem]

# ──────────────────── Lectura segura de UploadFile ────────────────────
async def safe_read_upload(upload: UploadFile, limit: int = MAX_IMAGE_SIZE) -> bytes:
    size = 0
    chunks: list[bytes] = []

    while True:
        chunk = await upload.read(CHUNK_SIZE)
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise HTTPException(
                status_code=413,
                detail=f"Imagen supera el límite de {limit//1024//1024} MB",
            )
        chunks.append(chunk)

    return b"".join(chunks)

# ──────────────────── Pool Oracle y OpenAI ────────────────────
@app.on_event("startup")
def startup_event():
    global clf, classes, pool, openai_client

    # Modelo
    if not settings.model_path.is_file():
        raise RuntimeError(f"Modelo no encontrado: {settings.model_path}")
    clf = joblib.load(settings.model_path)
    logger.info("✅ Modelo cargado")

    # Clases
    cf: Path = settings.classes_file
    if cf.is_file():
        with cf.open(encoding="utf-8") as f:
            classes[:] = sorted(json.load(f))
    else:
        root = settings.root_dir
        classes[:] = sorted(p.name for p in root.iterdir() if p.is_dir())
        cf.parent.mkdir(parents=True, exist_ok=True)
        with cf.open("w", encoding="utf-8") as f:
            json.dump(classes, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ {len(classes)} clases cargadas")

    # Oracle session pool
    pool = cx_Oracle.SessionPool(
        user=settings.oracle_user,
        password=settings.oracle_password,
        dsn=settings.oracle_dsn,
        min=2,
        max=10,
        increment=1,
        encoding="UTF-8",
    )
    pool.acquire().cursor().execute("SELECT 1 FROM DUAL").close()
    pool.release(pool.acquire())
    logger.info("✅ Pool Oracle OK")

    # OpenAI
    from openai import AsyncOpenAI  # import tardío para evitar coste si falla arriba

    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    logger.info("✅ Cliente OpenAI listo")


def get_db():
    conn = pool.acquire()
    try:
        conn.ping()
    except cx_Oracle.DatabaseError:
        pool.release(conn)
        conn = pool.acquire()
    try:
        yield conn
    finally:
        pool.release(conn)

# ──────────────────── Utils ────────────────────
async def extract_features(file: UploadFile) -> np.ndarray | None:
    """Lee imagen (limite 10 MB) y calcula LBP en pool de hilos."""
    content = await safe_read_upload(file, MAX_IMAGE_SIZE)
    async with lbp_semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, compute_lbp, content)

# ──────────────────── Endpoints ────────────────────
@app.post("/ceramicas", response_model=ProductsResponse)
async def recomendar_ceramicas(
    imagen: UploadFile | None = File(None),
    consulta: str | None = Form(None),
    topk: int = Form(3, gt=0, le=20),
    conn=Depends(get_db),
):
    if bool(imagen) == bool(consulta):
        raise HTTPException(400, "Envía una imagen O un texto, pero no ambos")

    if imagen:
        familia, productos = await _flujo_por_imagen(imagen, topk, conn)
        resumen = await _redactar_resumen(familia, productos)
        return {"resumen": resumen, "productos": productos}

    filtros = await _extraer_filtros_desde_texto(consulta)
    productos = _consultar_por_filtros(conn, filtros)
    resumen = await _redactar_resumen_desde_texto(consulta, productos)
    return {"resumen": resumen, "productos": productos}

# ──────────────────── Flujo imagen ────────────────────
async def _flujo_por_imagen(imagen: UploadFile, topk: int, conn):
    feats = await extract_features(imagen)
    if feats is None:
        raise HTTPException(422, "Imagen inválida o no soportada")

    probs = clf.predict_proba(feats.reshape(1, -1))[0]
    mejores = probs.argsort()[::-1][:topk]
    familia = classes[int(mejores[0])]
    productos = _consultar_por_familia(conn, familia)
    return familia, productos

# ──────────────────── LLM función JSON ────────────────────
FUNC_SCHEMA = {
    "name": "extract_filters",
    "description": "Convierte la solicitud en filtros SQL válidos",
    "parameters": {
        "type": "object",
        "properties": {
            "COLOR": {"type": "string"},
            "FORMATO": {"type": "string"},
            "CALIDAD": {"type": "string"},
            "APLICACION": {"type": "string"},
            "FAMILIA": {"type": "string"},
        },
        "required": [],
    },
}

async def _extraer_filtros_desde_texto(consulta: str) -> dict:
    system = (
        "Eres un experto de Graiman. Devuelve un JSON con filtros SQL exactos."
    )

    resp = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": consulta.strip()},
        ],
        functions=[FUNC_SCHEMA],
        function_call={"name": "extract_filters"},
    )

    try:
        return json.loads(resp.choices[0].message.function_call.arguments)
    except (KeyError, json.JSONDecodeError) as e:
        logger.error("LLM args error: %s", e)
        raise HTTPException(502, "Respuesta inesperada del modelo LLM")

# ──────────────────── Consultas Oracle ────────────────────
def _consultar_por_familia(conn, familia: str) -> list[ProductItem]:
    sql = """
        SELECT COD_PRODUCTO_NK2, LINEA_NEGOCIO, ESTADO_INVENTARIO, CALIDAD,
               MARCA, FORMATO, APLICACION, COLOR, COLECCION, SUBMARCA
          FROM DWH_PROD.DIM_PRODUCTO
         WHERE FAMILIA = :family
    """
    cur = conn.cursor()
    try:
        cur.execute(sql, family=familia.upper())
        cols = [c[0] for c in cur.description]
        return [ProductItem(**dict(zip(cols, r))) for r in cur.fetchall()]
    finally:
        cur.close()

def _consultar_por_filtros(conn, filtros: dict) -> list[ProductItem]:
    campos = ("FAMILIA", "COLOR", "FORMATO", "CALIDAD", "APLICACION")
    clauses, params = [], {}

    for campo in campos:
        val = filtros.get(campo)
        if val:
            clauses.append(f"{campo} LIKE :{campo}")
            params[campo] = f"%{val.upper()}%"

    if not clauses:
        raise HTTPException(400, "No se pudo inferir ningún filtro válido")

    sql = (
        "SELECT COD_PRODUCTO_NK2, LINEA_NEGOCIO, ESTADO_INVENTARIO, CALIDAD,"
        "       MARCA, FORMATO, APLICACION, COLOR, COLECCION, SUBMARCA"
        "  FROM DWH_PROD.DIM_PRODUCTO"
        f" WHERE {' AND '.join(clauses)}"
        " FETCH FIRST 200 ROWS ONLY"
    )
    cur = conn.cursor()
    try:
        cur.execute(sql, params)
        cols = [c[0] for c in cur.description]
        return [ProductItem(**dict(zip(cols, r))) for r in cur.fetchall()]
    finally:
        cur.close()

# ──────────────────── Resúmenes con LLM ────────────────────
async def _redactar_resumen_desde_texto(consulta: str, productos: list[ProductItem]):
    formatos = sorted({p.FORMATO for p in productos})
    colores = sorted({p.COLOR for p in productos})
    familias = sorted({p.COLECCION for p in productos if p.COLECCION != "NO APLICA"})
    n_refs = len(productos)

    system_msg = (
        "Eres un asesor experto de Graiman. "
        "Resume en ≤120 palabras y termina con una tabla Markdown "
        "Formato–Color. Si solo hay 1 producto, menciónalo explícitamente."
    )
    user_msg = (
        f"Petición: {consulta}\nCoincidencias: {n_refs}\n"
        f"Formatos: {', '.join(formatos) or '—'}\n"
        f"Colores:  {', '.join(colores) or '—'}\n"
        f"Colecciones: {', '.join(familias) or '—'}"
    )

    resp = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
    )
    return resp.choices[0].message.content.strip()

async def _redactar_resumen(familia: str, productos: list[ProductItem]):
    formatos = sorted({p.FORMATO for p in productos})
    colores = sorted({p.COLOR for p in productos})
    calidades = sorted({p.CALIDAD for p in productos})
    n_refs = len(productos)

    system_msg = (
        "Eres un asesor experto de Graiman. "
        "Máx 200 palabras, tono cercano, segunda persona, "
        "y tabla Markdown Formato–Color al final."
    )
    user_msg = (
        f"Familia: {familia}\nReferencias: {n_refs}\n"
        f"Formatos: {', '.join(formatos)}\n"
        f"Colores:  {', '.join(colores)}\n"
        f"Calidades: {', '.join(calidades)}"
    )

    resp = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
    )
    return resp.choices[0].message.content.strip()
