#https://chatgpt.com/share/686fc2c9-7158-8008-a320-0b7d06311aac
#https://platform.openai.com/assistants/asst_bGXgPN8OAhYQOirDQkXqYnVg
#https://platform.openai.com/settings/organization/billing/overview
#https://platform.openai.com/docs/assistants/quickstart

from __future__ import annotations

import asyncio, json, logging
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import List

import cx_Oracle, joblib, numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse
from starlette.middleware.cors import CORSMiddleware


from config import settings
from features import compute_lbp

from openai import AsyncOpenAI
from openai.types.beta.threads import Message


# ─────────────── Logger ───────────────
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("ceramica-api")

# ─────────────── Constantes ───────────────
MAX_IMAGE_SIZE = 10 * 1024 * 1024
CPU_WORKERS = max(2, cpu_count() // 2)
CHUNK_SIZE = 8192

# ─────────────── App ───────────────
app = FastAPI(
    title="Agente Cerámicas Graiman",
    description="Clasificador LBP + Assistant con memoria y SQL real",
    version="2.1",
)


# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O especifica dominios como ["https://miapp.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    BaseHTTPMiddleware,
    dispatch=lambda request, call_next: (
        PlainTextResponse("Archivo demasiado grande", status_code=413)
        if (cl := request.headers.get("content-length")) and int(cl) > MAX_IMAGE_SIZE
        else call_next(request)
    ),
)

# ─────────────── Globals ───────────────
clf: joblib.BaseEstimator | None = None
classes: List[str] = []
executor = ThreadPoolExecutor(max_workers=CPU_WORKERS)
lbp_semaphore = asyncio.Semaphore(CPU_WORKERS)

openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
assistant_id = None

# ─────────────── Modelos ───────────────
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

# ─────────────── Inicialización ───────────────
@app.on_event("startup")
async def startup_event():
    global clf, classes, pool, assistant_id

    clf = joblib.load(settings.model_path)
    logger.info("✅ Modelo cargado")

    cf: Path = settings.classes_file
    if cf.is_file():
        classes[:] = sorted(json.load(cf.open(encoding="utf-8")))
    else:
        classes[:] = sorted(p.name for p in settings.root_dir.iterdir() if p.is_dir())
        cf.parent.mkdir(parents=True, exist_ok=True)
        json.dump(classes, cf.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    logger.info(f"✅ {len(classes)} clases cargadas")

    cx_Oracle.init_oracle_client(
        lib_dir=str(settings.oracle_lib_dir),
        config_dir=str(settings.oracle_config_dir),
    )
    pool = cx_Oracle.SessionPool(
        user=settings.oracle_user,
        password=settings.oracle_password,
        dsn=settings.oracle_dsn,
        min=2,
        max=10,
        increment=1,
        encoding="UTF-8",
    )
    logger.info("✅ Pool Oracle OK")

    assistant = await openai_client.beta.assistants.create(
        name="Asistente Graiman",
        instructions="Eres un asesor de cerámicas Graiman. Usa la herramienta '_consultar_por_filtros_real' si necesitas buscar productos reales.",
        tools=[{
            "type": "function",
            "function": {
                "name": "_consultar_por_filtros_real",
                "description": "Buscar productos por filtros en la base de datos de Graiman",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "COLOR": {"type": "string"},
                        "FORMATO": {"type": "string"},
                        "CALIDAD": {"type": "string"},
                        "APLICACION": {"type": "string"},
                        "FAMILIA": {"type": "string"}
                    },
                    "required": ["FAMILIA"]
                }
            }
        }],
        model="gpt-4o-mini"
    )
    assistant_id = assistant.id
    logger.info(f"✅ Assistant creado: {assistant_id}")

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

# ─────────────── Endpoint de conversación con función SQL ───────────────
@app.post("/graiman/sqlchat")
async def conversar_sql(mensaje: str = Form(...), thread_id: str = Form(None)):
    if not thread_id:
        thread = await openai_client.beta.threads.create()
        thread_id = thread.id

    await openai_client.beta.threads.messages.create(thread_id=thread_id, role="user", content=mensaje)
    run = await openai_client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)

    while True:
        status = await openai_client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if status.status == "requires_action":
            tool_call = status.required_action.submit_tool_outputs.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            productos = _consultar_por_filtros_real(**args)
            await openai_client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=[{
                    "tool_call_id": tool_call.id,
                    "output": json.dumps([p.dict() for p in productos])
                }]
            )
        elif status.status == "completed":
            break
        await asyncio.sleep(1)

    messages = await openai_client.beta.threads.messages.list(thread_id=thread_id)
    for msg in messages.data:
        if msg.role == "assistant":
            for c in msg.content:
                if hasattr(c, "text"):
                    return {"respuesta": c.text.value.strip(), "thread_id": thread_id}

    return {"respuesta": "Sin respuesta del asistente", "thread_id": thread_id}

# ─────────────── Consulta SQL real ───────────────
def _consultar_por_filtros_real(**filtros) -> List[ProductItem]:
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
        " MARCA, FORMATO, APLICACION, COLOR, COLECCION, SUBMARCA "
        "FROM DWH_PROD.DIM_PRODUCTO "
        f"WHERE {' AND '.join(clauses)} FETCH FIRST 50 ROWS ONLY"
    )

    conn = pool.acquire()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        cols = [c[0] for c in cur.description]
        return [ProductItem(**dict(zip(cols, r))) for r in cur.fetchall()]
    finally:
        pool.release(conn)


# # ─────────────── Clasificación por imagen + Assistant ───────────────
# @app.post("/graiman/imagechat")
# async def clasificar_imagen_y_conversar(
#     imagen: UploadFile = File(...),
#     topk: int = Form(3, gt=0, le=20),
# ):
#     # Extraer características
#     content = await safe_read_upload(imagen)
#     async with lbp_semaphore:
#         loop = asyncio.get_running_loop()
#         feats = await loop.run_in_executor(executor, compute_lbp, content)

#     if feats is None:
#         raise HTTPException(422, "Imagen inválida")

#     # Clasificación
#     probs = clf.predict_proba(feats.reshape(1, -1))[0]
#     mejores = probs.argsort()[::-1][:topk]
#     familia = classes[int(mejores[0])]

#     # Consultar en BD
#     conn = pool.acquire()
#     try:
#         productos = _consultar_por_familia(conn, familia)
#     finally:
#         pool.release(conn)

#     # Crear hilo
#     thread = await openai_client.beta.threads.create()
#     thread_id = thread.id

#     # Enviar contexto al Assistant
#     user_msg = (
#     f"La imagen enviada fue clasificada como familia '{familia}'. "
#     f"Usa la herramienta disponible para mostrar productos relacionados en base a esta familia. "
#     f"Quiero conocer ejemplos y características clave."
#     )

#     await openai_client.beta.threads.messages.create(thread_id=thread_id, role="user", content=user_msg)

#     # Ejecutar Assistant
#     run = await openai_client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)
#     while True:
#         status = await openai_client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
#         if status.status == "completed":
#             break
#         await asyncio.sleep(1)

#     # Obtener respuesta
#     messages = await openai_client.beta.threads.messages.list(thread_id=thread_id)
#     for msg in messages.data:
#         if msg.role == "assistant":
#             for c in msg.content:
#                 if hasattr(c, "text"):
#                     return {
#                         "familia": familia,
#                         "productos": [p.dict() for p in productos],
#                         "respuesta": c.text.value.strip(),
#                         "thread_id": thread_id
#                     }

#     return {"respuesta": "No se obtuvo respuesta del asistente", "thread_id": thread_id}




#----------------------------------------------------------------------------------------------------------
class ProductsResponse(BaseModel):
    resumen: str
    productos: list[ProductItem]

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
        "Formato-Color. Si solo hay 1 producto, menciónalo explícitamente."
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


async def _flujo_por_imagen(imagen: UploadFile, topk: int, conn):
    feats = await extract_features(imagen)
    if feats is None:
        raise HTTPException(422, "Imagen inválida o no soportada")

    probs = clf.predict_proba(feats.reshape(1, -1))[0]
    mejores = probs.argsort()[::-1][:topk]
    familia = classes[int(mejores[0])]
    productos = _consultar_por_familia(conn, familia)
    return familia, productos


async def extract_features(file: UploadFile) -> np.ndarray | None:
    """Lee imagen (limite 10 MB) y calcula LBP en pool de hilos."""
    content = await safe_read_upload(file, MAX_IMAGE_SIZE)
    async with lbp_semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, compute_lbp, content)
    

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