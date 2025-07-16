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
from starlette.responses import PlainTextResponse, Response
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request


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

@app.middleware("http")
async def check_content_length(request: Request, call_next) -> Response:
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_IMAGE_SIZE:
        return PlainTextResponse(f"Imagen supera el límite de {round(MAX_IMAGE_SIZE / (1024 * 1024))} MB", status_code=413)
    return await call_next(request)

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

    assistant_file = Path("assistant_id.json")

    if assistant_file.exists():
        with assistant_file.open("r", encoding="utf-8") as f:
            assistant_data = json.load(f)
            assistant_id = assistant_data.get("id")
            logger.info(f"✅ Assistant existente cargado: {assistant_id}")
    else:
        assistant = await openai_client.beta.assistants.create(
            name="Asistente Graiman",
            instructions="Eres un asesor de cerámicas Graiman. Usa la herramienta '_consultar_por_filtros_real' si necesitas buscar productos en la base de datos. No puedes responder nada mas que no sea productos de Graiman pero no lo recuerdes al usuario a cada rato, solo si es necesario. Se amable y formal.",
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
        with assistant_file.open("w", encoding="utf-8") as f:
            json.dump({"id": assistant_id}, f)
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
@app.get("/graimanchat")
async def conversar_sql(mensaje: str = Form(None), thread_id: str = Form(None), imagen: UploadFile | None = File(None), topk: int = Form(3, gt=1, le=5)):

    #Inicializar thread
    if not thread_id:
        thread = await openai_client.beta.threads.create()
        thread_id = thread.id

    #Si se envía imagen, extraer características
    principal = None
    otros = None

    if imagen:
        principal, otros = await _clasificar_imagen(imagen, topk)
        logger.info(f"Familia principal: {principal}")
        logger.info(f"Otras familias: {otros}")
        # Mensaje de imagen como contexto fuerte
        mensaje_imagen = (
            f"Este mensaje NO es para mostrar al usuario, solo para que lo uses como contexto: "
            f"Se subió una imagen a un modelo de ML que la clasificó como familia principal '{principal}', "
            f"y otras posibles familias similares fueron: {', '.join(otros) if otros else 'ninguna otra'}.\n\n"
            f"En tu respuesta al usuario:\n"
            f"1. Si el usuario NO dijo nada adicional, responde con:\n"
            f"   - Una sección clara con información detallada SOLO de la familia principal '{principal}'.\n"
            f"   - Luego, una sección opcional donde menciones otras familias similares.\n"
            f"2. Si el usuario escribió algo, PRIORIZA responder exactamente lo que pidió, usando la información de la imagen como apoyo.\n\n"
            f"Siempre responde en **Markdown válido**, de forma formal, clara, amable y sin repetir que eres un modelo o dar información irrelevante. "
            f"NUNCA respondas sobre temas que no sean cerámicas Graiman."
        )


        if mensaje:
            contenido_para_assistant = (
                f"{mensaje_imagen}\n\nEl usuario escribió lo siguiente: \"{mensaje.strip()}\".\n"
                "Responde de acuerdo a eso, como si fuera el jefe, y si puedes usar información del análisis de imagen, úsala como soporte."
            )
        else:
            contenido_para_assistant = mensaje_imagen

    else:
        contenido_para_assistant = mensaje

    #Enviar mensaje al assistant
    await openai_client.beta.threads.messages.create(thread_id=thread_id, role="user", content=contenido_para_assistant)
    run = await openai_client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)

    while True:
        status = await openai_client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if status.status == "requires_action":
            tool_outputs = []
            for tool_call in status.required_action.submit_tool_outputs.tool_calls:
                args = json.loads(tool_call.function.arguments)
                productos = _consultar_por_filtros_real(**args)
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": json.dumps([p.dict() for p in productos])
                })

            await openai_client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )


        elif status.status == "completed":
            break
        await asyncio.sleep(1)

    messages = await openai_client.beta.threads.messages.list(thread_id=thread_id)
    for msg in messages.data:
        if msg.role == "assistant":
            for c in msg.content:
                if hasattr(c, "text"):
                    return {"respuesta": c.text.value.strip(), "thread_id": thread_id, "Principal": principal, "Otras": otros}

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

# ─────────────── Utils ───────────────

# ─────────────── Clasificación por imagen ───────────────
async def _clasificar_imagen(imagen: UploadFile, topk: int) -> str:
    # 1. Leer la imagen en chunks y validar tamaño
    content = await safe_read_upload(imagen)
    
    # 2. Calcular características LBP en un hilo
    async with lbp_semaphore:
        loop = asyncio.get_running_loop()
        feats = await loop.run_in_executor(executor, compute_lbp, content)
    
    # 3. Validar extracción
    if feats is None:
        raise HTTPException(422, "Imagen inválida")

    # 4. Realizar predicción
    probs = clf.predict_proba(feats.reshape(1, -1))[0]
    mejores = probs.argsort()[::-1][:topk]

    # 5. Obtener familias predichas y sus probabilidades
    predicciones = [(classes[i], round(probs[i] * 100, 2)) for i in mejores]
    familia_principal = predicciones[0][0]
    try:
        otrasFamilias = [fam for fam, prob in predicciones[1:]]
    except:
        otrasFamilias = []

    return familia_principal, otrasFamilias


# ─────────────── Safe read upload ───────────────
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