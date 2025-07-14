#features.py
'''Módulo optimizado para cálculo y caching de descriptores LBP.''' 

import logging
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
import hashlib

from joblib import Parallel, delayed


# Parámetros LBP
RADIUS = 3
N_POINTS = 8 * RADIUS
METHOD = "uniform"
# Resolución menor para reducir costo de cómputo
IMAGE_SIZE = (512, 512)
# Nombre por defecto para cache en disco
CACHE_NAME = "dataset_lbp.npz"

# Cache en memoria: key = SHA256 del contenido
_memcache: dict[str, np.ndarray] = {}
# Bins precomputados para histogramas
_LBP_BINS = np.arange(0, N_POINTS + 3)

def compute_lbp(image_input) -> np.ndarray | None:
    """
    Calcula histograma LBP normalizado.
    Cachea por SHA256 en _memcache para evitar keying por bytes completos.
    """
    try:
        # Obtén los bytes para el hash
        if isinstance(image_input, (str, Path)):
            data = Path(image_input).read_bytes()
        elif isinstance(image_input, (bytes, bytearray)):
            data = bytes(image_input)
        elif isinstance(image_input, Image.Image):
            buf = BytesIO()
            image_input.save(buf, format="PNG")
            data = buf.getvalue()
        else:
            raise TypeError(f"Tipo no soportado: {type(image_input)}")

        key = hashlib.sha256(data).hexdigest()
        if key in _memcache:
            return _memcache[key]

        # Reconstruye imagen y calcula LBP
        img = Image.open(BytesIO(data)).convert("L").resize(IMAGE_SIZE)
        arr = np.asarray(img, dtype=np.uint8)
        lbp_arr = local_binary_pattern(arr, P=N_POINTS, R=RADIUS, method=METHOD)
        hist, _ = np.histogram(lbp_arr.ravel(),
                               bins=_LBP_BINS,
                               range=(0, N_POINTS + 2))
        hist = hist.astype(np.float32)
        result = hist / (hist.sum() + 1e-8)

        _memcache[key] = result
        return result

    except Exception as e:
        logging.getLogger(__name__).warning("Error LBP en %s: %s", image_input, e)
        return None



def load_dataset(
    root_dir: Path,
    cache_file: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga o genera (y cachea) arrays de features y labels LBP.

    - Busca NPZ cacheado en `cache_file` o en `root_dir/CACHE_NAME`.
    - En su defecto, itera directorios (cada subcarpeta = label),
      extrae LBP con `compute_lbp` y guarda resultados.
    """
    root = Path(root_dir)
    cache = Path(cache_file) if cache_file else (root / CACHE_NAME)

    if cache.exists():
        data = np.load(cache, mmap_mode="r")
        logging.getLogger(__name__).info("Dataset cargado de cache (mmap): %s", cache)
        return data["features"], data["labels"]


    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    cls_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Recolectar rutas e índices
    items = []
    for cls in classes:
        for img_path in (root / cls).iterdir():
            if img_path.is_file():
                items.append((img_path, cls_to_idx[cls]))

    # ── Sustituir bucle secuencial por paralelización con Joblib ──
    results = Parallel(n_jobs=-1)(
        delayed(compute_lbp)(path) for path, _ in items
    )
    # Construir arrays solo con las extracciones válidas
    features = np.vstack([f for f in results if f is not None])
    labels = np.array([
        lbl for (f, lbl) in zip(results, (lbl for _, lbl in items)) if f is not None
    ], dtype=int)

    # Cachear en disco
    np.savez_compressed(cache, features=features, labels=labels)
    logging.getLogger(__name__).info("Dataset generado y cacheado en: %s", cache)
    return features, labels

