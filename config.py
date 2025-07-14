# config.py
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """
    Configuración de la API, cargada desde variables de entorno o .env
    """
    root_dir: Path = Field(
        default=Path(r"C:\Users\juanj\Desktop\DATA FINAL\Ruido"),
        env="ROOT_DIR",
        description="Directorio raíz con subcarpetas por clase"
    )
    model_path: Path = Field(
        default=Path("mejor_modelo_lbp_RFC_FINAL.pkl"),
        env="MODEL_PATH",
        description="Ruta al archivo .pkl del modelo entrenado"
    )
    topk: int = Field(
        default=5,
        env="TOPK",
        ge=1,
        description="Número por defecto de predicciones a devolver"
    )
    classes_file: Path = Field(
        default=Path("classes.json"),
        env="CLASSES_FILE",
        description="Ruta al archivo JSON con las clases"
    )
    
    # Parámetros de Oracle
    oracle_lib_dir: Path = Field(
        ...,  # obligatorio
        env="ORACLE_LIB_DIR",
        description="Carpeta del Instant Client de Oracle"
    )
    oracle_config_dir: Path = Field(
        ...,  # obligatorio
        env="ORACLE_CONFIG_DIR",
        description="Carpeta del wallet/config de Oracle"
    )
    oracle_user: str = Field(
        ...,  # obligatorio
        env="ORACLE_USER",
        description="Usuario de la base de datos Oracle"
    )
    oracle_password: str = Field(
        ...,  # obligatorio
        env="ORACLE_PASSWORD",
        description="Password del usuario Oracle"
    )
    oracle_dsn: str = Field(
        ...,  # obligatorio
        env="ORACLE_DSN",
        description="DSN (host/servicio) de Oracle"
    )
    openai_api_key: str = Field(
        ...,  # obligatorio
        env="OPENAI_API_KEY",
        description="API Key de OpenAI"
    )


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
# Instancia única de configuración
settings = Settings()
