# Fase 1: Construcción con dependencias completas
FROM python:3.13-slim AS builder

WORKDIR /app

# Instalar herramientas de compilación
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        libpython3-dev \
        make \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt
COPY requirements.txt .

# Instalar dependencias con pip
RUN pip install --user -r requirements.txt


# Fase 2: Imagen final liviana
FROM python:3.13-slim

WORKDIR /app


ARG ORACLE_USER
ARG ORACLE_PASSWORD
ARG ORACLE_DSN
ARG OPENAI_API_KEY

ENV ORACLE_LIB_DIR=/opt/oracle/instantclient \
    ORACLE_CONFIG_DIR=/opt/oracle/wallet \
    ORACLE_USER=${ORACLE_USER} \
    ORACLE_PASSWORD=${ORACLE_PASSWORD} \
    ORACLE_DSN=${ORACLE_DSN}\
    OPENAI_API_KEY=${OPENAI_API_KEY}

# Inicializar LD_LIBRARY_PATH antes de usarlo
ENV LD_LIBRARY_PATH=""

# Variables de entorno para Oracle
ENV ORACLE_HOME=/opt/oracle/instantclient
ENV LD_LIBRARY_PATH=$ORACLE_HOME:$LD_LIBRARY_PATH
ENV TNS_ADMIN=/opt/oracle/wallet

# Instalar dependencias del sistema + libaio1
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libaio1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copiar dependencias instaladas desde builder
COPY --from=builder /root/.local /root/.local

# Añadir PATH globalmente
ENV PATH=/root/.local/bin:$PATH

# Copiar código fuente y recursos
COPY TestUnificado.py config.py features.py ./
COPY mejor_modelo_lbp_RFC_FINAL.pkl classes.json ./

# Copiar wallet de Oracle
COPY wallet/ $TNS_ADMIN/

# Ajustar permisos del wallet
RUN chmod -R 600 $TNS_ADMIN/* && \
    chown -R root:root $TNS_ADMIN

# Copiar Oracle Instant Client Basic
COPY instantclient/instantclient_23_7 $ORACLE_HOME/

# Crear symlink necesario
RUN cd $ORACLE_HOME && \
    ln -sf libclntsh.so.23.1 libclntsh.so || true

# Verificar que Oracle funcione
RUN if [ ! -f "$ORACLE_HOME/libclntsh.so.23.1" ]; then \
        echo "Oracle Instant Client no encontrado"; exit 1; \
    fi

# Valores por defecto de directorios Oracle
ENV ORACLE_LIB_DIR=/opt/oracle/instantclient \
    ORACLE_CONFIG_DIR=/opt/oracle/wallet


# Exponer puerto FastAPI
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "TestUnificado:app", "--host", "0.0.0.0", "--port", "8000"]
