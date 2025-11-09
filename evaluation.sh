#!/bin/bash
# Script de evaluación automática

# Verificar argumentos
if [ $# -ne 1 ]; then
    echo "Uso: $0 <directorio_del_proyecto>"
    echo "Ejemplo: $0 ."
    exit 1
fi

PROJECT_PATH=$1
ORIGINAL_DIR=$(pwd)

echo "=== Iniciando pipeline completo de empaquetado y evaluación ==="

# Paso 0: Cambiar al directorio del proyecto y hacer package
echo "Paso 0: Empaquetando proyecto..."
cd "$PROJECT_PATH" || exit 1
TEAM_NAME=$(grep "TEAM_NAME=" Makefile | cut -d'=' -f2)

make package
if [ $? -ne 0 ]; then
    echo "Error al empaquetar el proyecto"
    exit 1
fi

TARBALL="${TEAM_NAME}.tar.gz"

if [ ! -f "$TARBALL" ]; then
    echo "Error: No se encontró el tarball $TARBALL"
    exit 1
fi

echo "Tarball creado: $TARBALL"

# Volver al directorio original
cd "$ORIGINAL_DIR" || exit 1

# Copiar tarball si es necesario
if [ "$PROJECT_PATH" != "." ]; then
    cp "${PROJECT_PATH}/${TARBALL}" .
fi

echo "=== Iniciando evaluación de $TEAM_NAME ==="

# 1. Extraer el tarball
echo "Paso 1: Extrayendo tarball..."
tar -xzvf "$TARBALL"
if [ $? -ne 0 ]; then
    echo "Error al extraer el tarball"
    exit 1
fi

# 2. Encontrar y entrar al directorio del proyecto
PROJECT_DIR=$(tar -tzf "$TARBALL" | head -1 | cut -f1 -d"/")
cd "$PROJECT_DIR" || exit 1
echo "Paso 2: Directorio del proyecto: $PROJECT_DIR"

# 3. Construir imagen y levantar contenedor
echo "Paso 3: Construyendo imagen..."
make build
if [ $? -ne 0 ]; then
    echo "Error al construir la imagen"
    exit 1
fi

echo "Levantando contenedor..."
make run
if [ $? -ne 0 ]; then
    echo "Error al levantar el contenedor"
    exit 1
fi

# Esperar a que el contenedor esté listo
sleep 3

# 4. Probar endpoints
echo "Paso 4: Probando endpoints..."

make test

# 5. Detener y limpiar
echo "Paso 5: Deteniendo y limpiando contenedor..."
make stop
make clean

# Volver al directorio original
cd "$ORIGINAL_DIR" || exit 1

echo "=== Pipeline completado para $TEAM_NAME ==="