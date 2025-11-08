#!/usr/bin/env bash

# Logs con colores y timestamp

# Colores ANSI
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
BOLD="\033[1m"
RESET="\033[0m"

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

log() {
    local level="$1"; shift
    local color="$1"; shift
    printf "%s %b[%s]%b %s\n" "$(timestamp)" "$color" "$level" "$RESET" "$*"
}

info()  { log "INFO"  "$BLUE"  "$@"; }
ok()    { log "OK"    "$GREEN" "$@"; }
warn()  { log "WARN"  "$YELLOW" "$@"; }
err()   { log "ERROR" "$RED"   "$@"; }

send_request() {
    local method="$1"; local url="$2"; local data="$3"
    info "Request: $method $url"

    local tmp; tmp="$(mktemp)" || { err "No se pudo crear archivo temporal"; return 1; }

    local curl_args=("-sS" "-o" "$tmp" "-w" "%{http_code}" "-X" "$method" "-H" "Accept: application/json")
    if [[ -n "$data" ]]; then
        curl_args+=("-H" "Content-Type: application/json" "-d" "$data")
    fi
    curl_args+=("$url")

    local status
    if ! status="$(curl "${curl_args[@]}" 2>/dev/null)"; then
        err "curl fallo al conectar con $url"
        rm -f "$tmp"
        return 2
    fi

    local body
    body="$(cat "$tmp")"
    rm -f "$tmp"

    local pretty="$body"
    if command -v jq >/dev/null 2>&1 && [[ "$body" =~ ^[[:space:]]*[\{\[] ]]; then
        pretty="$(printf "%s" "$body" | jq . 2>/dev/null || printf "%s" "$body")"
    fi

    if (( status >= 200 && status < 300 )); then
        ok "[$status] Respuesta satisfactoria:"
        printf "%s\n" "$pretty"
        return 0
    else
        err "[$status] Respuesta inesperada:"
        printf "%s\n" "$pretty"
        return 3
    fi
}

run_requests() {
    local base="$1"

    if [[ "$base" != http://* && "$base" != https://* ]]; then
        base="http://$base"
    fi

    base="${base%/}"

    send_request GET  "${base}/health" || return $?
    send_request GET  "${base}/info"   || return $?
    send_request POST "${base}/predict" '{"features": {"pclass": 3, "sex": "male", "age": 59.0, "sibsp": 0, "parch": 0, "fare": 7.25, "embarked": "S"}}' || return $?

    ok "Todas las peticiones realizadas correctamente."
}


if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [[ $# -lt 1 ]]; then
        echo "Uso: $0 <base-url-con-puerto>  (ej: localhost:8000 o http://localhost:8000)" >&2
        exit 2
    fi
    run_requests "$1"
fi