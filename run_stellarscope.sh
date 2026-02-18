#!/usr/bin/env bash
set -e

APP_DIR="stellarscope"
COMPOSE_URL="https://raw.githubusercontent.com/ShahriNasa/StarsPrediction/refs/heads/main/stellarscopeApp/docker-compose.release.yml"

# -------------------------
# docker wrapper (fallback to sudo)
# -------------------------
DOCKER_BIN="docker"

function docker_ok() {
  # checks if docker works without sudo
  docker info >/dev/null 2>&1
}

function ensure_docker() {
  if command -v docker >/dev/null 2>&1; then
    :
  else
    echo "❌ Docker not found. Install Docker Desktop (Mac/Windows) or docker engine (Linux)."
    exit 1
  fi

  if docker_ok; then
    DOCKER_BIN="docker"
    return
  fi

  # If docker fails, try sudo docker (Linux case)
  if command -v sudo >/dev/null 2>&1; then
    if sudo docker info >/dev/null 2>&1; then
      DOCKER_BIN="sudo docker"
      echo "ℹ️  Docker requires sudo on this machine. Using: $DOCKER_BIN"
      return
    fi
  fi

  echo "❌ Docker is installed but not usable."
  echo "   Common reasons:"
  echo "   - Docker daemon not running"
  echo "   - Your user is not in the docker group (Linux)"
  echo "   - Docker Desktop not started (Mac/Windows)"
  echo ""
  echo "Try:"
  echo "  docker info"
  echo "or on Linux:"
  echo "  sudo docker info"
  exit 1
}

function dc() {
  # docker compose wrapper (works for both docker and sudo docker)
  $DOCKER_BIN compose "$@"
}

# -------------------------
# setup
# -------------------------
function ensure_setup() {
  mkdir -p "$APP_DIR"
  cd "$APP_DIR"

  mkdir -p data

  if [ ! -f docker-compose.yml ]; then
    echo "⬇️  Downloading docker-compose file..."
    curl -L -o docker-compose.yml "$COMPOSE_URL"
  fi
}

# -------------------------
# commands
# -------------------------
function start_app() {
  ensure_docker
  ensure_setup
  echo "🐳 Starting StellarScope..."
  dc up -d
  echo ""
  echo "✅ Running at:"
  echo "   http://localhost:8080"
}

function stop_app() {
  ensure_docker
  if [ -d "$APP_DIR" ]; then
    cd "$APP_DIR"
    echo "🛑 Stopping StellarScope..."
    dc down
    echo "✅ Containers stopped."
  else
    echo "No installation found."
  fi
}

function restart_app() {
  stop_app
  start_app
}

function show_logs() {
  ensure_docker
  if [ -d "$APP_DIR" ]; then
    cd "$APP_DIR"
    dc logs -f
  else
    echo "No installation found."
  fi
}

# -------------------------
# entrypoint
# -------------------------
case "${1:-}" in
  start)
    start_app
    ;;
  stop)
    stop_app
    ;;
  restart)
    restart_app
    ;;
  logs)
    show_logs
    ;;
  *)
    echo "Usage:"
    echo "  ./stellarscope.sh start"
    echo "  ./stellarscope.sh stop"
    echo "  ./stellarscope.sh restart"
    echo "  ./stellarscope.sh logs"
    ;;
esac

