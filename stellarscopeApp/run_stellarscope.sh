#!/usr/bin/env bash

set -e

echo "🚀 Setting up StellarScope..."

# Create project folder
mkdir -p stellarscope
cd stellarscope

# Create persistent data directory
mkdir -p data

echo "⬇️  Downloading docker-compose file..."
curl -L -o docker-compose.yml \
https://raw.githubusercontent.com/ShahriNasa/StarsPrediction/refs/heads/main/stellarscopeApp/docker-compose.release.yml

echo "🐳 Starting containers..."
docker compose up -d

echo "✅ StellarScope should now be running at:"
echo "http://localhost:8080"

