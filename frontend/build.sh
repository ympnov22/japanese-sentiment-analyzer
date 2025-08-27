#!/bin/bash
set -e

echo "=== BUILD SCRIPT DEBUG ==="
echo "BUILD_ENV from environment: '$BUILD_ENV'"
echo "All environment variables:"
env | grep -E "(BUILD_ENV|VITE_)" || echo "No BUILD_ENV or VITE_ variables found"
echo "=========================="

BUILD_ENV=${BUILD_ENV:-production}
echo "Building frontend for environment: $BUILD_ENV"

mkdir -p /tmp/build

cp -r . /tmp/build/

if [ -f ".env.$BUILD_ENV" ]; then
    echo "Loading environment from .env.$BUILD_ENV"
    source ".env.$BUILD_ENV"
else
    echo "Warning: .env.$BUILD_ENV not found, using defaults"
    if [ "$BUILD_ENV" = "staging" ]; then
        VITE_API_BASE_URL="https://japanese-sentiment-analyzer-staging.fly.dev"
    else
        VITE_API_BASE_URL="https://japanese-sentiment-analyzer.fly.dev"
    fi
fi

echo "Using API Base URL: $VITE_API_BASE_URL"

echo "Before replacement:"
grep -n "__VITE_API_BASE_URL__" /tmp/build/script.js || echo "Placeholder not found"

sed -i "s|__VITE_API_BASE_URL__|$VITE_API_BASE_URL|g" /tmp/build/script.js

echo "After replacement:"
grep -n -A2 -B2 "API_BASE_URL.*=" /tmp/build/script.js || echo "API_BASE_URL not found"

echo "Build completed successfully"
