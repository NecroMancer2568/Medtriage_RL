#!/bin/bash
set -e

echo "Starting MedTriage-RL..."

# Start FastAPI environment server on port 7860
uvicorn src.server:app --host 0.0.0.0 --port 7860 &
FASTAPI_PID=$!
echo "FastAPI started on port 7860 (PID $FASTAPI_PID)"

# Wait for FastAPI to be ready
echo "Waiting for FastAPI to be ready..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:7860/health > /dev/null 2>&1; then
        echo "FastAPI is ready"
        break
    fi
    sleep 1
done

# Start Gradio UI on port 7861
python3 ui/app.py &
GRADIO_PID=$!
echo "Gradio started on port 7861 (PID $GRADIO_PID)"

echo "MedTriage-RL is running"
echo "  Environment API : http://localhost:7860"
echo "  Interactive UI  : http://localhost:7861"

# Keep container alive — exit if either process dies
wait -n $FASTAPI_PID $GRADIO_PID
EXIT_CODE=$?
echo "A process exited with code $EXIT_CODE"
exit $EXIT_CODE