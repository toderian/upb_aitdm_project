#!/bin/bash
# run_fl_experiment.sh

set -e

# Set Python path to include the project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "[FL Experiment] Starting Federated Learning experiment..."

# Cleanup on exit
cleanup() {
    echo "[FL Experiment] Cleaning up processes..."
    kill $(jobs -p) 2>/dev/null || true
    exit 0
}
trap cleanup EXIT INT TERM

# Start server in background
echo "[FL Experiment] Starting server on port 8080..."
python model_side/federated/server.py --server_address 127.0.0.1:8080 --num_rounds 10 --num_clients 3 --local_epochs 3 &
SERVER_PID=$!

# Wait for server to start
sleep 5

if ! ps -p $SERVER_PID > /dev/null; then
    echo "[FL Experiment] ERROR: Server failed to start"
    exit 1
fi

# Start clients
echo "[FL Experiment] Starting 3 clients..."
python model_side/federated/client.py --client_id 0 --server_address 127.0.0.1:8080 &
python model_side/federated/client.py --client_id 1 --server_address 127.0.0.1:8080 &
python model_side/federated/client.py --client_id 2 --server_address 127.0.0.1:8080 &

echo "[FL Experiment] All processes started. Monitoring..."

# Wait for all processes    
wait