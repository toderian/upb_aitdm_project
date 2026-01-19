#!/bin/bash
# run_fl_experiment.sh

# Set Python path to include the project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start server in background
python model_side/federated/server.py --num_rounds 10 --num_clients 3 --local_epochs 3 &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Start clients
python model_side/federated/client.py --client_id 0 --server_address 0.0.0.0:8080 &
python model_side/federated/client.py --client_id 1 --server_address 0.0.0.0:8080 &
python model_side/federated/client.py --client_id 2 --server_address 0.0.0.0:8080 &

# Wait for all processes
wait