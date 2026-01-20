#!/bin/bash
# Run Federated Learning with Differential Privacy
# Usage: ./run_fl_dp.sh [epsilon] [num_rounds]

EPSILON=${1:-5.0}
NUM_ROUNDS=${2:-3}
NUM_CLIENTS=3
LOCAL_EPOCHS=1
BATCH_SIZE=32

echo "=============================================="
echo "FL + DP Experiment"
echo "  Epsilon: $EPSILON"
echo "  Rounds: $NUM_ROUNDS"
echo "  Clients: $NUM_CLIENTS"
echo "=============================================="

# Create output directories
mkdir -p results/fl_dp
mkdir -p models/fl_dp

# Kill any existing processes
pkill -f "server_dp.py" 2>/dev/null
pkill -f "client_dp.py" 2>/dev/null
sleep 2

# Start server in background
echo "Starting server..."
python -m model_side.federated.server_dp \
    --server_address 127.0.0.1:8080 \
    --num_clients $NUM_CLIENTS \
    --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --output_dir results/fl_dp &
SERVER_PID=$!

sleep 5

# Start clients in background
for i in $(seq 0 $((NUM_CLIENTS-1))); do
    echo "Starting client $i..."
    python -m model_side.federated.client_dp \
        --server_address 127.0.0.1:8080 \
        --client_id $i \
        --batch_size $BATCH_SIZE \
        --num_classes 2 \
        --target_epsilon $EPSILON \
        --target_delta 1e-5 \
        --max_grad_norm 1.0 &
    sleep 2
done

# Wait for server to finish
wait $SERVER_PID

echo ""
echo "Training complete!"
echo "Check results in results/fl_dp/"
echo "Models saved in models/fl_dp/"
