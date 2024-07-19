#!/bin/bash

# Start Docker daemon in the background
dockerd &

# Wait for Docker daemon to start
max_attempts=10
attempt=1
while (! docker info > /dev/null 2>&1); do
    echo "Waiting for Docker to start..."
    sleep 1
    if [ $attempt -eq $max_attempts ]; then
        echo "Maximum number of attempts reached. Exiting..."
        exit 1
    fi
    attempt=$((attempt+1))
done

# Execute the main process specified as CMD in the Dockerfile
python3 -u run_algorithm.py "$@"
