#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to wait for the database to be ready
function wait_for_db() {
    echo "Waiting for database ($DB_HOST)..."
    # Try connecting to the port; retry every 1 second
    while ! nc -z $DB_HOST $DB_PORT; do
      sleep 1
    done
    echo "Database is up and running!"
}

# Default values if environment variables aren't set
export DB_HOST=${DB_HOST:-db}
export DB_PORT=${DB_PORT:-5432}

# 1. Wait for Postgres
wait_for_db

# 2. Run migrations (Only once per deployment)
# We check if we are the 'web' container to avoid running migrations multiple times
if [ "$CONTAINER_TYPE" = "web" ]; then
    echo "Applying database migrations..."
    python manage.py migrate --noinput
    
    echo "Collecting static files..."
    python manage.py collectstatic --noinput
fi

# 3. Execute the CMD from docker-compose
echo "Starting $CONTAINER_TYPE..."
exec "$@"
