#!/bin/bash

Xvfb :99 -screen 0 1280x720x24 &
export DISPLAY=:99
stellarium --server &
sleep 5 curl -s http://localhost:8090/api/main/status > /dev/null if [ $? -eq 0 ]; then echo "Stellarium API is running on port 8090" else echo "Failed to start Stellarium API" exit 1 fi

tail -f /dev/null
