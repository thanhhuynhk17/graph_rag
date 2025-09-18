#!/bin/bash

echo "killing old docker processes"
docker compose down --rmi all -v --remove-orphans

echo "building docker containers"
docker compose up --build -d