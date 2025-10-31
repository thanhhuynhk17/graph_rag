#!/bin/bash

echo "killing old docker processes"
docker compose down

echo "building docker containers"
docker compose up neo4j --build -d

echo "building docker containers"
docker compose logs -f