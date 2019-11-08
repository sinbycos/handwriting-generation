#! /bin/bash

docker build -t "sol:v1" .
docker run -it -p 8888:8888 sol:v1
