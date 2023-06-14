#!/bin/bash

docker build . -f Dockerfile -t lama:latest
docker tag lama:latest eu.gcr.io/res-interns/lama:latest
docker push eu.gcr.io/res-interns/lama:latest