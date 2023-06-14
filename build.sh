#!/bin/bash

docker build . -f Dockerfile -t object-removal:latest
docker tag object-removal:latest eu.gcr.io/res-interns/silvan-object-removal:latest
docker push eu.gcr.io/res-interns/silvan-object-removal:latest