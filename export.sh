#!/usr/bin/env bash

./build.sh

docker save bondbidhie2023_algorithm | gzip -c > bondbidhie2023_algorithm.tar.gz
