#!/bin/bash

dtype=dns
for seed in $(seq 0 9); do
    python preprocess/generate_test_noise_pair.py ${dtype} ${seed} || exit 0;
    python preprocess/generate_test_dns_pair.py ${dtype} ${seed} || exit 0;
done
