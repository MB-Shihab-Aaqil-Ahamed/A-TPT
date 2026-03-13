#!/bin/bash

data_root='/home/chathurya/TPT/data'
testsets=$1
# arch=RN50
arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a
run_type=tpt_atpt
# lambda_term=-14.5
lambda_terms=(-10.0)
tau_term=0.99999 

for lambda_term in "${lambda_terms[@]}"; do
echo "Testing with lambda_term = $lambda_term"
python ./atpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 1 \
--tpt --ctx_init ${ctx_init} --run_type ${run_type} --I_augmix --lambda_term ${lambda_term} --tau_term ${tau_term}
echo "Completed testing for lambda_term = $lambda_term"
done