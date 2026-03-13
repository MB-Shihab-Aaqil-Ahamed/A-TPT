#!/bin/bash

data_root='/home/chathurya/TPT/data'
testsets=$1
arch=RN50
#arch=ViT-B/16

bs=64
ctx_init=a_photo_of_a
# ctx_inits=("a" "a_toy" "this_is_a_photo_of" "there_are_[CLS]_objects" "the_nearest_shape_in_this_image_is")
run_type=tpt_atpt
# lambda_term=80.0
#lambda_terms=(35 40 45 50 65 70 75)
#lambda_terms=(290 best 490, 690 , 890, 990 so far 4.14)
lambda_terms=(90)
#lambda_terms=(-30 -10 -0.1 0.1 10 30)

tau_term=0.99999

# echo "Testing with lambda_term = $lambda_term"
# python ./atpt_classification.py ${data_root} --test_sets ${testsets} \
# -a ${arch} -b ${bs} --gpu 0 \
# --tpt --ctx_init ${ctx_init} --run_type ${run_type} --lambda_term ${lambda_term} --tau_term ${tau_term}
# echo "Completed testing for lambda_term = $lambda_term"

# for ctx_init in "${ctx_inits[@]}"; do
#     echo "Running with ctx_init = $ctx_init"
#     python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
#     -a ${arch} -b ${bs} --gpu 0 \
#     --tpt --ctx_init ${ctx_init} --run_type ${run_type} --lambda_term ${lambda_term} --tau_term ${tau_term}
#     echo "Completed for ctx_init = $ctx_init"
# done

for lambda_term in "${lambda_terms[@]}"; do
echo "Testing with lambda_term = $lambda_term"
python ./atpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 \
--tpt --ctx_init ${ctx_init} --run_type ${run_type} --lambda_term ${lambda_term} --tau_term ${tau_term}
echo "Completed testing for lambda_term = $lambda_term"
done