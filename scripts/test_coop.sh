#!/bin/bash

data_root='/home/chathurya/TPT/data'
coop_weight='/home/chathurya/TPT/data/to_gdrive/vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed2/prompt_learner/model.pth.tar-50'
testsets=$1
# arch=RN50
arch=ViT-B/16
bs=64
# ctx_init=a_photo_of_a
# ctx_inits=("a" "a_toy" "this_is_a_photo_of" "there_are_[CLS]_objects" "the_nearest_shape_in_this_image_is")
run_type=tpt_atpt
# lambda_term=-24.0
lambda_terms=(25.0  30.0 35.0 40.0 45.0)
tau_term=0.99999

# echo "Testing with lambda_term = $lambda_term"
# python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
# -a ${arch} -b ${bs} --gpu 1 \
# --tpt --ctx_init ${ctx_init} --run_type ${run_type} --lambda_term ${lambda_term}
# echo "Completed testing for lambda_term = $lambda_term"

# for ctx_init in "${ctx_inits[@]}"; do
#     echo "Running with ctx_init = $ctx_init"
#     python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
#     -a ${arch} -b ${bs} --gpu 0 \
#     --tpt --ctx_init ${ctx_init} --run_type ${run_type} --lambda_term ${lambda_term}
#     echo "Completed for ctx_init = $ctx_init"
# done

# for lambda_term in "${lambda_terms[@]}"; do
# echo "Testing with lambda_term = $lambda_term"
# python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
# -a ${arch} -b ${bs} --gpu 0 \
# --tpt --ctx_init ${ctx_init} --run_type ${run_type} --lambda_term ${lambda_term}
# echo "Completed testing for lambda_term = $lambda_term"
# done

for lambda_term in "${lambda_terms[@]}"; do
echo "Testing with lambda_term = $lambda_term"
python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 1 \
--tpt --load ${coop_weight} --run_type ${run_type} --lambda_term ${lambda_term} --tau_term ${tau_term}
echo "Completed testing for lambda_term = $lambda_term"
done