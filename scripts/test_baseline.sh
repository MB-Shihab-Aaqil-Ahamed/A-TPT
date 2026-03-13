#!/bin/bash

data_root='/home/chathurya/TPT/data'
testsets=$1
arch=RN50
# arch=ViT-B/16
bs=64
# ctx_init=a_photo_of_a
ctx_inits=(a a_toy this_is_a_photo_of "there_are_[CLS]_objects" the_nearest_shape_in_this_image_is)
run_type=baseline

for ctx_init in "${ctx_inits[@]}"; do
echo "Running with ctx_init = $ctx_init"
python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 \
--tpt --ctx_init ${ctx_init} --run_type ${run_type}
echo "Completed for ctx_init = $ctx_init"
done