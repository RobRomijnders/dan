#!/bin/bash
sudo ldconfig /usr/local/cuda-9.0/lib64

# Promp user for full or small mode
while [ -z "$mode" ]; do
  read -p 'Training mode [full/small]: ' answer
  case "$answer" in
    full) mode="full" ;;
    small) mode="small" ;;
    *) echo "Please choose mode [full/small], not $answer" ;;
  esac
done

direc_exp1=uda-drop-lambda-conf-

# set the variables depending on the mode
if [ $mode = full ]
then
  num_samples_cityscapes=2975
  num_samples_gta5=3000
  num_epochs=17
  echo $num_samples_cityscapes
elif [ $mode = small ]
then
  num_samples_cityscapes=10
  num_samples_gta5=10
  num_epochs=1
else
  echo "Allowed training mode only full/small, not $mode"
fi

source set_env_names.sh

echo "Start time"
date +"%T"
source "$tf_env_activate"

for lam_conf in 0.0001 0.0003 0.0005 0.0008 0.0010 ; do
    echo "$lam_conf"
    echo "training \n"
    sudo ldconfig /usr/local/cuda-9.0/lib64
    nvidia-smi -q -d temperature | grep GPU
    python ../train.py "$log_direc$mode$direc_exp1$lam_conf" "$data_direc/cityscapes/tfrecords_384/trainFine.tfrecords" $num_samples_cityscapes "$data_direc/problem_uda.json" --Nb 2 --init_ckpt_path "$data_direc/init_ckpt/resnet_v1_50_official.ckpt" --Ne $num_epochs --tfrecords_path_add "$data_direc/mapillary/tfrecords_384/trainFine.tfrecords" --additional_problem_def_path "$data_direc/problem_uda_vistas1.json" --switch_train_op True --unsupervised_adaptation True --dom_class_type 1 --custom_normalization_mode custombatch --lambda_conf "$lam_conf" --ema_decay -0.95 --ramp_start -1
done


echo "End time"
date +"%T"
nvidia-smi -q -d temperature | grep GPU
deactivate

# Also run the evaluation
echo
echo "Also do evaluation"
echo "$mode" | ./evaluate_many.sh

#echo
#echo "Also do extraction"
#echo "$mode" | ./extract_many.sh

#if [ $mode = full ]
#then
#    # And shutdown after overnight training
#    sudo shutdown +20
#fi


