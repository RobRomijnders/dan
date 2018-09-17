#!/bin/bash

if [[ -p /dev/stdin ]]
then
  mode=$STDIN
fi

# Promp user for full or small mode
while [ -z "$mode" ]; do
  read -p 'Evaluation mode [full/small]: ' answer
  case "$answer" in
    full) mode="full" ;;
    small) mode="small" ;;
    *) echo "Please choose mode [full/small], not $answer" ;;
  esac
done


# set the variables depending on the mode
if [ $mode = full ]
then
  numsamples=500
  final_ckpts=3
  echo $numsamples
elif [ $mode = small ]
then
  numsamples=6
  final_ckpts=1
else
  echo "Allowed evaluation mode only full/small, not $mode"
fi

source set_env_names.sh

# Step in the virtual environment for the project
source "$tf_env_activate"

# Loop over all directories
#
for direc in $(find "$log_direc" -maxdepth 1 -mindepth 1 -type d) ; do
    sudo ldconfig /usr/local/cuda-9.0/lib64
    echo "$(basename "$direc")"
    # Evaluate on Cityscapes
    python ../evaluate.py "$direc" "$data_direc/cityscapes/tfrecords_384/valFine.tfrecords" $numsamples --Nb 4 "$data_direc/problem_uda.json" --confusion_matrix_filename "$out_direc$(basename "$direc")__cityscapes.txt" --eval_all_ckpts $final_ckpts --restore_emas
    # Evaluate on GTA5
    python ../evaluate.py "$direc" "$data_direc/gta5/tfrecords_384/valFine.tfrecords" $numsamples --Nb 4 "$data_direc/problem_uda.json" --confusion_matrix_filename "$out_direc$(basename "$direc")__gta5.txt" --eval_all_ckpts $final_ckpts --restore_emas
    # Evaluate on Mapillary
    python ../evaluate.py "$direc" "$data_direc/mapillary/tfrecords_384/valFine.tfrecords" $numsamples --Nb 4 "$data_direc/problem_uda_vistas1.json" --confusion_matrix_filename "$out_direc$(basename "$direc")__mapillary.txt" --eval_all_ckpts $final_ckpts --restore_emas
    # Evaluate on Apollo
    python ../evaluate.py "$direc" "$data_direc/apollo/tfrecords_384/valFine.tfrecords" $numsamples --Nb 4 "$data_direc/problem_uda_apollo.json" --confusion_matrix_filename "$out_direc$(basename "$direc")__apollo.txt" --eval_all_ckpts $final_ckpts --restore_emas
done

# Make a summary of all the confusion matrices
python ././print_mean_iou.py "$out_direc"

deactivate

