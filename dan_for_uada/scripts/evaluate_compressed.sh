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
  numsamples_bdd=700
  numsamples_apollo=1000
  final_ckpts=2
  echo $numsamples
elif [ $mode = small ]
then
  numsamples=6
  numsamples_bdd=6
  numsamples_apollo=6
  final_ckpts=1
else
  echo "Allowed evaluation mode only full/small, not $mode"
fi

source set_env_names.sh

# Step in the virtual environment for the project
source "$tf_env_activate"
log_direc=/hdd/logs_overnight_training/newer/overnight_0509/
out_direc=/hdd/dropbox/Dropbox/grad/results/conf_matrices/results_from_0509_apollo_bdd_full/

# Loop over all directories
#
for direc in $(find "$log_direc" -maxdepth 1 -mindepth 1 -type d) ; do
    sudo ldconfig /usr/local/cuda-9.0/lib64
    echo "$(basename "$direc")"

#    # Evaluate on Cityscapes
#    python ../evaluate.py "$direc" "$data_direc/cityscapes/tfrecords_384/valFine.tfrecords" $numsamples --Nb 4 "$data_direc/problem_uda.json" --confusion_matrix_filename "$out_direc$(basename "$direc")__cityscapes.txt" --eval_all_ckpts $final_ckpts --restore_emas
#    # Evaluate on GTA5
#    python ../evaluate.py "$direc" "$data_direc/gta5/tfrecords_384/valFine.tfrecords" $numsamples --Nb 4 "$data_direc/problem_uda.json" --confusion_matrix_filename "$out_direc$(basename "$direc")__gta5.txt" --eval_all_ckpts $final_ckpts --restore_emas
#    # Evaluate on Mapillary
#    python ../evaluate.py "$direc" "$data_direc/mapillary/tfrecords_384/valFine.tfrecords" 8000 --Nb 4 "$data_direc/problem_uda_vistas1.json" --confusion_matrix_filename "$out_direc$(basename "$direc")__mapillary.txt" --eval_all_ckpts $final_ckpts --restore_emas
    # Evaluate on Apollo
    python ../evaluate.py "$direc" "$data_direc/apollo/tfrecords_384/valFine.tfrecords" $numsamples_apollo --Nb 4 "$data_direc/problem_uda_apollo.json" --confusion_matrix_filename "$out_direc$(basename "$direc")__apollo.txt" --eval_all_ckpts $final_ckpts
    # Evaluate on BDD
    python ../evaluate.py "$direc" "$data_direc/bdd/tfrecords_384/valFine.tfrecords" $numsamples_bdd --Nb 4 "$data_direc/problem_uda_bdd.json" --confusion_matrix_filename "$out_direc$(basename "$direc")__bdd.txt" --eval_all_ckpts $final_ckpts
done

## Make a summary of all the confusion matrices
python ././print_mean_iou.py "$out_direc"



# Do the same for UADA2
log_direc=/hdd/logs_overnight_training/newer/overnight_0528/
out_direc=/hdd/dropbox/Dropbox/grad/results/conf_matrices/results_from_0528_apollo_bdd_full/

# Loop over all directories
#
for direc in $(find "$log_direc" -maxdepth 1 -mindepth 1 -type d) ; do
    sudo ldconfig /usr/local/cuda-9.0/lib64
    echo "$(basename "$direc")"

#    # Evaluate on Cityscapes
#    python ../evaluate.py "$direc" "$data_direc/cityscapes/tfrecords_384/valFine.tfrecords" $numsamples --Nb 4 "$data_direc/problem_uda.json" --confusion_matrix_filename "$out_direc$(basename "$direc")__cityscapes.txt" --eval_all_ckpts $final_ckpts --restore_emas
#    # Evaluate on GTA5
#    python ../evaluate.py "$direc" "$data_direc/gta5/tfrecords_384/valFine.tfrecords" $numsamples --Nb 4 "$data_direc/problem_uda.json" --confusion_matrix_filename "$out_direc$(basename "$direc")__gta5.txt" --eval_all_ckpts $final_ckpts --restore_emas
#    # Evaluate on Mapillary
#    python ../evaluate.py "$direc" "$data_direc/mapillary/tfrecords_384/valFine.tfrecords" 8000 --Nb 4 "$data_direc/problem_uda_vistas1.json" --confusion_matrix_filename "$out_direc$(basename "$direc")__mapillary.txt" --eval_all_ckpts $final_ckpts --restore_emas
    # Evaluate on Apollo
    python ../evaluate.py "$direc" "$data_direc/apollo/tfrecords_384/valFine.tfrecords" $numsamples_apollo --Nb 4 "$data_direc/problem_uda_apollo.json" --confusion_matrix_filename "$out_direc$(basename "$direc")__apollo.txt" --eval_all_ckpts $final_ckpts
    # Evaluate on BDD
    python ../evaluate.py "$direc" "$data_direc/bdd/tfrecords_384/valFine.tfrecords" $numsamples_bdd --Nb 4 "$data_direc/problem_uda_bdd.json" --confusion_matrix_filename "$out_direc$(basename "$direc")__bdd.txt" --eval_all_ckpts $final_ckpts
done

## Make a summary of all the confusion matrices
python ././print_mean_iou.py "$out_direc"

deactivate