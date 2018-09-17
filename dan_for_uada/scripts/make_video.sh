#!/usr/bin/env bash
sudo ldconfig /usr/local/cuda-9.0/lib64

. set_env_names.sh

# Step in the virtual environment for the project
source "$tf_env_activate"

#base_movie_dir=/home/mps/Documents/rob/datasets/eindhoven/city/
base_movie_dir=/home/mps/Documents/rob/datasets/cityscapes/demo_video/raw_images/stuttgart_00/

echo "Start on source only"
#python ../predict.py /hdd/logs_overnight_training/newer/overnight_0509/fullgta-1 /home/mps/Documents/rob/datasets/problem_uda_apollo.json $base_movie_dir --results_dir "$base_movie_dir/predictions_gta5/"

echo "Start on Adapted"
#python ../predict.py /hdd/logs_overnight_training/newer/overnight_0516/fulluda-drop-lambda-conf-0.01 /home/mps/Documents/rob/datasets/problem_uda_apollo.json $base_movie_dir --results_dir "$base_movie_dir/predictions_uada1/"
#python ../predict.py /hdd/logs_overnight_training/newer/overnight_0517/fulluda-drop-lambda-conf-0.001 /home/mps/Documents/rob/datasets/problem_uda_apollo.json $base_movie_dir --results_dir "$base_movie_dir/predictions_uada1/"

#echo "Start making the movie"
python ../misc/plotting/make_video.py $base_movie_dir

if [[ $base_movie_dir = *"cityscape"* ]]; then
  convert -delay 3 -loop 0 "$base_movie_dir/movie/*.png" "$base_movie_dir/out/video2.gif"
else
  ffmpeg -r 4 -f image2 -i "$base_movie_dir/movie/%06d.png" -vcodec libx264 -pix_fmt yuv420p "$base_movie_dir/out/output.mp4"
fi

deactivate
