# Run hand tracking for all the folders in the DATA_DIR
# export DATA_DIR=/mnt/scratch/experimentals/first_perspective/20250524_jian_home/
# export DATA_DIR=/mnt/scratch/experimentals/first_perspective/20250524_haonan_home
export DATA_DIR=/home/haonan/Project/hamer/example_data

for folder in $DATA_DIR/*; do
    if [ -d "$folder" ] && [ "$folder" != "$DATA_DIR/calibration" ]; then
        echo "Processing folder: $folder"
        export DATA_SEQ=$folder

        # Hand tracking
        echo "Start hand tracking"
        python handtrack_from_video.py --raw_data_folder $DATA_SEQ --redo_raw_process
    fi
done