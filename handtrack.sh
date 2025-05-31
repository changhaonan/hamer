# Run hand tracking for all the folders in the DATA_DIR
# export DATA_DIR=/mnt/scratch/experimentals/first_perspective/20250524_jian_home/
# export DATA_DIR=/mnt/scratch/experimentals/first_perspective/20250524_haonan_home
export DATA_DIR=/home/haonan/Project/hamer/example_data

for folder in $DATA_DIR/*; do
    if [ -d "$folder" ] && [ "$folder" != "$DATA_DIR/calibration" ] && [ "$folder" != "$DATA_DIR/test_001" ]; then
        echo "Processing folder: $folder"
        export DATA_SEQ=$folder

        # Hand tracking
        echo "Start hand tracking"
        python handtrack_from_video.py --video_path $DATA_SEQ/raw_data/camera/camera_0.mp4 --out_path $DATA_SEQ/handtrack/camera_0_hamer.mp4
    fi
done