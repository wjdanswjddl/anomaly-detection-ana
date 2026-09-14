export TRAIN_FLAGS="--lr 1e-4 --batch_size 20"
export CLASSIFIER_FLAGS="--image_size 512 --classifier_attention_resolutions 16,32 --classifier_depth 4 --classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm False"
export DATADIR="--data_dir /scratch/7DayLifetime/munjung/anomaly-detection/npz --charge_scale 1"
export CLASSIFIER_TRAIN_FLAGS="$DATADIR $CLASSIFIER_FLAGS $TRAIN_FLAGS"

