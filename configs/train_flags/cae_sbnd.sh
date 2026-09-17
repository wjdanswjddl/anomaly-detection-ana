# Flags for training the CAE anomaly detector on SBND images.
# Usage: source model_flags_CAE_SBND.sh && python scripts/autoencoder_train.py $AE_TRAIN_FLAGS
#
# The data is raw SBND h5 files in dCache, streamed over xrootd. The data
# flags point at file lists (one root:// URL per line) rather than a
# directory, and the training process must run with the xrootd POSIX
# preload so h5py can open root:// URLs:
#   LD_PRELOAD=/usr/lib64/libXrdPosixPreload.so HDF5_USE_FILE_LOCKING=FALSE \
#       python scripts/autoencoder_train.py $AE_TRAIN_FLAGS
# Reading requires a valid SBND bearer token (default /tmp/bt_u$(id -u)).
# Regenerate the file lists with:
#   xrdfs root://fndcadoor.fnal.gov:1094 ls -R /pnfs/fnal.gov/usr/sbnd/scratch/users/munjung/v10_06_00/raw/h5 \
#     | grep '\.h5$' | sed 's|^|root://fndcadoor.fnal.gov:1094/|'
export MODEL_FLAGS="--ae_type cae --image_size 512 --in_channels 1 --ae_hidden_dims 32,64,128,256,512,512,512 --ae_latent_dim 512 --spatial_latent True --ssim_weight 0.0 --final_activation tanh"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 16 --microbatch 16 --weight_batches False --weight_pixels False --weight_decay 0.01 --use_fp16 False --save_interval 1000 --plot_interval 2000"
export DATADIR="--data_dir /scratch/7DayLifetime/gputnam/training-SBND-CAE/iterA/filelists/train.txt --validation_dir /scratch/7DayLifetime/gputnam/training-SBND-CAE/iterA/filelists/validation.txt --charge_scale 1"
export AE_TRAIN_FLAGS="$DATADIR $MODEL_FLAGS $TRAIN_FLAGS"
