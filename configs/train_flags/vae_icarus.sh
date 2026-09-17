# Flags for training the VAE anomaly detector on ICARUS images.
# Usage: source model_flags_VAE_ICARUS.sh && python scripts/autoencoder_train.py $AE_TRAIN_FLAGS
# The DNN-ROI directory holds ICARUS deconvolved-signal .h5 images
# (event_N/deconvolved_signal), tiled to image_size by ImageDataset.
export MODEL_FLAGS="--ae_type vae --image_size 512 --in_channels 1 --ae_hidden_dims 32,64,128,256,512,512,512 --ae_latent_dim 512 --kld_weight 1e-4 --final_activation tanh"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 16 --microbatch 4 --weight_batches False --weight_pixels False --weight_decay 0.01 --use_fp16 False --save_interval 1000 --plot_interval 2000"
export DATADIR="--data_dir /exp/sbnd/data/users/gputnam/DNN-ROI-images/ --validation_dir /exp/sbnd/data/users/gputnam/DNN-ROI-images/ --charge_scale 1"
# Alternative ICARUS dataset (fill in and uncomment):
# export DATADIR="--data_dir /path/to/icarus/data/ --validation_dir /path/to/icarus/data/ --charge_scale 1"
export AE_TRAIN_FLAGS="$DATADIR $MODEL_FLAGS $TRAIN_FLAGS"
