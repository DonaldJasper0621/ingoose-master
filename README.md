# igoose-master

# Voice Conversion

## [kmic_demo_2023q2_sr_22050_nfft_512_hop_128](conf/kmic_demo_2023q2_sr_22050_nfft_512_hop_128)

* sample rate: 22050
* `n_fft`: 512
* mel `hop_length`: 128
* `n_mels`: 80

### Experiments

#### [debug](conf/kmic_demo_2023q2_sr_22050_nfft_512_hop_128/experiment/debug.yaml)

To train on speeches from 1 speakers, run

```cmd
CUDA_VISIBLE_DEVICES=0 NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 python train.py --config-path=conf/kmic_demo_2023q2_sr_22050_nfft_512_hop_128 +experiment=debug
```

Note that it may not converge as the training batch size is small.

To pre-compute features (f0, loudness & frame content) of audios, run

```cmd
CUDA_VISIBLE_DEVICES=0 NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 python precompute_features.py --config-path=conf/kmic_demo_2023q2_sr_22050_nfft_512_hop_128 audio_root_folder=AUDIO_FOLDER
```
