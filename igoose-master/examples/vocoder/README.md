# Vocoder

## [sr_22050_nfft_512_hop_128_mels_80_cnn1d](conf/sr_22050_nfft_512_hop_128_mels_80_cnn1d)

* sample rate: 22050
* `n_fft`: 512
* `hop_length`: 128
* `n_mels`: 80

### Experiments

#### [debug](conf/sr_22050_nfft_512_hop_128_mels_80_cnn1d/experiment/debug.yaml)

```cmd
CUDA_VISIBLE_DEVICES=0 NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 python train.py --config-path=conf/sr_22050_nfft_512_hop_128_mels_80_cnn1d +experiment=debug
```

Train on speeches from 1 female and 1 male. Note that it may not converge as the
training batch size is small.
