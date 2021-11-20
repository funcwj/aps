# Records

1. 2spk-C-16k-min TCN++

    - configuration: `1a_2spk_c_16k_min.yaml`
    - commit: `97e6c6368654cfd7d951687dc84ca6d6341d96ca`
    - training data: train-360, min, clean mixture

    | Metric | Mode | Dev | Test |
    |:---:|:---:|:---:|:---:|
    | Si-SNR | max | 15.625 | 15.277 |
    | Si-SNRi | max | 15.626 | 15.278 |
    | Si-SNR | min | 14.868 | 14.566 |
    | Si-SNRi | min | 14.869 | 14.565 |

2. 2spk-N-16k-min TCN++

    - configuration: `1b_2spk_c_16k_min.yaml`
    - commit: `97e6c6368654cfd7d951687dc84ca6d6341d96ca`
    - training data: train-360, min, noisy mixture

    | Metric | Mode | Dev | Test |
    |:---:|:---:|:---:|:---:|
    | Si-SNR | max | 9.761 | 9.525 |
    | Si-SNRi | max | 12.105 | 11.997 |
    | Si-SNR | min | 9.658 | 9.395 |
    | Si-SNRi | min | 11.470 | 11.330 |
