
* 1a.yaml: Conv1d + RNN encoder & RNN decoder

    | LM | lm weight | beam | EOS | Dev (SUB/INS/DEL) | Test (SUB/INS/DEL) |
    |:---:|:---:|:---:|:---:|:---:|:---:|
    | - | 0 | 0 | 16 | 8.02% (15936/265/274) | 9.20% (9199/245/194) |
    | 5gram | 0.6 | 16 | 1 | 7.36% (14570/253/284) | 8.21% (8178/225/194) |
