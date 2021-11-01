# Records

1. AED: Conformer encoder + Transformer decoder

    * configuration: `1a.yaml`
    * commit: `54c7f48bdadd7f495cbdfde6dee8e023a483205c`

    | LM | lm weight | beam | CTC weight | Dev (SUB/INS/DEL) | Test (SUB/INS/DEL) |
    |:---:|:---:|:---:|:---:|:---:|:---:|
    | - | 0 | 8 | 0.2 | 14.84% (11466/3728/3986) | 14.43% (36655/10637/9305) |
