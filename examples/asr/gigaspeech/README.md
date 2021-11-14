# Records

1. AED: Conformer encoder + Transformer decoder

    * configuration: `1a.yaml`
    * commit: `a5042e836cbb290b63a8347dd939461af33a20c2`

    | LM | lm weight | beam | CTC weight | Dev (SUB/INS/DEL) | Test (SUB/INS/DEL) |
    |:---:|:---:|:---:|:---:|:---:|:---:|
    | - | 0 | 8 | 1 | 14.00% (10590/3731/3772) | 13.74% (34345/10916/8638) |
    | - | 0 | 8 | 0 | 12.84% (9270/3628/3693) | 12.57% (30417/10254/8625) |
    | - | 0 | 8 | 0.2 | 12.73% (9232/3464/3760) | 12.41% (30335/9760/8609) |
