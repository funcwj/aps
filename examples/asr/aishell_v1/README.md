# Records

1. AED: RNN encoder + RNN decoder

    * configuration: `1a.yaml`
    * commit: `c35d2ac9243024736b827c383ff2692c732e25e3`

    | LM | lm weight | beam | CTC weight | Dev (SUB/INS/DEL) | Test (SUB/INS/DEL) |
    |:---:|:---:|:---:|:---:|:---:|:---:|
    | - | 0 | 16 | 0 | 6.78% (13080/638/213) | 7.55% (7479/298/136) |
    | - | 0 | 16 | 0.2 | 6.63% (13135/266/207) | 7.42% (7432/218/125) |
    | - | 0 | 16 | 1.0 | 8.76% (17390/350/238) | 9.90% (9927/279/168) |
    | RNN | 0.2 | 16 | 0.2 | 6.35% (12555/216/265) | 6.94% (6912/200/157) |

2. AED: Transformer encoder + Transformer decoder

    * configuration: `1b.yaml`
    * commit: `5b015c4945ee0468d1f00b35924eb36e13f2bc65`

    | LM | lm weight | beam | CTC weight | Dev (SUB/INS/DEL) | Test (SUB/INS/DEL) |
    |:---:|:---:|:---:|:---:|:---:|:---:|
    | - | 0 | 8 | 0 | 5.71% (11337/210/188) | 6.53% (6553/186/105) |
    | - | 0 | 8 | 0.2 | 4.79% (9471/199/168) | 5.51% (5409/286/77) |
    | - | 0 | 8 | 0.5 | 4.91% (9732/179/166) | 5.60% (5589/202/81) |

3. Transducer: Transformer encoder + RNN decoder

    * configuration: `1c.yaml`
    * commit: `5b015c4945ee0468d1f00b35924eb36e13f2bc65`

    | LM | lm weight | beam | Dev (SUB/INS/DEL) | Test (SUB/INS/DEL) |
    |:---:|:---:|:---:|:---:|:---:|
    | - | 0 | 1 | 9.64% (17371/2207/215) | 10.47% (9568/1303/93) |
    | - | 0 | 4 | 8.40% (15678/1348/218) | 9.20% (8752/787/103) |
    | - | 0 | 8 | 8.36% (15670/1285/215) | 9.11% (8669/765/106) |

4. Transducer: RNN encoder + RNN decoder

    * configuration: `1d.yaml`
    * commit: `1b074e07c77cc611a55499c26877d217933d2b1d`

    | LM | lm weight | beam | Dev (SUB/INS/DEL) | Test (SUB/INS/DEL) |
    |:---:|:---:|:---:|:---:|:---:|
    | - | 0 | 1 | 9.64% (18621/909/256) | 10.80% (10609/562/141) |
    | - | 0 | 4 | 8.85% (17274/619/278) | 9.78% (9699/395/155) |
    | - | 0 | 8 | 8.84% (17250/611/281) | 9.76% (9679/390/153) |
