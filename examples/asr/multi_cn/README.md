# Records

1. AED: Conformer encoder + Transformer decoder

    - configuration: `1a.yaml`
    - commit: `459a17463ac0d4fa59c014bac6f07ff863ffe065`

    - ctc weight = 0.5 & beam size = 8 & without aishell2 data
        | Dataset | dev (SUB/INS/DEL) | test (SUB/INS/DEL) |
        |:---:|:---:|:---:|
        | aishell | 4.15% (8197/156/159) | 4.59% (4603/127/83) |
        | aidatatang | 3.99% (7635/1367/362) | 4,60% (18111/2637/806) |
        | magicdata | 3.64% (3578/394/279) | 2.64% (5391/603/314) |
        | thchs | 0.51% (111/17/20) | 12.91% (10109/273/96) |

    - ctc weight = 0.2 & beam size = 8 & without aishell2 data
        | Dataset | dev (SUB/INS/DEL) | test (SUB/INS/DEL) |
        |:---:|:---:|:---:|
        | aishell | 3.99% (7855/166/168) | 4.45% (4423/149/87) |
        | aidatatang | 3.80% (7243/1279/384) | 4.40% (17276/2501/876) |
        | magicdata | 3.58% (3410/476/300) | 2.57% (5146/663/347) |
        | thchs | 0.43% (92/16/17) | 13.02% (9985/478/101) |

    - ctc weight = 0.2 & beam size = 8 & with aishell2 data
        | Dataset | dev (SUB/INS/DEL) | test (SUB/INS/DEL) |
        |:---:|:---:|:---:|
        | aishell | 1.64% (3123/128/128) | 1.85% (1744/145/44) |
        | aidatatang | 3.94% (7637/1182/410) | 4.53% (18084/2228/950) |
        | magicdata | 4.27% (4000/640/345) | 3.18% (6283/909/411) |
        | thchs | 0.87% (189/46/20) | 10.37% (7768/541/102) |

    - ctc weight = 0.2 & beam size = 8 & with aishell2 data & RNN LM (weight 0.2)
        | Dataset | dev (SUB/INS/DEL) | test (SUB/INS/DEL) |
        |:---:|:---:|:---:|
        | aishell | 1.33% (2469/149/113) | 1.44% (1303/173/31) |
        | aidatatang | 3.48% (6586/1231/334) | 4.02% (15683/2379/801) |
        | magicdata | 3.45% (3032/731/265) | 2.71% (5081/1039/360) |
        | thchs | 0.40% (77/29/11) | 10.76% (7844/787/102) |
