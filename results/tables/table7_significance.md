| question                                      | metric                |      A |      B |    diff |   ci_low |   ci_high |     p |     n | winner   |
|:----------------------------------------------|:----------------------|-------:|-------:|--------:|---------:|----------:|------:|------:|:---------|
| RQ1 small fine-tuned model vs LLM judge       | accuracy              | 0.643  | 0.623  |  0.02   |  -0.014  |    0.055  | 0.252 |  1000 | n.s.     |
| RQ1 small fine-tuned model vs LLM judge       | ece                   | 0.0638 | 0.3154 | -0.2517 |  -0.2808 |   -0.2125 | 0     |  1000 | A        |
| RQ1 same, both symmetrized                    | accuracy              | 0.65   | 0.634  |  0.016  |  -0.021  |    0.049  | 0.411 |  1000 | n.s.     |
| RQ1 DeBERTa vs BERT                           | accuracy              | 0.6577 | 0.6363 |  0.0215 |   0.0092 |    0.0333 | 0     |  6194 | A        |
| RQ2 symmetrization (BERT)                     | accuracy              | 0.6464 | 0.6363 |  0.0102 |   0.0013 |    0.0182 | 0.027 |  6194 | A        |
| RQ2 symmetrization (DeBERTa)                  | accuracy              | 0.6682 | 0.6577 |  0.0105 |   0.0029 |    0.0184 | 0.008 |  6194 | A        |
| RQ2 swap augmentation (BERT)                  | accuracy              | 0.6445 | 0.6363 |  0.0082 |  -0.0018 |    0.0179 | 0.108 |  6194 | n.s.     |
| Bi-encoder vs cross-encoder (BERT)            | accuracy              | 0.6459 | 0.6363 |  0.0097 |   0.0003 |    0.0192 | 0.046 |  6194 | A        |
| Ablation: no prompt vs prompt (BERT)          | accuracy              | 0.6371 | 0.6363 |  0.0008 |  -0.0073 |    0.0092 | 0.866 |  6194 | n.s.     |
| RQ3 soft vs hard (BiLSTM)                     | accuracy              | 0.6664 | 0.6629 |  0.0035 |  -0.0016 |    0.0084 | 0.165 | 18406 | n.s.     |
| RQ3 soft vs hard (BiLSTM)                     | ece                   | 0.0207 | 0.0676 | -0.0468 |  -0.0556 |   -0.0374 | 0     | 18406 | A        |
| RQ3 soft vs hard (BiLSTM)                     | kl_to_human           | 0.1256 | 0.206  | -0.0804 |  -0.0832 |   -0.0777 | 0     | 18406 | A        |
| RQ3 soft vs hard (BiLSTM)                     | ece_high_disagreement | 0.0328 | 0.1018 | -0.069  |  -0.0806 |   -0.0555 | 0     | 18406 | A        |
| RQ3 soft vs hard (BiLSTM)                     | kl_high_disagreement  | 0.063  | 0.1825 | -0.1195 |  -0.1251 |   -0.1143 | 0     | 18406 | A        |
| RQ3 soft vs hard (BERT)                       | accuracy              | 0.6866 | 0.6795 |  0.0071 |   0.0017 |    0.0122 | 0.015 | 18406 | A        |
| RQ3 soft vs hard (BERT)                       | ece                   | 0.0209 | 0.0621 | -0.0412 |  -0.0471 |   -0.0317 | 0     | 18406 | A        |
| RQ3 soft vs hard (BERT)                       | kl_to_human           | 0.1311 | 0.2045 | -0.0734 |  -0.0761 |   -0.0708 | 0     | 18406 | A        |
| RQ3 soft vs hard (BERT)                       | ece_high_disagreement | 0.0461 | 0.0984 | -0.0523 |  -0.0618 |   -0.0373 | 0     | 18406 | A        |
| RQ3 soft vs hard (BERT)                       | kl_high_disagreement  | 0.0863 | 0.1895 | -0.1032 |  -0.1086 |   -0.0981 | 0     | 18406 | A        |
| RQ3 soft vs hard (DeBERTa)                    | accuracy              | 0.7011 | 0.6992 |  0.002  |  -0.0024 |    0.0062 | 0.371 | 18406 | n.s.     |
| RQ3 soft vs hard (DeBERTa)                    | ece                   | 0.0238 | 0.1085 | -0.0848 |  -0.0909 |   -0.0764 | 0     | 18406 | A        |
| RQ3 soft vs hard (DeBERTa)                    | kl_to_human           | 0.137  | 0.339  | -0.2019 |  -0.2066 |   -0.1974 | 0     | 18406 | A        |
| RQ3 soft vs hard (DeBERTa)                    | ece_high_disagreement | 0.0478 | 0.1528 | -0.105  |  -0.1118 |   -0.0919 | 0     | 18406 | A        |
| RQ3 soft vs hard (DeBERTa)                    | kl_high_disagreement  | 0.1045 | 0.3843 | -0.2799 |  -0.2903 |   -0.2698 | 0     | 18406 | A        |
| RQ3 soft vs temperature-scaled hard (BERT)    | accuracy              | 0.6866 | 0.6795 |  0.0071 |   0.0021 |    0.0123 | 0.005 | 18406 | A        |
| RQ3 soft vs temperature-scaled hard (BERT)    | ece                   | 0.0209 | 0.0141 |  0.0067 |   0.0007 |    0.0135 | 0.029 | 18406 | B        |
| RQ3 soft vs temperature-scaled hard (BERT)    | kl_to_human           | 0.1311 | 0.1374 | -0.0063 |  -0.0077 |   -0.0048 | 0     | 18406 | A        |
| RQ3 soft vs temperature-scaled hard (BERT)    | ece_high_disagreement | 0.0461 | 0.0445 |  0.0016 |  -0.0115 |    0.0151 | 0.766 | 18406 | n.s.     |
| RQ3 soft vs temperature-scaled hard (BERT)    | kl_high_disagreement  | 0.0863 | 0.092  | -0.0057 |  -0.0081 |   -0.0034 | 0     | 18406 | A        |
| RQ3 soft vs temperature-scaled hard (DeBERTa) | accuracy              | 0.7011 | 0.6992 |  0.002  |  -0.0026 |    0.0065 | 0.422 | 18406 | n.s.     |
| RQ3 soft vs temperature-scaled hard (DeBERTa) | ece                   | 0.0238 | 0.0146 |  0.0092 |   0.0016 |    0.014  | 0.013 | 18406 | B        |
| RQ3 soft vs temperature-scaled hard (DeBERTa) | kl_to_human           | 0.137  | 0.1482 | -0.0112 |  -0.0125 |   -0.0098 | 0     | 18406 | A        |
| RQ3 soft vs temperature-scaled hard (DeBERTa) | ece_high_disagreement | 0.0478 | 0.0514 | -0.0036 |  -0.0127 |    0.0069 | 0.552 | 18406 | n.s.     |
| RQ3 soft vs temperature-scaled hard (DeBERTa) | kl_high_disagreement  | 0.1045 | 0.1224 | -0.0179 |  -0.0203 |   -0.0154 | 0     | 18406 | A        |