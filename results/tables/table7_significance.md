| question                                                              | metric                |      A |      B |    diff |   ci_low |   ci_high |     p |     n | winner   |
|:----------------------------------------------------------------------|:----------------------|-------:|-------:|--------:|---------:|----------:|------:|------:|:---------|
| RQ1 small fine-tuned model vs LLM judge                               | accuracy              | 0.6652 | 0.6187 |  0.0465 |   0.032  |    0.0609 | 0     |  6194 | A        |
| RQ1 small fine-tuned model vs LLM judge                               | ece                   | 0.0394 | 0.3127 | -0.2732 |  -0.2843 |   -0.2563 | 0     |  6194 | A        |
| RQ1 same, both symmetrized                                            | accuracy              | 0.6752 | 0.6364 |  0.0387 |   0.0253 |    0.0534 | 0     |  6194 | A        |
| RQ1 DeBERTa vs BERT                                                   | accuracy              | 0.6577 | 0.6363 |  0.0215 |   0.0089 |    0.0337 | 0.003 |  6194 | A        |
| RQ2 symmetrization (BERT)                                             | accuracy              | 0.6464 | 0.6363 |  0.0102 |   0.0016 |    0.0189 | 0.024 |  6194 | A        |
| RQ2 symmetrization (DeBERTa)                                          | accuracy              | 0.6682 | 0.6577 |  0.0105 |   0.0031 |    0.0184 | 0.009 |  6194 | A        |
| RQ2 swap augmentation (BERT)                                          | accuracy              | 0.6445 | 0.6363 |  0.0082 |  -0.0011 |    0.0182 | 0.099 |  6194 | n.s.     |
| Bi-encoder vs cross-encoder (BERT)                                    | accuracy              | 0.6459 | 0.6363 |  0.0097 |   0      |    0.0197 | 0.054 |  6194 | n.s.     |
| Ablation: no prompt vs prompt (BERT)                                  | accuracy              | 0.6371 | 0.6363 |  0.0008 |  -0.0076 |    0.009  | 0.877 |  6194 | n.s.     |
| RQ3 soft vs hard (BiLSTM)                                             | accuracy              | 0.6664 | 0.6629 |  0.0035 |  -0.0014 |    0.0083 | 0.165 | 18406 | n.s.     |
| RQ3 soft vs hard (BiLSTM)                                             | ece                   | 0.0207 | 0.0676 | -0.0468 |  -0.0553 |   -0.0377 | 0     | 18406 | A        |
| RQ3 soft vs hard (BiLSTM)                                             | kl_to_human           | 0.1256 | 0.206  | -0.0804 |  -0.0829 |   -0.0777 | 0     | 18406 | A        |
| RQ3 soft vs hard (BiLSTM)                                             | ece_high_disagreement | 0.0328 | 0.1018 | -0.069  |  -0.0794 |   -0.0552 | 0     | 18406 | A        |
| RQ3 soft vs hard (BiLSTM)                                             | kl_high_disagreement  | 0.063  | 0.1825 | -0.1195 |  -0.1251 |   -0.1143 | 0     | 18406 | A        |
| RQ3 soft vs hard (BERT)                                               | accuracy              | 0.6866 | 0.6795 |  0.0071 |   0.0018 |    0.0125 | 0.008 | 18406 | A        |
| RQ3 soft vs hard (BERT)                                               | ece                   | 0.0209 | 0.0621 | -0.0412 |  -0.0473 |   -0.0314 | 0     | 18406 | A        |
| RQ3 soft vs hard (BERT)                                               | kl_to_human           | 0.1311 | 0.2045 | -0.0734 |  -0.076  |   -0.0709 | 0     | 18406 | A        |
| RQ3 soft vs hard (BERT)                                               | ece_high_disagreement | 0.0461 | 0.0984 | -0.0523 |  -0.0617 |   -0.0375 | 0     | 18406 | A        |
| RQ3 soft vs hard (BERT)                                               | kl_high_disagreement  | 0.0863 | 0.1895 | -0.1032 |  -0.1085 |   -0.0978 | 0     | 18406 | A        |
| RQ3 soft vs hard (DeBERTa)                                            | accuracy              | 0.7011 | 0.6992 |  0.002  |  -0.0023 |    0.0064 | 0.391 | 18406 | n.s.     |
| RQ3 soft vs hard (DeBERTa)                                            | ece                   | 0.0238 | 0.1085 | -0.0848 |  -0.0909 |   -0.0765 | 0     | 18406 | A        |
| RQ3 soft vs hard (DeBERTa)                                            | kl_to_human           | 0.137  | 0.339  | -0.2019 |  -0.2063 |   -0.1973 | 0     | 18406 | A        |
| RQ3 soft vs hard (DeBERTa)                                            | ece_high_disagreement | 0.0478 | 0.1528 | -0.105  |  -0.1119 |   -0.0916 | 0     | 18406 | A        |
| RQ3 soft vs hard (DeBERTa)                                            | kl_high_disagreement  | 0.1045 | 0.3843 | -0.2799 |  -0.2906 |   -0.2699 | 0     | 18406 | A        |
| RQ3 soft vs temperature-scaled hard (BERT)                            | accuracy              | 0.6866 | 0.6795 |  0.0071 |   0.0019 |    0.0121 | 0.008 | 18406 | A        |
| RQ3 soft vs temperature-scaled hard (BERT)                            | ece                   | 0.0209 | 0.0141 |  0.0067 |   0.0005 |    0.0138 | 0.033 | 18406 | B        |
| RQ3 soft vs temperature-scaled hard (BERT)                            | kl_to_human           | 0.1311 | 0.1374 | -0.0063 |  -0.0077 |   -0.0048 | 0     | 18406 | A        |
| RQ3 soft vs temperature-scaled hard (BERT)                            | ece_high_disagreement | 0.0461 | 0.0445 |  0.0016 |  -0.0106 |    0.015  | 0.74  | 18406 | n.s.     |
| RQ3 soft vs temperature-scaled hard (BERT)                            | kl_high_disagreement  | 0.0863 | 0.092  | -0.0057 |  -0.0079 |   -0.0033 | 0     | 18406 | A        |
| RQ3 soft vs temperature-scaled hard (DeBERTa)                         | accuracy              | 0.7011 | 0.6992 |  0.002  |  -0.0026 |    0.0065 | 0.404 | 18406 | n.s.     |
| RQ3 soft vs temperature-scaled hard (DeBERTa)                         | ece                   | 0.0238 | 0.0146 |  0.0092 |   0.0018 |    0.0141 | 0.008 | 18406 | B        |
| RQ3 soft vs temperature-scaled hard (DeBERTa)                         | kl_to_human           | 0.137  | 0.1482 | -0.0112 |  -0.0125 |   -0.0098 | 0     | 18406 | A        |
| RQ3 soft vs temperature-scaled hard (DeBERTa)                         | ece_high_disagreement | 0.0478 | 0.0514 | -0.0036 |  -0.0125 |    0.0069 | 0.559 | 18406 | n.s.     |
| RQ3 soft vs temperature-scaled hard (DeBERTa)                         | kl_high_disagreement  | 0.1045 | 0.1224 | -0.0179 |  -0.0203 |   -0.0154 | 0     | 18406 | A        |
| Reverse domain shift: harmless-trained vs helpful-trained on harmless | accuracy              | 0.696  | 0.3501 |  0.3459 |   0.3206 |    0.3704 | 0     |  3993 | A        |
| Reverse domain shift: harmless-trained vs helpful-trained on helpful  | accuracy              | 0.4196 | 0.6363 | -0.2167 |  -0.236  |   -0.1958 | 0     |  6194 | B        |
| AI labels vs human labels, unseen human data (SHP)                    | accuracy              | 0.578  | 0.5502 |  0.0278 |   0.0181 |    0.0377 | 0     | 18406 | A        |
| AI labels vs human labels, unseen human data (SHP)                    | ece                   | 0.1302 | 0.0986 |  0.0316 |   0.0214 |    0.04   | 0     | 18406 | B        |
| AI labels vs human labels, on HH human labels                         | accuracy              | 0.5525 | 0.6363 | -0.0838 |  -0.0991 |   -0.0675 | 0     |  6194 | B        |
| AI labels vs human labels, on HH human labels                         | ece                   | 0.1243 | 0.0217 |  0.1026 |   0.0835 |    0.1143 | 0     |  6194 | B        |