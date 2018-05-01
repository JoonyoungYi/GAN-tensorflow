# GAN-tensorflow

- The repository reproducing MNIST experiment in the original GAN paper, [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf).
- In this repository, I used tensorflow to implement the paper.
- I tried to simulate the settings of the paper.
  - For generator,
    - I used ReLU and sigmoid units.
  - For discriminator,
    - I used dropout(rate=0.5) and max-out units. Also, I used sigmoid units for final layer.

- How to init and run
```
mkdir logs
virtualenv .venv -p python3
. .venv/bin/activate
pip install -r requirements.txt
python run.py
```

- In Proposition 2, Authors said that p_g converges to p_d if G and D have enough capacities.
  - In the experience of running this code, the generated images are natural if the epochs are more than 100.
- Also, Vanilla GAN has discovered, in this experiments, that it is learned not only to generate all 10 digits, but to produce only specific digits.
  - I think that this cons motivates the research, [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784).
  - I uploaded the generated images in the `images/` folder.

- Example Results in console
```
[EPOCH    0] D_loss_value: 0.0080, G_loss_value: 32.1689
[EPOCH    1] D_loss_value: 0.0022, G_loss_value: 12.0490
[EPOCH    2] D_loss_value: 0.0009, G_loss_value: 10.6573
[EPOCH    3] D_loss_value: 0.0000, G_loss_value: 17.2036
[EPOCH    4] D_loss_value: 0.0001, G_loss_value: 14.9201
[EPOCH    5] D_loss_value: 0.0003, G_loss_value: 14.5045
[EPOCH    6] D_loss_value: 0.0003, G_loss_value: 13.4628
[EPOCH    7] D_loss_value: 0.0007, G_loss_value: 14.6148
[EPOCH    8] D_loss_value: 0.0000, G_loss_value: 15.4840
[EPOCH    9] D_loss_value: 0.0001, G_loss_value: 13.9772
[EPOCH   10] D_loss_value: 0.0598, G_loss_value: 17.4824
[EPOCH   11] D_loss_value: 0.0036, G_loss_value: 11.3463
[EPOCH   12] D_loss_value: 0.0190, G_loss_value: 16.7078
[EPOCH   13] D_loss_value: 0.0045, G_loss_value: 12.7095
[EPOCH   14] D_loss_value: 0.0002, G_loss_value: 15.4213
[EPOCH   15] D_loss_value: 0.0176, G_loss_value: 11.8951
[EPOCH   16] D_loss_value: 0.0316, G_loss_value: 8.8072
[EPOCH   17] D_loss_value: 0.0076, G_loss_value: 9.8487
[EPOCH   18] D_loss_value: 0.0012, G_loss_value: 11.5320
[EPOCH   19] D_loss_value: 0.0100, G_loss_value: 13.1394
[EPOCH   20] D_loss_value: 0.0017, G_loss_value: 11.6942
[EPOCH   21] D_loss_value: 0.0205, G_loss_value: 10.4079
[EPOCH   22] D_loss_value: 0.0044, G_loss_value: 9.4977
[EPOCH   23] D_loss_value: 0.0260, G_loss_value: 10.2554
[EPOCH   24] D_loss_value: 0.0130, G_loss_value: 8.2407
[EPOCH   25] D_loss_value: 0.0699, G_loss_value: 7.5603
[EPOCH   26] D_loss_value: 0.0485, G_loss_value: 10.3432
[EPOCH   27] D_loss_value: 0.0042, G_loss_value: 10.0981
[EPOCH   28] D_loss_value: 0.0152, G_loss_value: 7.6090
[EPOCH   29] D_loss_value: 0.0082, G_loss_value: 7.9508
[EPOCH   30] D_loss_value: 0.0511, G_loss_value: 7.7765
[EPOCH   31] D_loss_value: 0.0522, G_loss_value: 9.0819
[EPOCH   32] D_loss_value: 0.0196, G_loss_value: 8.3869
[EPOCH   33] D_loss_value: 0.0562, G_loss_value: 7.2563
[EPOCH   34] D_loss_value: 0.0362, G_loss_value: 8.1146
[EPOCH   35] D_loss_value: 0.0197, G_loss_value: 6.5053
[EPOCH   36] D_loss_value: 0.1263, G_loss_value: 7.1804
[EPOCH   37] D_loss_value: 0.0575, G_loss_value: 8.1146
[EPOCH   38] D_loss_value: 0.1039, G_loss_value: 5.9564
[EPOCH   39] D_loss_value: 0.1309, G_loss_value: 5.8973
[EPOCH   40] D_loss_value: 0.1396, G_loss_value: 6.1387
[EPOCH   41] D_loss_value: 0.0342, G_loss_value: 5.7191
[EPOCH   42] D_loss_value: 0.0823, G_loss_value: 5.7406
[EPOCH   43] D_loss_value: 0.0843, G_loss_value: 5.2452
[EPOCH   44] D_loss_value: 0.0800, G_loss_value: 6.7948
[EPOCH   45] D_loss_value: 0.1549, G_loss_value: 6.7107
[EPOCH   46] D_loss_value: 0.0712, G_loss_value: 6.2155
[EPOCH   47] D_loss_value: 0.0810, G_loss_value: 6.0844
[EPOCH   48] D_loss_value: 0.1037, G_loss_value: 6.4112
[EPOCH   49] D_loss_value: 0.1081, G_loss_value: 6.6832
[EPOCH   50] D_loss_value: 0.0190, G_loss_value: 6.4954
[EPOCH   51] D_loss_value: 0.0756, G_loss_value: 6.8696
[EPOCH   52] D_loss_value: 0.0934, G_loss_value: 6.0437
[EPOCH   53] D_loss_value: 0.0890, G_loss_value: 5.5836
[EPOCH   54] D_loss_value: 0.1048, G_loss_value: 5.4501
[EPOCH   55] D_loss_value: 0.1694, G_loss_value: 5.9481
[EPOCH   56] D_loss_value: 0.1195, G_loss_value: 6.1955
[EPOCH   57] D_loss_value: 0.0727, G_loss_value: 6.0380
[EPOCH   58] D_loss_value: 0.0453, G_loss_value: 5.8594
[EPOCH   59] D_loss_value: 0.0946, G_loss_value: 5.7924
[EPOCH   60] D_loss_value: 0.0697, G_loss_value: 5.6335
[EPOCH   61] D_loss_value: 0.1382, G_loss_value: 4.8336
[EPOCH   62] D_loss_value: 0.0992, G_loss_value: 6.1248
[EPOCH   63] D_loss_value: 0.1016, G_loss_value: 6.8778
[EPOCH   64] D_loss_value: 0.0648, G_loss_value: 5.8481
[EPOCH   65] D_loss_value: 0.0588, G_loss_value: 6.3864
[EPOCH   66] D_loss_value: 0.0926, G_loss_value: 7.7488
[EPOCH   67] D_loss_value: 0.1129, G_loss_value: 6.8627
[EPOCH   68] D_loss_value: 0.0452, G_loss_value: 5.8018
[EPOCH   69] D_loss_value: 0.0302, G_loss_value: 7.9877
[EPOCH   70] D_loss_value: 0.0896, G_loss_value: 6.4343
[EPOCH   71] D_loss_value: 0.0850, G_loss_value: 6.1875
[EPOCH   72] D_loss_value: 0.0606, G_loss_value: 7.2912
[EPOCH   73] D_loss_value: 0.1310, G_loss_value: 6.3587
[EPOCH   74] D_loss_value: 0.0726, G_loss_value: 7.7686
[EPOCH   75] D_loss_value: 0.1080, G_loss_value: 6.0512
[EPOCH   76] D_loss_value: 0.1611, G_loss_value: 6.1512
[EPOCH   77] D_loss_value: 0.1051, G_loss_value: 6.7559
[EPOCH   78] D_loss_value: 0.0507, G_loss_value: 6.2230
[EPOCH   79] D_loss_value: 0.0907, G_loss_value: 5.0350
[EPOCH   80] D_loss_value: 0.1011, G_loss_value: 6.6254
[EPOCH   81] D_loss_value: 0.0795, G_loss_value: 5.2672
[EPOCH   82] D_loss_value: 0.0702, G_loss_value: 5.6329
[EPOCH   83] D_loss_value: 0.0741, G_loss_value: 5.7450
[EPOCH   84] D_loss_value: 0.0759, G_loss_value: 5.7181
[EPOCH   85] D_loss_value: 0.1287, G_loss_value: 6.1753
[EPOCH   86] D_loss_value: 0.1422, G_loss_value: 6.0836
[EPOCH   87] D_loss_value: 0.0651, G_loss_value: 6.5480
[EPOCH   88] D_loss_value: 0.0954, G_loss_value: 6.9165
[EPOCH   89] D_loss_value: 0.1396, G_loss_value: 6.4697
[EPOCH   90] D_loss_value: 0.1548, G_loss_value: 6.4776
[EPOCH   91] D_loss_value: 0.0508, G_loss_value: 6.7553
[EPOCH   92] D_loss_value: 0.1665, G_loss_value: 5.5344
[EPOCH   93] D_loss_value: 0.0900, G_loss_value: 6.0187
[EPOCH   94] D_loss_value: 0.2432, G_loss_value: 7.4612
[EPOCH   95] D_loss_value: 0.0627, G_loss_value: 6.0234
[EPOCH   96] D_loss_value: 0.2265, G_loss_value: 5.9658
[EPOCH   97] D_loss_value: 0.0511, G_loss_value: 7.2241
[EPOCH   98] D_loss_value: 0.0935, G_loss_value: 5.0405
[EPOCH   99] D_loss_value: 0.0886, G_loss_value: 6.5891
[EPOCH  100] D_loss_value: 0.1290, G_loss_value: 5.3030
[EPOCH  101] D_loss_value: 0.1380, G_loss_value: 7.0934
[EPOCH  102] D_loss_value: 0.0737, G_loss_value: 6.5702
[EPOCH  103] D_loss_value: 0.0902, G_loss_value: 6.0958
[EPOCH  104] D_loss_value: 0.1246, G_loss_value: 5.5639
[EPOCH  105] D_loss_value: 0.1663, G_loss_value: 5.6081
[EPOCH  106] D_loss_value: 0.1508, G_loss_value: 6.2920
[EPOCH  107] D_loss_value: 0.1543, G_loss_value: 6.3511
[EPOCH  108] D_loss_value: 0.1274, G_loss_value: 7.5373
[EPOCH  109] D_loss_value: 0.1321, G_loss_value: 7.3543
[EPOCH  110] D_loss_value: 0.1212, G_loss_value: 6.3580
[EPOCH  111] D_loss_value: 0.1497, G_loss_value: 5.0521
[EPOCH  112] D_loss_value: 0.0619, G_loss_value: 7.2138
[EPOCH  113] D_loss_value: 0.0797, G_loss_value: 7.5042
[EPOCH  114] D_loss_value: 0.1278, G_loss_value: 6.2151
[EPOCH  115] D_loss_value: 0.1282, G_loss_value: 5.5582
[EPOCH  116] D_loss_value: 0.1178, G_loss_value: 6.7124
[EPOCH  117] D_loss_value: 0.1034, G_loss_value: 6.0661
[EPOCH  118] D_loss_value: 0.0731, G_loss_value: 6.6560
[EPOCH  119] D_loss_value: 0.1005, G_loss_value: 7.4144
[EPOCH  120] D_loss_value: 0.0920, G_loss_value: 6.5401
[EPOCH  121] D_loss_value: 0.1982, G_loss_value: 7.6528
[EPOCH  122] D_loss_value: 0.1418, G_loss_value: 5.6287
[EPOCH  123] D_loss_value: 0.1259, G_loss_value: 6.4666
[EPOCH  124] D_loss_value: 0.0630, G_loss_value: 6.2562
[EPOCH  125] D_loss_value: 0.0937, G_loss_value: 7.8595
[EPOCH  126] D_loss_value: 0.1716, G_loss_value: 6.0756
[EPOCH  127] D_loss_value: 0.0659, G_loss_value: 6.4680
[EPOCH  128] D_loss_value: 0.1162, G_loss_value: 6.2898
[EPOCH  129] D_loss_value: 0.1098, G_loss_value: 6.4762
[EPOCH  130] D_loss_value: 0.0876, G_loss_value: 6.1827
[EPOCH  131] D_loss_value: 0.1074, G_loss_value: 7.0764
[EPOCH  132] D_loss_value: 0.1097, G_loss_value: 6.3453
[EPOCH  133] D_loss_value: 0.0934, G_loss_value: 6.0962
[EPOCH  134] D_loss_value: 0.0950, G_loss_value: 6.4032
[EPOCH  135] D_loss_value: 0.0460, G_loss_value: 6.4763
[EPOCH  136] D_loss_value: 0.1185, G_loss_value: 6.1663
[EPOCH  137] D_loss_value: 0.1362, G_loss_value: 6.4183
[EPOCH  138] D_loss_value: 0.1433, G_loss_value: 5.1076
[EPOCH  139] D_loss_value: 0.0638, G_loss_value: 8.0808
[EPOCH  140] D_loss_value: 0.1911, G_loss_value: 5.5184
[EPOCH  141] D_loss_value: 0.0255, G_loss_value: 7.0241
[EPOCH  142] D_loss_value: 0.0917, G_loss_value: 7.0252
[EPOCH  143] D_loss_value: 0.0799, G_loss_value: 6.8296
[EPOCH  144] D_loss_value: 0.2416, G_loss_value: 5.9543
[EPOCH  145] D_loss_value: 0.0836, G_loss_value: 6.0113
[EPOCH  146] D_loss_value: 0.0441, G_loss_value: 6.8695
[EPOCH  147] D_loss_value: 0.1636, G_loss_value: 6.8355
[EPOCH  148] D_loss_value: 0.1003, G_loss_value: 6.8252
[EPOCH  149] D_loss_value: 0.0572, G_loss_value: 6.7487
[EPOCH  150] D_loss_value: 0.1695, G_loss_value: 5.5399
[EPOCH  151] D_loss_value: 0.1195, G_loss_value: 6.2271
[EPOCH  152] D_loss_value: 0.0925, G_loss_value: 6.6135
[EPOCH  153] D_loss_value: 0.1401, G_loss_value: 5.9098
[EPOCH  154] D_loss_value: 0.0425, G_loss_value: 7.6760
[EPOCH  155] D_loss_value: 0.0495, G_loss_value: 6.8408
[EPOCH  156] D_loss_value: 0.1129, G_loss_value: 6.3365
[EPOCH  157] D_loss_value: 0.2442, G_loss_value: 7.3536
[EPOCH  158] D_loss_value: 0.1694, G_loss_value: 8.3792
[EPOCH  159] D_loss_value: 0.0666, G_loss_value: 6.5563
[EPOCH  160] D_loss_value: 0.1072, G_loss_value: 6.8735
[EPOCH  161] D_loss_value: 0.1052, G_loss_value: 6.8293
[EPOCH  162] D_loss_value: 0.0454, G_loss_value: 7.4873
[EPOCH  163] D_loss_value: 0.0531, G_loss_value: 6.9219
[EPOCH  164] D_loss_value: 0.0425, G_loss_value: 7.1207
[EPOCH  165] D_loss_value: 0.1025, G_loss_value: 7.5675
[EPOCH  166] D_loss_value: 0.0963, G_loss_value: 6.7009
[EPOCH  167] D_loss_value: 0.0921, G_loss_value: 6.4189
[EPOCH  168] D_loss_value: 0.0639, G_loss_value: 8.1699
[EPOCH  169] D_loss_value: 0.0246, G_loss_value: 7.2320
[EPOCH  170] D_loss_value: 0.1239, G_loss_value: 6.3692
[EPOCH  171] D_loss_value: 0.1441, G_loss_value: 8.0174
[EPOCH  172] D_loss_value: 0.1748, G_loss_value: 7.0068
[EPOCH  173] D_loss_value: 0.0325, G_loss_value: 8.6464

```
