Evaluating on test set...

Label                           Positives   Freq%       F1
------------------------------------------------------------
advancedPawn                      154,401   6.01%    0.890
advantage                         652,372  25.40%    0.538
anastasiaMate                       2,697   0.11%    0.304
arabianMate                         2,718   0.11%    0.250
attackingF2F7                       6,920   0.27%    0.756
attraction                         89,116   3.47%    0.640
backRankMate                       87,938   3.42%    0.815
bishopEndgame                      32,581   1.27%    0.860
bodenMate                             481   0.02%    0.000 ←
capturingDefender                  19,378   0.75%    0.349
castling                              764   0.03%    0.000 ←
clearance                          35,001   1.36%    0.434
crushing                        1,256,811  48.94%    0.748
defensiveMove                     167,442   6.52%    0.458
deflection                        117,023   4.56%    0.634
discoveredAttack                  137,591   5.36%    0.572
doubleBishopMate                      869   0.03%    0.000 ←
doubleCheck                        11,153   0.43%    0.355
dovetailMate                        1,201   0.05%    0.000 ←
enPassant                           3,642   0.14%    0.568
endgame                         1,369,767  53.34%    0.975
equality                           42,271   1.65%    0.072 ←
exposedKing                        73,896   2.88%    0.568
fork                              373,834  14.56%    0.601
hangingPiece                      108,317   4.22%    0.647
hookMate                            3,785   0.15%    0.457
interference                       10,273   0.40%    0.015 ←
intermezzo                         33,775   1.32%    0.451
kingsideAttack                    166,379   6.48%    0.558
knightEndgame                      21,646   0.84%    0.853
long                              634,982  24.73%    0.855
master                            241,174   9.39%    0.218
masterVsMaster                     23,062   0.90%    0.000 ←
mate                              615,763  23.98%    0.892
mateIn1                           201,034   7.83%    0.957
mateIn2                           320,745  12.49%    0.878
mateIn3                            79,927   3.11%    0.735
mateIn4                            11,518   0.45%    0.188 ←
mateIn5                             2,539   0.10%    0.000 ←
middlegame                      1,138,839  44.35%    0.910
oneMove                           233,974   9.11%    1.000
opening                            56,466   2.20%    0.727
pawnEndgame                        76,194   2.97%    0.956
pin                               165,640   6.45%    0.301
promotion                          58,685   2.29%    0.685
queenEndgame                       25,364   0.99%    0.933
queenRookEndgame                   18,954   0.74%    0.713
queensideAttack                    31,878   1.24%    0.672
quietMove                         108,783   4.24%    0.389
rookEndgame                       134,892   5.25%    0.928
sacrifice                         184,836   7.20%    0.601
short                           1,483,310  57.76%    0.999
skewer                             61,376   2.39%    0.740
smotheredMate                       5,811   0.23%    0.802
superGM                             2,114   0.08%    0.000 ←
trappedPiece                       31,260   1.22%    0.354
underPromotion                        534   0.02%    0.000 ←
veryLong                          214,951   8.37%    0.456
xRayAttack                          9,708   0.38%    0.639
zugzwang                           21,536   0.84%    0.409

Macro F1: 0.5384
Micro F1: 0.7539

Correlation between positive count and F1: 0.437

Best Thresholds per Label:
                Label  Best Threshold  F1 at Best Threshold
51              short            0.74              0.999859
40            oneMove            0.25              0.999803
42        pawnEndgame            0.64              0.996002
20            endgame            0.36              0.991003
49        rookEndgame            0.52              0.981175
34            mateIn1            0.60              0.980305
39         middlegame            0.61              0.978580
45       queenEndgame            0.31              0.964033
33               mate            0.51              0.928183
35            mateIn2            0.50              0.920303
7       bishopEndgame            0.32              0.919988
0        advancedPawn            0.26              0.915215
29      knightEndgame            0.26              0.895971
6        backRankMate            0.29              0.856459
30               long            0.15              0.854787
53      smotheredMate            0.14              0.804296
4       attackingF2F7            0.28              0.796971
36            mateIn3            0.36              0.794214
19          enPassant            0.03              0.792646
41            opening            0.29              0.788306
52             skewer            0.32              0.787092
12           crushing            0.40              0.765320
44          promotion            0.30              0.763385
46   queenRookEndgame            0.07              0.758849
5          attraction            0.33              0.736461
23               fork            0.36              0.722526
22        exposedKing            0.35              0.702575
28     kingsideAttack            0.41              0.695171
14         deflection            0.23              0.689092
47    queensideAttack            0.18              0.688419
50          sacrifice            0.30              0.687359
24       hangingPiece            0.28              0.679752
15   discoveredAttack            0.28              0.647463
58         xRayAttack            0.13              0.639445
25           hookMate            0.05              0.600707
1           advantage            0.25              0.569192
3         arabianMate            0.05              0.520256
57           veryLong            0.22              0.505553
59           zugzwang            0.28              0.500472
2       anastasiaMate            0.05              0.490610
13      defensiveMove            0.19              0.477492
27         intermezzo            0.16              0.458221
11          clearance            0.08              0.452235
48          quietMove            0.17              0.405008
55       trappedPiece            0.07              0.402774
16   doubleBishopMate            0.02              0.386740
17        doubleCheck            0.08              0.364840
9   capturingDefender            0.09              0.359176
43                pin            0.12              0.301365
37            mateIn4            0.09              0.253292
31             master            0.11              0.224131
26       interference            0.04              0.196816
10           castling            0.01              0.187793
21           equality            0.06              0.153102
18       dovetailMate            0.01              0.116041
8           bodenMate            0.01              0.082192
38            mateIn5            0.03              0.069005
56     underPromotion            0.03              0.065672
32     masterVsMaster            0.01              0.024669
54            superGM            0.01              0.004151

Optimized Macro F1 Score (using best thresholds): 0.6049