**pass@8 (leaderboard metric)**

| Model | AIME24 | MATH500 | CN_MATH_24 | KAOYAN | AMC23 | MINERVA | OLYMPIAD | GPQA | AVG |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.167 | 0.846 | 0.333 | 0.578 | 0.775 | 0.482 | 0.498 | 0.747 | 0.569 |
| hybrid | 0.200 | 0.876 | 0.433 | 0.603 | 0.750 | 0.474 | 0.551 | 0.727 | 0.590 |

**pass@1 (mean over 8 samples)**

| Model | AIME24 | MATH500 | CN_MATH_24 | KAOYAN | AMC23 | MINERVA | OLYMPIAD | GPQA | AVG |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.062 | 0.650 | 0.108 | 0.225 | 0.419 | 0.308 | 0.281 | 0.301 | 0.274 |
| hybrid | 0.050 | 0.646 | 0.129 | 0.273 | 0.403 | 0.265 | 0.284 | 0.217 | 0.262 |

**% generations with a \boxed{} answer**

| Model | AIME24 | MATH500 | CN_MATH_24 | KAOYAN | AMC23 | MINERVA | OLYMPIAD | GPQA |
|---|---|---|---|---|---|---|---|---|
| baseline | 97% | 99% | 96% | 95% | 99% | 100% | 98% | 99% |
| hybrid | 76% | 93% | 89% | 94% | 87% | 89% | 88% | 79% |

**median generation length (chars)**

| Model | AIME24 | MATH500 | CN_MATH_24 | KAOYAN | AMC23 | MINERVA | OLYMPIAD | GPQA |
|---|---|---|---|---|---|---|---|---|
| baseline | 2,372 | 1,378 | 2,362 | 1,338 | 1,995 | 1,599 | 2,001 | 2,200 |
| hybrid | 39,257 | 12,509 | 31,178 | 21,374 | 21,096 | 20,512 | 22,270 | 28,005 |

(x/N) = partial coverage. AVG = mean of CN_MATH_24, KAOYAN, AMC23, MINERVA, OLYMPIAD, GPQA (leaderboard set minus AIME25), only when all six are complete.
