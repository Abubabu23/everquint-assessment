# Max Profit Problem

Mr. X builds Theatres, Pubs and Commercial Parks on Mars Land. Only one building can be under construction at a time. A building earns money per time unit only after it is built, until time `n`. Find the building mix (T, P, C) that gives the maximum earnings.

| Building | Build time | Earnings per unit time |
|---|---|---|
| Theatre (T) | 5 | $1500 |
| Pub (P) | 4 | $1000 |
| Commercial Park (C) | 10 | $2000 |

Land is unlimited, so parcel sizes do not affect the result.

## Approach
Recursion with memoization (dynamic programming).

- `best(t)` = maximum earnings if we are free to start a new building at time `t`
- For each building: `finish = t + build_time`; if `finish < n`, earnings = `rate * (n - finish)` + `best(finish)`
- Pick the maximum. If several mixes tie, all of them are kept (e.g. n = 7 gives both T:1 and P:1)
- `lru_cache` stores results so each `t` is computed once

Time: O(n), space: O(n).

## Results
| Input | Earnings | Solutions |
|---|---|---|
| 7 | $3000 | T:1 P:0 C:0, T:0 P:1 C:0 |
| 8 | $4500 | T:1 P:0 C:0 |
| 13 | $16500 | T:2 P:0 C:0 |

## How to run
python max_profit.py

The script prints the results for n = 7, 8 and 13. To test another value, call `show(n)`.