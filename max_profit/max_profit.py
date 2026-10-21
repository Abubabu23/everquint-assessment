from functools import lru_cache

building = [
    ("T", 5, 1500),
    ("P", 4, 1000),
    ("C", 10, 2000),
]

def max_profit(n):
    @lru_cache(maxsize=None)
    def best(t):
        best_earn = 0
        best_sols = {(0, 0, 0)}
        for i, (name, built_time, rate) in enumerate(building):
            finish = t + built_time
            if finish >= n:
                continue
            earn = rate * (n - finish)
            sub_earn, sub_sols = best(finish)
            total = earn + sub_earn
            new_sols = set()
            for s in sub_sols:
                s = list(s)
                s[i] += 1
                new_sols.add(tuple(s))
            if total > best_earn:
                best_earn, best_sols = total, new_sols
            elif total == best_earn:
                best_sols = best_sols | new_sols
        return best_earn, frozenset(best_sols)

    return best(0)

def show(n):
    earn, sols = max_profit(n)
    print(f"Time Unit: {n}")
    print(f"Earnings: ${earn}")
    print("Solutions")
    for k, (t, p, c) in enumerate(sorted(sols, reverse=True), 1):
        print(f"{k}. T: {t} P: {p} C: {c}")
    print()

if __name__ == "__main__":
    for n in (7, 8, 13):
        show(n)