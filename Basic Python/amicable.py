def sum_proper_divisors(n):
    total = 1
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            total += i
            if i != n // i:
                total += n // i
    return total

amicables = set()
for i in range(2, 10000):
    a = sum_proper_divisors(i)
    if a != i and a < 10000:
        b = sum_proper_divisors(a)
        if b == i:
            amicables.add(i)
            amicables.add(a)
print(sum(amicables))
# 31626