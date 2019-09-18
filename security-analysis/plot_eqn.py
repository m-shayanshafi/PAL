from sympy import *

# committee size is k
# stake is s
# probability is p

p = 0.5
s = 0.1
N = 100

k, i = symbols("k, i")

# eq = p - summation(((binomial(N, i) )* (s**i) * (1-s)**(N-i)) , (i, ((k/2)+1), N))

eq = p < summation((binomial(N,i)* (s**i) * (1-s)**(N-i)) , (i, ((k/2)+1), N))


print(str(solve(eq, [k])))