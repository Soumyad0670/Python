# Converting a decimal string to an integer
decimal_str = "123"
decimal_int = int(decimal_str, 10)
print(f"Decimal string {decimal_str} converted to integer: {decimal_int}")

# Converting a binary string to an integer
binary_str = "10110000"
binary_int = int(binary_str, 2)
print(f"Binary string {binary_str} converted to integer: {binary_int}")

# Converting a hexadecimal string to an integer
hex_str = "1A"
hex_int = int(hex_str, 16)
print(f"Hexadecimal string {hex_str} converted to integer: {hex_int}")

# Converting a octal string to an integer
oct_str = "234"
oct_int = int(oct_str, 8)
print(f"Octal string {oct_str} converted to integer: {oct_int}")

# Converting a decimal to a binary 
a=51
b=bin(a)[2:]
print(f"decimal value {a} converted to integer: {b}")

# Converting a decimal to a hexadecimal
c=51
d=hex(c)[2:]
print(f"decimal value {c} converted to integer: {d}")

# Converting a decimal to a octal
e=51
f=oct(e)[2:]
print(f"decimal value {e} converted to integer: {f}")

# Converting a binary to a decimal
g=10110000
h=int(str(g),2)
print(f"binary value {g} converted to integer: {h}")

# Converting a binary to a hexadecimal
i=10110000
j=hex(int(str(i),2))[2:]
print(f"binary value {i} converted to integer: {j}")

# Converting a binary to a octal
k=10110000
l=oct(int(str(k),2))[2:]
print(f"binary value {k} converted to integer: {l}")

# Converting a hexadecimal to a decimal
m='1A'
n=int(str(m),16)
print(f"hexadecimal value {m} converted to integer: {n}")

# Converting a hexadecimal to a binary
o='1A'
p=bin(int(str(o),16))[2:]
print(f"hexadecimal value {o} converted to integer: {p}")

# Converting a hexadecimal to a octal
q='1A'
r=oct(int(str(q),16))[2:]
print(f"hexadecimal value {q} converted to integer: {r}")

# Converting a octal to a decimal
s=123
t=int(str(s),8)
print(f"octal value {s} converted to integer: {t}")

# Converting a octal to a binary
u=123
v=bin(int(str(u),8))[2:]
print(f"octal value {u} converted to integer: {v}")

# Converting a octal to a hexadecimal
w=123
x=hex(int(str(w),8))[2:]
print(f"octal value {w} converted to integer: {x}")
