import numpy as np

A1 = 3
B1 = 3

A2 = 0
B2 = 5

A3 = 5
B3 = 0

A4 = 1
B4 = 1

q1 = (B4 - B3) / (B1 - B2 - B3 + B4)
q2 = (A4 - A2) / (A1 - A3 - A2 + A4)

print(f"q1 = {q1}, q2 = {q2}")