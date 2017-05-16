# Test File for Practice Codes


import random
import numpy as np



def bdayExp():
    birthdays = []
    while True:
        bday = random.randint(0,365)
        if bday in birthdays:
            break
        else:
            birthdays.append(bday)
        ppl = len(birthdays)
    return ppl


t = 10000
x = []

for i in range(t):
    y = bdayExp()
    x.append(y)

print np.mean(x)


