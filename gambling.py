
import random
import numpy as np



def gambleExp(c):
    numbersHit = []
    moneyBet = c
    winnings = 0
    while True:
        number = random.randint(0,32)
        if number in numbersHit:
            break
        else:
            numbersHit.append(number)
        games = len(numbersHit)
    for i in range(1,games+1):
        moneyBet += 1
        winnings -= moneyBet*i
    winnings += (moneyBet*32) - moneyBet*(games+1)
    return winnings


t = 10000
x = []

for i in range(t):
    y = gambleExp(1)
    x.append(y)

print(np.mean(x))




