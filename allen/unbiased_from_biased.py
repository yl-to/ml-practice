import random

def getRandom01():
    return 0 if random.random() < 0.8 else 1  # p = 0.8

def getUnbiasedRandom01():
    while True:
        a = getRandom01()
        b = getRandom01()
        # p(01) = p(1-p)
        # p(10) = (1-p)p
        # if we return a, it could be 0 of 01 or 1 of 10, and the prob is exactly the same
        if a != b:
            return a

def getUnbiasedRandom0N(N):
    # N + 1 -> [0, ... N] 
    bits_len = (N + 1).bit_length()
    while True:
        x = 0
        for _ in range(bits_len):
            x = (x << 1) | getUnbiasedRandom01() # move to left
        if x <= N:
            return x

print(getUnbiasedRandom0N(4))