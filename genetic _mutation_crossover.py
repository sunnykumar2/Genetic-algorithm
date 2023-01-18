import matplotlib.pyplot as plt
import numpy as np
import random
# single point crossover binary coded GA
pop_size = 8
pop = [np.random.randint(0, 32) for i in range(0, pop_size)]


def calculate(a):
    max = 0
    for i in range(0, 8):
        if(max < a[i]):
            max = a[i]
    return max


def fitnessFunction(x):
    return x*x+x*x*x + x


def mutationFunction(a):
    temp = a
    mid = random.randint(0, 5)
    temp[mid] = 1 - temp[mid]

    return temp


def crossover(parent1, parent2):
    child1 = 0
    child2 = 0
    mid = random.randrange(0, 5)
    # print(f'{mid}')
    p1 = parent1
    p2 = parent2
    p1_bits = [0]*5
    p2_bits = [0]*5

    for j in range(0, 5):
        p1_bits[j] = p1 % 2
        p2_bits[j] = p2 % 2
        p1 = int(p1/2)
        p2 = int(p2/2)
    mult = 1
    for j in range(0, mid+1):
        child1 += p1_bits[j]*mult
        child2 += p2_bits[j]*mult
        mult = mult*2
    for j in range(mid+1, 5):
        child1 += p2_bits[j]*mult
        child2 += p1_bits[j]*mult
        mult = mult*2

    return child1, child2


maxvalues = []
for itr in range(0, 15):
    new_a = []
    max = calculate(pop)
    maxvalues.append(max)
    pop.sort(reverse=True, key=fitnessFunction)
    for i in range(0, 8, 2):
        child1, child2 = crossover(pop[i], pop[i+1])
        new_a.append(child1)
        new_a.append(child2)

    new_a = mutationFunction(new_a)
    pop = new_a


fitnessvalues = [num**2 for num in maxvalues]
plt.plot(fitnessvalues)
plt.show()
