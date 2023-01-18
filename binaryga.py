import matplotlib.pyplot as plt
import math
import random
size = 10
a = [0]*size

for i in range(0, size):
    a[i] = random.randrange(1, 50)
print(a)


def fitness(x):

    return x*x+1


def param(a):
    return fitness(a)


def func_util(a):
    sum = 0
    max = 0
    for i in range(0, size):
        if(max < a[i]):
            max = a[i]
        sum += a[i]

    avg = sum/size
    return avg, max, sum


prob_mut = 0.3
prob_c = 0.7


def crossover(parent1, parent2):
    if(random.random() < prob_c):
        return parent1, parent2
    child1 = 0
    child2 = 0
    mid = random.randrange(0, size-1)
    # print(f'{mid}')
    p1 = parent1
    p2 = parent2
    p1_bits = [0]*size
    p2_bits = [0]*size
    for j in range(0, size):
        p1_bits[j] = p1 % 2
        p2_bits[j] = p2 % 2
        p1 = int(p1/2)
        p2 = int(p2/2)
    mult = 1
    for j in range(0, mid+1):
        child1 += p1_bits[j]*mult
        child2 += p2_bits[j]*mult
        mult = mult*2
    for j in range(mid+1, size):
        child1 += p2_bits[j]*mult
        child2 += p1_bits[j]*mult
        mult = mult*2

    return child1, child2


def mutation(a):
    new_a = a
    mid = random.randrange(0, size-1)
    new_a[mid] = random.randrange(1, 2**(size - 1)-1)

    return new_a


average_obs = []
maximum_obs = []
sum_obs = []
for itr in range(0, 50):
    new_a = []
    avg, max, sum = func_util(a)
    average_obs.append(avg)
    maximum_obs.append(max)
    sum_obs.append(sum)
    a.sort(reverse=True, key=param)
    for i in range(0, size, 2):
        child1, child2 = crossover(a[i], a[i+1])

        new_a.append(child1)
        new_a.append(child2)

    if(random.random() < prob_mut):
        new_a = mutation(new_a)
        a = new_a
    print(a)

print(average_obs)
print(maximum_obs)


plt.xlabel('Iterations')
plt.ylabel('Value')
plt.plot(maximum_obs, label="Max")
plt.plot(average_obs, label='Average')
plt.legend()
plt.savefig('plt.png')
