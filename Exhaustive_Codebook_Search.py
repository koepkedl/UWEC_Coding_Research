#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Duncan Koepke, University of Wisconsin-Eau Claire

# 2023 Research with Dr. Allison Beemer


import numpy as np
import math
from tqdm import tqdm
from itertools import chain, combinations, product
import time

class TimeChecker:
    def __init__(self):
        self.time_check_count = 0
        self.start_time = time.time()
    def print_elapsed_time(self):
        print("(%s) --- %s seconds ---" % (
            str(self.time_check_count), 
            str(self.elapsed_time())
            )
        )
        self.time_check_count = self.time_check_count + 1
    
    def elapsed_time(self) -> float:
        return 0.01 * round(100 * (time.time() - self.start_time))


def powerset(iterable, width):
    # s = list(iterable)
    return chain.from_iterable(combinations(iterable, r) for r in range(2,width+1))

# powerset makes sets of sizes from 2 until the width of the poset. 

def binomialcoef(j,k):
    j = int(j)
    k = int(k)
    return math.factorial(j)/(math.factorial(k)*math.factorial(j-k))

# binomialcoef generates the binomial coefficient n choose k.

def unique(list1):
    list_set = set(list1)
    unique_list = list(list_set)
    return unique_list

# unique returns a list containing the unique values of the input list.

timer = TimeChecker()

n = 5 #This sets the block length.
codewords = []
adversary_words = []

width = int(binomialcoef(n,np.ceil(n/2)))

for z in range(2**n):
    codewords.append(int(np.binary_repr(z)))
    adversary_words.append(int(np.binary_repr(z)))

# 0
timer.print_elapsed_time()

codewords.pop(2**n-1)
codewords.pop(0)
adversary_words.pop(0)

# 1
timer.print_elapsed_time()

codebooks = []
# codebooks = list(map(lambda c: set(c), powerset(codewords, width)))
codebooks = list(powerset(codewords, width))
codebookpairs = []

# 2
timer.print_elapsed_time()


i = 0
# Question: can we generate this faster using a partition of the set of codewords?
# EG remove this quadtratic loop, instead iterate on partitions of subsets of the set of codewords.
# and split these as we iterate, instead of iterating and checking for empty intersection.
codebookpairs = [
    [codebooks[i], codebooks[x]]
    for i in tqdm(range(len(codebooks)))
    for x in range(i+1,len(codebooks))
    
    if len(set(codebooks[i]).intersection(set(codebooks[x]))) == 0
]
# while len(codebooks) > i:
#     print(i, "//", len(codebooks))
#     for x in range(i+1,len(codebooks)):
#         pairs = []
#         if len(set(codebooks[i]).intersection(set(codebooks[x]))) == 0:
#             pairs.append(codebooks[i])
#             pairs.append(codebooks[x])
#             codebookpairs.append(pairs)
#     print(len(pairs))
#     i = i+1

## ERIC'S CODE
# ones = [x for x in codebookpairs if x[0][0] == 1]
# from collections import defaultdict
# s = defaultdict(list)
# for one in ones:
#     s[one[0][1]].append(one[1])
## END ERIC'S CODE

# 3
timer.print_elapsed_time()

    
adversarial_sums = []
potential_sums = []
codebooks_12 =[]

Bs = []
Cs = []
partially_correctable_codebooks =[]
coords = []

print("big loop start")
for a in tqdm(range(len(codebookpairs))):
    # combo = [
    #     [
    #         Q[0] + Q[1],
    #         Q[0]+Q[1]+Q[2],
    #         [Q[0],Q[1]]
    #     ]
    #     for Q in list(product(codebookpairs[a][0], codebookpairs[a][1], adversary_words))
    # ]
    for Q in list(product(codebookpairs[a][0], codebookpairs[a][1], adversary_words)):
        potential_sums.append(Q[0]+Q[1])
        adversarial_sums.append(Q[0]+Q[1]+Q[2])
        coords.append([Q[0],Q[1]])
    # potential_sums = [Q[0] for Q in combo]
    # adversarial_sums = [Q[1] for Q in combo]
    # coords = [Q[2] for Q in combo]
    potential_sums_set = set(potential_sums)
    adversarial_sums_set = set(adversarial_sums)
    unique_sums = unique(potential_sums)
    if len(potential_sums_set.intersection(adversarial_sums_set)) == 0 \
            and (2**n-1)*len(unique_sums) == len(potential_sums):
        unique_adversarial_sums = unique(adversarial_sums)
        check = 1
        index = 0
        while check == 1 and index < len(unique_adversarial_sums):
            j = unique_adversarial_sums[index]
            for k in range(len(adversarial_sums)):
                if adversarial_sums[k] == j:
                    Bs.append(coords[k][0])
                    Cs.append(coords[k][1])
            uBs = unique(Bs)
            uCs = unique(Cs)
            if len(uBs) > 1 and len(uCs) > 1:
                check = 0
            Bs = []
            Cs = []
            index = index + 1
        if check == 1:
            partially_correctable_codebooks.append(codebookpairs[a])
    partial_sums=[]
    coords=[]
    adversarial_sums = []
    potential_sums = []

print("--- %s seconds ---" % (timer.elapsed_time()))
print('\n')
print(len(partially_correctable_codebooks))
print("Codebooks that are partially correctable for block length "+ str(n))
# print(partially_correctable_codebooks)

# import pandas as pd

# df = pd.DataFrame(partially_correctable_codebooks)
# df.to_csv('Good_Codebooks.csv', index=True)


# In[4]:





# In[ ]:




