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

def binomialcoef(j,k):
    j = int(j)
    k = int(k)
    return math.factorial(j)/(math.factorial(k)*math.factorial(j-k))

class TimeChecker:
    def __init__(self, places_to_print: int = 3):
        self.time_check_count = 0
        self.start_time = time.time()
        self.scale_up = 10**places_to_print
        self.normalize = 1 / self.scale_up

    def print_elapsed_time(self):
        print("(%s) --- %s seconds ---" % (
            str(self.time_check_count), 
            str(self.elapsed_time())
            )
        )
        self.time_check_count = self.time_check_count + 1
    
    def elapsed_time(self) -> float:
        return self.normalize * round(self.scale_up * (time.time() - self.start_time))

# TaiChi code
n = 5 #This sets the block length.
import taichi as ti
ti.init(arch=ti.gpu)

powerset_ti = ti.field(shape=(2**n, n), dtype=ti.i32)
codewords = []
adversary_words = []

width = int(binomialcoef(n,np.ceil(n/2)))

for z in range(2**n):
    codewords.append(int(np.binary_repr(z)))
    adversary_words.append(int(np.binary_repr(z)))

@ti.kernel
def fill_powerset(iterable, width):
    # s = list(iterable)
    return chain.from_iterable(combinations(iterable, r) for r in range(2,width+1))

# powerset makes sets of sizes from 2 until the width of the poset. 


# binomialcoef generates the binomial coefficient n choose k.

def unique(list1):
    list_set = set(list1)
    unique_list = list(list_set)
    return unique_list

# unique returns a list containing the unique values of the input list.

timer = TimeChecker()


# 0
timer.print_elapsed_time()

codewords.pop(2**n-1)
codewords.pop(0)
# codewords_idx = {x: i for i, x in enumerate(codewords)}
adversary_words.pop(0)

# 1
timer.print_elapsed_time()

codebooks = []
# codebooks = list(map(lambda c: set(c), powerset(codewords, width)))
codebooks = list(fill_powerset(codewords, width))
codebookpairs = []

# 2
timer.print_elapsed_time()

# 2D field of 8 bit integers (just inclusion) 
codebookpairs_ti = ti.field(shape=(len(codebooks), 2, len(codewords)), dtype=ti.i32)
codebookpairs_ti.fill(0)

for i in range(len(codebooks)):
    for x in range(i+1,len(codebooks)):
        codebookpairs_ti[i, 0]
        # if len(codebooks[i]) + len(codebooks[x]) == len(set(codebooks[i] + codebooks[x])):
        #     for c in codebooks[i]:
        #         codebookpairs_ti[i, codewords_idx[c]] = 1
        #     for c in codebooks[x]:
        #         codebookpairs_ti[x, codewords_idx[c]] = 1
    
timer.print_elapsed_time()


a = []
for j in range(5):
    a.append(j * j)

b = [j * j for j in range(5) if j % 2 == 0]





i = 0
# Question: can we generate this faster using a partition of the set of codewords?
# EG remove this quadtratic loop, instead iterate on partitions of subsets of the set of codewords.
# and split these as we iterate, instead of iterating and checking for empty intersection.
# changing to list comprehension saves 33% of time
# the if statement at the end is another 33% savings
# STILL NOT ENOUGH for n=5, which has 53009071 iterations on the outer loop
codebookpairs = [
    [codebooks[i], codebooks[x]]
    for i in tqdm(range(len(codebooks)))
    for x in range(i+1,len(codebooks))
    
    if len(codebooks[i]) + len(codebooks[x]) == len(set(codebooks[i] + codebooks[x]))
    # if len(set(codebooks[i]).intersection(set(codebooks[x]))) == 0
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
    for Q in list(product(codebookpairs[a][0], codebookpairs[a][1], adversary_words)):
        potential_sums.append(Q[0]+Q[1])
        adversarial_sums.append(Q[0]+Q[1]+Q[2])
        coords.append([Q[0],Q[1]])
    potential_sums_set = set(potential_sums)
    adversarial_sums_set = set(adversarial_sums)
    unique_sums = unique(potential_sums)
    if len(potential_sums_set.intersection(adversarial_sums_set)) == 0 \
            and (2**n-1)*len(unique_sums) == len(potential_sums):
        unique_adversarial_sums = unique(adversarial_sums)
        check = 1
        index = 0
        # when adversary does act, get repeated sums,
        # track sum back to 1 of the 2 code books.
        # eg 112 comes out, meaning (adversary contrib) + (codebook 1 contrib) + (codebook 2 contrib)
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




