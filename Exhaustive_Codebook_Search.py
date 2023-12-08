#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Duncan Koepke, University of Wisconsin-Eau Claire

# 2023 Research with Dr. Allison Beemer


import numpy as np
from itertools import chain, combinations, product
import time

start_time = time.time()


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(2,width+1))

# powerset makes sets of sizes from 2 until the width of the poset. 

def binomialcoef(j,k):
    return np.math.factorial(j)/(np.math.factorial(k)*np.math.factorial(j-k))

# binomialcoef generates the binomial coefficient n choose k.

def unique(list1):
    list_set = set(list1)
    unique_list = list(list_set)
    return unique_list

# unique returns a list containing the unique values of the input list.

n = 3 #This sets the block length.
codewords = []
adversary_words = []

width = int(binomialcoef(n,np.ceil(n/2)))

for z in range(2**n):
    codewords.append(int(np.binary_repr(z)))
    adversary_words.append(int(np.binary_repr(z)))

codewords.pop(2**n-1)
codewords.pop(0)
adversary_words.pop(0)


codebooks = []
codebooks = list(powerset(codewords))
codebookpairs = []

i = 0
while len(codebooks) > i:
    for x in range(i+1,len(codebooks)):
        pairs = []
        if len(set(codebooks[i]).intersection(set(codebooks[x]))) == 0:
            pairs.append(codebooks[i])
            pairs.append(codebooks[x])
            codebookpairs.append(pairs)
    i = i+1

    
adversarial_sums = []
potential_sums = []
codebooks_12 =[]

Bs = []
Cs = []
partially_correctable_codebooks =[]
coords = []

for a in range(len(codebookpairs)):
    for Q in list(product(codebookpairs[a][0], codebookpairs[a][1], adversary_words)):
        potential_sums.append(Q[0]+Q[1])
        adversarial_sums.append(Q[0]+Q[1]+Q[2])
        coords.append([Q[0],Q[1]])
    potential_sums_set = set(potential_sums)
    adversarial_sums_set = set(adversarial_sums)
    unique_sums = unique(potential_sums)
    if len(potential_sums_set.intersection(adversarial_sums_set)) == 0 and (2**n-1)*len(unique_sums) == len(potential_sums):
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

print("--- %s seconds ---" % (time.time() - start_time))
print('\n')
print(len(partially_correctable_codebooks))
print("Codebooks that are partially correctable for block length "+ str(n))
print(partially_correctable_codebooks)


# In[4]:





# In[ ]:




