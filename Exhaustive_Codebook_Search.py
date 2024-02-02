#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Duncan Koepke, University of Wisconsin-Eau Claire

# 2023-24 Research with Dr. Allison Beemer


import numpy as np
import math
from itertools import chain, combinations, product
import time

# this gives us a progress bar on for loops automagically
from tqdm import tqdm

# A class that keeps track of elapsed time for us, and how many times we've checked the time
# Then you can just add a line like "timer.print_elapsed_time()" to your code to see how long it's been running,
# and note how many times you've called this function before now. Probably better ways, but this is easy.
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

# powerset makes sets of sizes from 2 until the width of the poset. 
# make width a parameter, better than having it bound to the global variable width below
def powerset(iterable, width):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(2,width+1))



# unique returns a list containing the unique values of the input list.
def unique(list1):
    list_set = set(list1)
    unique_list = list(list_set)
    return unique_list

# binomialcoef generates the binomial coefficient n choose k.
def binomialcoef(j,k):
    j = int(j)
    k = int(k)
    return math.factorial(j)/(math.factorial(k)*math.factorial(j-k))


n = 5 #This sets the block length.

codewords = []
adversary_words = []

width = int(binomialcoef(n,np.ceil(n/2)))

for z in range(2**n):
    codewords.append(int(np.binary_repr(z)))
    adversary_words.append(int(np.binary_repr(z)))


timer = TimeChecker()


# Since this is the first time we called .print_elapsed_time(), the counter is at (0)
# (0) --- #### seconds ---
timer.print_elapsed_time()

codewords.pop(2**n-1)
codewords.pop(0)
adversary_words.pop(0)

# (1)
timer.print_elapsed_time()

codebooks = []
codebooks = list(powerset(codewords, width))
codebookpairs = []

# 2
timer.print_elapsed_time()

# this is called a "list comprehension"
# They can be quite a bit faster than .append in a for loop
a = [j * j for j in range(5)]
b = []
for j in range(5):
    b.append(j * j)
a == b


# compute pairs [A, B] of subsets of `codewords` such that A and B are disjoint
# where |A| and |B| are between 2 and width 
### Questions for Duncan:
##   > Can we instead generate subsets C of `codewords` and 
##   > then here compute the list [A, B] of all ways of writing C = A U B, A n B = 0?
##   > So, first we'd handle all subsets C of size 4. They produce some disjoint pairs [A, B], 
##   > and [A, B] doesn't appear in any other subset C of size 4.
##   > Then, move on to |C| = 5, |C| = 6, etc.
codebookpairs = [ 
    [codebooks[i], codebooks[x]]
    for i in tqdm(range(len(codebooks)))
    for x in range(i+1,len(codebooks))
    
    ## it's faster to check it this way than in the commented out way
    ## but this is still O(N^2), N = len(codebooks) since pairwise-check
    if len(codebooks[i]) + len(codebooks[x]) == len(set(codebooks[i] + codebooks[x]))
    ## This check happens to be 33% slower, but it's still O(N^2)
    # if len(set(codebooks[i]).intersection(set(codebooks[x]))) == 0

]
# i = 0
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

import pandas as pd

df = pd.DataFrame(partially_correctable_codebooks)
df.to_csv('Good_Codebooks.csv', index=True)


# In[4]:





# In[ ]:




