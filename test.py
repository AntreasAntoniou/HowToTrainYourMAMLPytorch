# A Python program to print all combinations
# of given length with unsorted input.
from itertools import combinations

all_combos = []

# Print the obtained combinations
for length in range(1, 6):
    gen_combinations = combinations([0, 1, 2, 3, 4], length)
    for combination in list(gen_combinations):
        all_combos.append(combination)

print(len(all_combos))