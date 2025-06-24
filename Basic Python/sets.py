# Creating a set
my_set = {1, 2, 3, 4, 5}

# Adding elements
my_set.add(6)
print(my_set)

# Updating elements
my_set.update([7, 8, 9])
print(my_set)

# Removing elements
my_set.remove(3)
print(my_set)

# Discarding elements
my_set.discard(2)
print(my_set)

# Popping elements
popped_element = my_set.pop()
print(popped_element)

# Clearing the set
my_set.clear()
print (my_set)

# Set operations
set1 = {1, 2, 3}
set2 = {3, 4, 5}
union_set = set1.union(set2)
print(union_set)
intersection_set = set1.intersection(set2)
print(intersection_set)
difference_set = set1.difference(set2)
print(difference_set)
sym_diff_set = set1.symmetric_difference(set2)
print(sym_diff_set)
is_subset = set1.issubset(set2)
print(is_subset)
is_superset = set1.issuperset(set2)
print(is_superset)