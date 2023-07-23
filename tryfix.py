def remove_extra_keys(list1, list2):
    keys_set1 = set().union(*(d.keys() for d in list1))
    keys_set2 = set().union(*(d.keys() for d in list2))
    common_keys = keys_set1.intersection(keys_set2)

    for item in list1:
        item_keys = set(item.keys())
        extra_keys = item_keys - common_keys
        for key in extra_keys:
            del item[key]

    for item in list2:
        item_keys = set(item.keys())
        extra_keys = item_keys - common_keys
        for key in extra_keys:
            del item[key]

    return list1, list2

# Example lists of dictionaries
list1 = [
    {"a": 1, "b": 2, "c": 3},
    {"b": 5, "c": 6, "d": 7}
]

list2 = [
    {"a": 10, "b": 20, "c": 30},
    {"b": 50, "c": 60, "e": 70}
]

list1, list2 = remove_extra_keys(list1, list2)
print("List 1 after removing extra keys:")
print(list1)
print("List 2 after removing extra keys:")
print(list2)
