import pickle
# list1 = [3,5,-4,-1,0,-2,-6]
# print(sorted(list1, key=lambda x: abs(x)))

# with open( '../data/dev.pickle', 'rb') as f:
#     dev_data = pickle.load(f)
# print(len(dev_data))
# print(dev_data[0:3])

import json

with open('../data/id2word.obj', 'rb') as f:
    dict = pickle.load(f)
print(type(dict[1]))
with open("../pre_vector.json", 'rb') as f:
    id2word = json.load(f)
print(id2word['，'])