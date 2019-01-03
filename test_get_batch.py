from prepro import DataProcessor,read_data
import json
from pprint import pprint

modOpts = json.load(open('Models/config.json','r'))['rnet']['dev']
print('Model Configs:')
pprint(modOpts)

print('Reading data')
dp = read_data('dev', modOpts)
# context, context_original, paragraph, question, paragraph_c, question_c, answer_si, answer_ei, ID, n = dp.get_testing_batch(3)
# context, context_original, paragraph, question, paragraph_c, question_c, answer_si, answer_ei, ID, n
a, b = dp.get_training_batch(3)
# print(paragraph_c==a['paragraph_c'])
print(a['question'][4])
print(len(b))
