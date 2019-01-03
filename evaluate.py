import prepro
from Models import model_rnet
import numpy as np
import tensorflow as tf
import argparse
import codecs
import json
from pprint import pprint
import os

os.environ["CUDA_VISIBLE_DEVICES"]= '0,1'

def run():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='rnet', help='Model: match_lstm, bidaf, rnet')
	parser.add_argument('--debug', type=bool, default=False, help='print debug msgs')
	parser.add_argument('--dataset', type=str, default='testa', help='dataset')
	parser.add_argument('--model_path', type=str, default='Models/save/rnet_model_final.ckpt', help='saved model path')

	args = parser.parse_args()
	if not args.model == 'rnet':
		raise NotImplementedError

	modOpts = json.load(open('Models/config.json','r'))[args.model]['dev']
	print('Model Configs:')
	pprint(modOpts)

	print('Reading data')
	if args.dataset == 'train':
		raise NotImplementedError
	elif args.dataset == 'testa':
		dp = prepro.read_data(args.dataset, modOpts)
    
	model = model_rnet.R_NET(modOpts)
	input_tensors, loss, acc, pred = model.build_model()
	saved_model = args.model_path


	num_batches = int(np.ceil(dp.num_samples/modOpts['batch_size']))
	print(num_batches, 'batches')
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	new_saver = tf.train.Saver()
	sess = tf.InteractiveSession(config=config)
	new_saver.restore(sess, saved_model)

	predictions = []

	for batch_no in range(num_batches):
		if args.model == 'rnet':
			paragraph, question, answer, ID, context, n = dp.get_testing_batch(batch_no)
			feed_dict={
				input_tensors['p']: paragraph,
				input_tensors['q']: question,
				input_tensors['a']: answer,
			}

			pred_vec = sess.run(pred, feed_dict=feed_dict)
			pred_vec = np.argmax(pred_vec, axis=1)
			for q_id, prediction, candidates in zip(ID, pred_vec, context):
				prediction_answer = u''.join(candidates[prediction])
				predictions.append(str(q_id) + '\t' + prediction_answer)
	outputs = u'\n'.join(predictions)
	with codecs.open('Results/prediction.a.txt', 'w',encoding='utf-8') as f:
		f.write(outputs)
	print('done!')
def f1_score(prediction, ground_truth):
	from collections import Counter

	prediction_tokens = prediction
	ground_truth_tokens = ground_truth
	# min(Counter_prediction_tokens['x']), Counter(ground_truth_tokens)['x'])  返回的也是Counter
	common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
	num_same = sum(common.values())
	if num_same == 0:
		return 0
	# 这里的prediction_tokens可以理解为预测为正例的集合		precision指标就是	正确预测为正占所有预测为正的比例(有些不应该在common中的被错误预测为正的)
	# ground_truth_tokens理解为实际为正例的集合		recall指标就是  所有正例中，被正确划分的比例(应该在但没有被选在common里的认为分错类了)
	precision = 1.0 * num_same / len(prediction_tokens)
	recall = 1.0 * num_same / len(ground_truth_tokens)
	#f1 得分就是recall和precision的调和平均
	f1 = (2 * precision * recall) / (precision + recall)
	return f1

if __name__ == '__main__':
	run()
