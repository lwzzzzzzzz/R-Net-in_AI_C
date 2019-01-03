from __future__ import print_function
import tensorflow as tf
import numpy as np
import codecs
import jieba
import json
import os
from collections import Counter
import pickle

def seg_line(line):
    return list(jieba.cut(line))

def seg_data(path):
    print('start process ', path)
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            dic = json.loads(line, encoding='utf-8')
            question = dic['query']
            doc = dic['passage']
            alternatives = dic['alternatives']
            data.append([seg_line(question), seg_line(doc), alternatives.split('|'), dic['query_id']])
    return data


def read_data(data_name, opts):
    return DataProcessor(data_name, opts)

class DataProcessor:
    def __init__(self, data_name, opts):
        self.data_name = data_name
        self.opts = opts
        # load预处理好的数据
        data_path = os.path.join('data', "{}.pickle".format(data_name))
        id2word_path = os.path.join('data', "id2word.obj")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        with open(id2word_path, 'rb') as f:
            self.id2word = pickle.load(f)
        with open("pre_vector.json", 'rb') as f:
            self.vector = json.load(f)

        # paragraph length filter: (train only)
        # if self.data_name == 'train':
        #     # 对于位置太靠后(在opts['p_length'] 300以后)的answer，直接丢掉
        #     self.data = [sample for sample in self.data if sample['answer'][0][-1] < self.opts['p_length']]
        self.num_samples = self.get_data_size()
        print("Loaded {} examples from {}".format(self.num_samples, data_name))

    def load_data(self, path):
        with open(path, 'r') as fh:
            data = json.load(fh)
        return data

    def get_data_size(self):
        return len(self.data)

    # 得到一个batch		还是tf.train.batch好用
    def get_training_batch(self, batch_no):
        opts = self.opts
        # 标记该batch在data中的起始和终止数据下标
        si = (batch_no * opts['batch_size'])
        ei = min(self.num_samples, si + opts['batch_size'])
        n = ei - si

        tensor_dict = {}
        paragraph = np.zeros((n, opts['p_length'], opts['word_emb_dim']))
        question = np.zeros((n, opts['q_length'], opts['word_emb_dim']))
        answer = np.zeros((n, 3, opts['a_length'], opts['word_emb_dim']))
        idxs = []

        count = 0
        for i in range(si, ei):
            idxs.append(i)
            sample = self.data[i]
            p = sample[1]
            q = sample[0]
            a = sample[2] #[[],[],[]]

            for j in range(len(p)):
                # 当paragragh长于opts['p_length']，后面的丢掉不要，只取前面opts['p_length']个单词
                if j >= opts['p_length']:
                    break
                try:
                    # 把glove中的对应赋给nparray，count指示batch中第几个 j表示第几个单词
                    paragraph[count][j][:opts['word_emb_dim']] = self.vector[self.id2word[p[j]]]
                except KeyError:
                    pass

            for j in range(len(q)):
                if j >= opts['q_length']:
                    break
                try:
                    question[count][j][:opts['word_emb_dim']] = self.vector[self.id2word[q[j]]]
                except KeyError:
                    pass

            for j in range(len(a)):
                for k in range(len(a[j])):
                    if k >= opts['a_length']:
                        break
                    try:
                        answer[count][j][k][:opts['word_emb_dim']] = self.vector[self.id2word[a[j][k]]]
                    except KeyError:
                        pass
            count += 1

        tensor_dict['paragraph'] = paragraph
        tensor_dict['question'] = question
        tensor_dict['answer'] = answer
        return tensor_dict, idxs

    def get_testing_batch(self, batch_no):
        opts = self.opts
        si = (batch_no * opts['batch_size'])
        ei = min(self.num_samples, si + opts['batch_size'])
        n = ei - si

        paragraph = np.zeros((opts['batch_size'], opts['p_length'], opts['word_emb_dim']))
        question = np.zeros((opts['batch_size'], opts['q_length'], opts['word_emb_dim']))
        ID = [None for _ in range(n)]
        context = [None for _ in range(n)]

        count = 0
        for i in range(si, ei):
            sample = self.data[i]
            p = sample[1]
            q = sample[0]
            a = sample[2] #[[],[],[]]

            for j in range(len(p)):
                # 当paragragh长于opts['p_length']，后面的丢掉不要，只取前面opts['p_length']个单词
                if j >= opts['p_length']:
                    break
                try:
                    # 把glove中的对应赋给nparray，count指示batch中第几个 j表示第几个单词
                    paragraph[count][j][:opts['word_emb_dim']] = self.vector[self.id2word[p[j]]]
                except KeyError:
                    pass

            for j in range(len(q)):
                if j >= opts['q_length']:
                    break
                try:
                    question[count][j][:opts['word_emb_dim']] = self.vector[self.id2word[q[j]]]
                except KeyError:
                    pass

            count += 1


            ID[count] = sample[3]
            context[count] = sample[4]
            count += 1
        # test的batch返回并没有pack起来
        # context是batch内的问题对应的分词后的context，因为一段context可能有多个问题，所以batch内可能很多都一样
        # context_original同理，但是就是未处理的原文
        # paragraph, question, paragraph_c, question_c, answer_si, answer_ei和trainning batch的一样，只不过没有封装在dict里
        # ID是问题的唯一标号
        # n就是batch_size
        return paragraph, question, ID, context, n


def build_word_count(data, threshold):
    wordCount = Counter()
    wordCount_subMin = Counter()
    def add_count(lst):
        for word in lst:
            if word not in wordCount:
                wordCount[word] = 0
            wordCount[word] += 1

    for one in data:
        # one[0:3]去掉id
        [add_count(x) for x in one[0:3]]

    for word in wordCount:
        if wordCount[word] >= threshold:
            wordCount_subMin[word] += 1
        else:
            chars = list(word)
            for char in chars:
                wordCount_subMin[char] += 1
    print('word type size ', len(wordCount_subMin))
    return wordCount_subMin

def get_word2vec(sgns_path, word_counter):
    word2vec_dict = {}
    with open(sgns_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            # 可以看出word_counter对于大小写区分的test	Test  TEST虽然对应的vector一样，但会作为不同的个体存放在word2vec_dict
            if word in word_counter:
                word2vec_dict[word] = vector
            if word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            if word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            if word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector
    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), sgns_path))
    return word2vec_dict

def transform_data_to_id(raw_data, word2id, is_test=False):
    data = []

    def map_word_to_id(word):
        output = []
        if word in word2id:
            output.append(word2id[word])
        else:
            chars = list(word)
            for char in chars:
                if char in word2id:
                    output.append(word2id[char])
                else:
                    output.append(1)
        return output

    def map_sent_to_id(sent):
        output = []
        for word in sent:
            output.extend(map_word_to_id(word))
        return output
    if is_test:
        for one in raw_data:
            question = map_sent_to_id(one[0])
            doc = map_sent_to_id(one[1])
            candidates = [map_word_to_id(x) for x in one[2]]
            length = [len(x) for x in candidates]
            max_length = max(length)
            if max_length > 1:
                pad_len = [max_length - x for x in length]
                candidates = [x[0] + [0] * x[1] for x in zip(candidates, pad_len)]
            data.append([question, doc, candidates, one[-1], one[-2]])
        return data
    else:
        for one in raw_data:
            question = map_sent_to_id(one[0])
            doc = map_sent_to_id(one[1])
            candidates = [map_word_to_id(x) for x in one[2]]
            length = [len(x) for x in candidates]
            max_length = max(length)
            if max_length > 1:
                pad_len = [max_length - x for x in length]
                candidates = [x[0] + [0] * x[1] for x in zip(candidates, pad_len)]
            data.append([question, doc, candidates, one[-1]])
        return data



def process_data(data_path, threshold=5):
    train_file_path = data_path + 'ai_challenger_oqmrc_validationset_20180816/ai_challenger_oqmrc_validationset.json'
    dev_file_path = data_path + 'ai_challenger_oqmrc_trainingset_20180816/ai_challenger_oqmrc_trainingset.json'
    # test_a_file_path = data_path + 'ai_challenger_oqmrc_testa_20180816/ai_challenger_oqmrc_testa.json'
    # test_b_file_path = data_path + 'ai_challenger_oqmrc_testb_20180816/ai_challenger_oqmrc_testb.json'
    # path_lst = [train_file_path, dev_file_path, test_a_file_path, test_b_file_path]
    path_lst = [train_file_path, dev_file_path]
    # output_path = [data_path + x for x in ['train.pickle', 'dev.pickle', 'testa.pickle', 'testb.pickle']]
    output_path = [data_path + x for x in ['train.pickle', 'dev.pickle']]
    return _process_data('./data/sgns.wiki.bigram-char', path_lst, threshold, output_path)


def _process_data(sgns_path, path_lst, threshold=5, output_file_path=[]):
    raw_data = []
    for path in path_lst:
        raw_data.append(seg_data(path))
    word_count = build_word_count([y for x in raw_data for y in x], threshold=threshold)
    with open('data/word-count.obj', 'wb') as f:
        pickle.dump(word_count, f)

    for one_raw_data, one_output_file_path in zip(raw_data, output_file_path):
        with open(one_output_file_path, 'wb') as f:
            one_data = get_word2vec(sgns_path, word_count)
            print(one_data)
            pickle.dump(one_data, f)

if __name__ == '__main__':
    # sgns_path = './data/sgns.wiki.bigram-char'
    # # 统计词频
    # word2id = build_word2id(wordCount, 0)
    # print(word2id)
    # data = transform_data_to_id(data,word2id)
    # print(data)
    # make_vocab(wordCount, "train.tsv")
    process_data("./data/")
    print("Done")