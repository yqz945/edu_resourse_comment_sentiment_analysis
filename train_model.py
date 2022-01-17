# -*- encoding: utf-8 -*-
'''
@TIME        :  2021/8/2
@Author      :  YQZ
@Version     :  1.0
@Desc        :  基于打标评论，建立词库，训练模型数据
'''

import numpy as np
import pandas as pd
import jieba
import jieba.posseg as pseg

l_labels = []

jieba.enable_paddle()

def build_vocabulary():
    """
    基于初始的打标的评论（comments_clean_labeled.csv）构建词库
    后续新增的评论网词库里增加
    """
    df_coms = pd.read_csv('data/comments_clean_labeled.csv', encoding='utf-8')

    #stops = ('r', 'u', 'nr', 'm', 'q', 'xc', 'PER', 'ns', 'LOC', 'nt', 'ORG', 'nw', 'nz', 'TIME', 'c', 'd')
    stops = ('r', 'u', 'nr', 'm', 'q', 'xc', 'PER', 'ns', 'LOC', 'nt', 'ORG', 'nw', 'nz', 'TIME')

    vocabulary = set()
    for row in df_coms.itertuples():
        l_words = pseg.lcut(row.comment, use_paddle=True)
        l_labels.append(row.label)
        for s_word in l_words:
            if s_word.flag not in stops:
                vocabulary.add(s_word.word)  # 因为是set，每次打印出来的不一样

    # 写入词库 vocabulary.txt
    with open('data/vocabulary.txt', 'w', encoding='utf-8') as file:
        for word in vocabulary:
            file.write("{0}\n".format(word))


def make_wc():
    """
    基于评论文本和词库，生成正负词列表
    """
    vocabulary = set()
    with open('data/vocabulary.txt', 'r', encoding='utf-8') as file:
        for ln in file:
            vocabulary.add(ln.strip())

    df_coms = pd.read_csv('data/comments_clean_labeled.csv', encoding='utf-8')

    with open('data/wc_1.txt', 'w', encoding='utf-8') as file_1:
        with open('data/wc_0.txt', 'w', encoding='utf-8') as file_0:
            for row in df_coms.itertuples():
                l_words = jieba.lcut(row.comment, use_paddle=True)
                if row.label == 1:
                    for s_word in l_words:
                        if s_word in vocabulary:
                            file_1.write("{0}\n".format(s_word))
                elif row.label == 0:
                    for s_word in l_words:
                        if s_word in vocabulary:
                            file_0.write("{0}\n".format(s_word))


def freq():
    """
    统计词频，计算词的正负概率，lambda和logprior
    """
    vocabulary = set()
    with open('data/vocabulary.txt', 'r', encoding='utf-8') as file:
        for ln in file:
            vocabulary.add(ln.strip())
    # 统计正负词频
    dict_1 = dict(zip(list(vocabulary), np.zeros(len(vocabulary), dtype=int)))
    dict_0 = dict(zip(list(vocabulary), np.zeros(len(vocabulary), dtype=int)))

    with open('data/wc_1.txt', 'r', encoding='utf-8') as file_1:
        for ln in file_1:
            dict_1[ln.strip()] = dict_1[ln.strip()] + 1

    with open('data/wc_0.txt', 'r', encoding='utf-8') as file_0:
        for ln in file_0:
            dict_0[ln.strip()] = dict_0[ln.strip()] + 1

    # 计算正负总次数
    t_1 = 0
    t_0 = 0
    for it in dict_1.values():
        t_1 = t_1 + it
    for it in dict_0.values():
        t_0 = t_0 + it

    with open('data/out_1.txt', 'w', encoding='utf-8') as file:
        for w in vocabulary:
            file.write("{0} {1} {2}\n".format(w, str(dict_1[w]), str(dict_0[w])))
        file.write("total {0} {1}\n".format(str(t_1), str(t_0)))

    # 统计正负词的百分比
    dp_1 = dict(zip(list(vocabulary), np.zeros(len(vocabulary), dtype=float)))
    dp_0 = dict(zip(list(vocabulary), np.zeros(len(vocabulary), dtype=float)))

    # 加入laplacian算子
    for k, v in dict_1.items():
        dp_1[k] = (dict_1[k] + 1) / (t_1 + len(vocabulary))
    for k, v in dict_0.items():
        dp_0[k] = (dict_0[k] + 1) / (t_0 + len(vocabulary))

    # 计算正负总次数
    tp_1 = 0
    tp_0 = 0
    for it in dp_1.values():
        tp_1 = tp_1 + it
    for it in dp_0.values():
        tp_0 = tp_0 + it

    with open('data/out_2.txt', 'w', encoding='utf-8') as file:
        for w in vocabulary:
            file.write("{0} {1} {2}\n".format(w, str(dp_1[w]), str(dp_0[w])))
        file.write("total {0} {1}\n".format(str(tp_1), str(tp_0)))

    # 计算lambda
    lam = dict(zip(list(vocabulary), np.zeros(len(vocabulary), dtype=float)))
    for k, v in dp_1.items():
        lam[k] = np.log10(dp_1[k] / dp_0[k])

    with open('data/data.csv', 'w', encoding='utf-8') as file:
        for w in vocabulary:
            file.write("{0},{1},{2},{3} \n".format(w, str(dp_1[w]), str(dp_0[w]), str(lam[w])))

    # 计算logprior
    d_pos = 0
    d_neg = 0
    df_cs = pd.read_csv('data/comments_clean_labeled.csv', encoding='utf-8')
    for row in df_cs.itertuples():
        if row.label == 1:
            d_pos = d_pos + 1
        elif row.label == 0:
            d_neg = d_neg + 1
    logprior = np.log10(d_pos/d_neg)
    with open('data/logprior.csv', 'w', encoding='utf-8') as file:
        file.write("{0},{1},{2}\n".format(d_pos, d_neg, logprior))


def test_coms():
    """
    测试模型
    """
    str_com = '差评'

    # # 计算logprior
    # d_pos = 0
    # d_neg = 0
    # df_cs = pd.read_csv('data/comments_clean_labeled.csv', encoding='utf-8')
    # for row in df_cs.itertuples():
    #     if row.label == 1:
    #         d_pos = d_pos + 1
    #     elif row.label == 0:
    #         d_neg = d_neg + 1
    # logprior = np.log10(d_pos / d_neg)

    df = pd.read_csv('data/data.csv', encoding='utf-8', header=None, names=['w', 'pos', 'neg', 'lambda'])

    l_v = []
    l_ws = jieba.lcut(str_com, use_paddle=True)

    for w in l_ws:
        ln = df.loc[df['w'] == w]
        if len(ln) > 0:
            print("{0},{1}".format(ln['w'].values[0], ln['lambda'].values[0]))
            l_v.append(ln['lambda'].values[0])

    # 读入prior
    df_prior = pd.read_csv('data/logprior.csv', encoding='utf-8', header=None, names=['pos', 'neg', 'logprior'])

    ret_val = 0
    for v in l_v:
        ret_val = ret_val + v
    ret_val = ret_val + df_prior['logprior'].values[0]
    print(ret_val)


# The main if needed
if __name__ == '__main__':
    build_vocabulary()
    make_wc()
    freq()

    test_coms()
