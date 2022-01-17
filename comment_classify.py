# -*- encoding: utf-8 -*-
'''
@TIME        :  2021/8/4
@Author      :  YQZ
@Version     :  1.0
@Desc        :  教育资源评论分类器
'''

import sys, getopt
import jieba
import pandas as pd

# The main if needed
if __name__ == '__main__':
    s_input = ''
    opts = []
    args = []

    try:
        opts, args = getopt.getopt(sys.argv[1:], '-c:')
    except getopt.GetoptError:
        print("comment_classify.py -c <'评论内容'>")

    for opt, arg in opts:
        if opt == '-c':
            s_input = arg

    df = pd.read_csv('data/data.csv', encoding='utf-8', header=None, names=['w', 'pos', 'neg', 'lambda'])

    l_v = []
    l_ws = jieba.lcut(s_input, use_paddle=True)
    print(l_ws)

    for w in l_ws:
        ln = df.loc[df['w'] == w]
        if len(ln) > 0:
            print("{0},{1}".format(ln['w'].values[0], ln['lambda'].values[0]))
            l_v.append(ln['lambda'].values[0])

    df_prior = pd.read_csv('data/logprior.csv', encoding='utf-8', header=None, names=['pos', 'neg', 'logprior'])

    ret_val = 0
    for v in l_v:
        ret_val = ret_val + v
    ret_val = ret_val + df_prior['logprior'].values[0]

    print(ret_val)