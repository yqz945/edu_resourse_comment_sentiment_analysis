# -*- encoding: utf-8 -*-
'''
@TIME        :  2021/7/30
@Author      :  YQZ
@Version     :  1.0
@Desc        :  评论原始数据清洗处理
'''

import re


def is_chinese(uchar):
    """
    判断是否中文字符
    """
    return u'\u4e00' <= uchar[0] <= u'\u9fa5'


def is_number(uchar):
    """判断一个unicode是否是数字"""
    return u'\u0030' <= uchar <= u'\u0039'


def clear_line(line):
    """
    去除一行的前后空格和多余标点符号
    """
    ln = ''
    for c in line:
        if is_chinese(c) or is_number(c):
            ln = ln + c
    ln = ln + '\n'
    return ln


def remove_invalid_comments():
    """
    尽量匹配清楚无效评论
    """
    text = ''
    lines = []
    with open('data/comments.csv', 'r', encoding='utf-8') as file:
        with open(r'data/comments_clean.csv', 'w', encoding='utf-8') as new_f:
            for ln in file:
                res = re.findall(r"^[a-zA-Z0-9!？'<>\s\"\\.,。，]+|(测试)+$", ln)
                if len(res) <= 0:
                    ln = clear_line(ln)
                    if len(ln) > 0:
                        if ln not in lines:
                            new_f.write(ln)
                            lines.append(ln)


# The main if needed
if __name__ == '__main__':
    remove_invalid_comments()
