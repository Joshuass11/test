# --coding: utf-8-
"""
@Time : 2022/4/25 10:25
@Author : 吴双
@File : clip_msra.py
@Software: PyCharm
"""
from utils.paths import msra_ner_cn_path, ontonote4ner_cn_path

#翻转文件，对下载的数据预处理
def create_cliped_file(fp, clip_len):
    f = open(fp,'r',encoding='utf-8')
    fp_out = fp + '_clip'
    f_out = open(fp_out,'w',encoding='utf-8')
    now_example_len = 0

    lines = f.readlines()
    last_line_split = ['','']
    for line in lines:
        line_split = line.strip().split()

        print(line,end='',file=f_out)
        now_example_len += 1
        if len(line_split) == 0 or \
                (line_split[0] in ['。','！','？']
                 and line_split[1] == 'O' and now_example_len > clip_len):
            print('',file=f_out)
            now_example_len = 0
        elif ((line_split[0] in ['，','；'] or (now_example_len > 1 and last_line_split[0] == '…' and line_split[0] == '…'))
                 and line_split[1] == 'O' and now_example_len > clip_len):
            print('',file=f_out)
            now_example_len = 0

        elif line_split[1][0].lower() == 'e' and now_example_len > clip_len:
            print('',file=f_out)
            now_example_len = 0

        last_line_split = line_split

    f_out.close()
    f_check = open(fp_out,'r',encoding='utf-8')
    lines = f_check.readlines()
    cliped_examples = [[]]
    now_example = cliped_examples[0]
    for line in lines:
        line_split = line.strip().split()
        if len(line_split) == 0:
            cliped_examples.append([])
            now_example = cliped_examples[-1]
        else:
            now_example.append(line.strip())

    check = 0
    max_length = 0
    for example in cliped_examples:
        if len(example)>200:
            print(len(example),''.join(map(lambda x:x.split(' ')[0],example)))
            check = 1

        max_length = max(max_length,len(example))

    print('最长的句子有:{}'.format(max_length))

    if check == 0:
        print('没句子超过200的长度')
#msra
#下面这个运行一半list index out of range
create_cliped_file(r'{}/train_dev.char.bmes'.format(msra_ner_cn_path), 210)
#下面这个可以正常运行
create_cliped_file(r'{}/test.char.bmes'.format(msra_ner_cn_path), 210)

#ontonote
#create_cliped_file(r'{}/train.char.bmes'.format(ontonote4ner_cn_path), 180)
#create_cliped_file(r'{}/dev.char.bmes'.format(ontonote4ner_cn_path), 180)
#create_cliped_file(r'{}/test.char.bmes'.format(ontonote4ner_cn_path), 180)

#weibo
#create_cliped_file(r'{}/train.char.bmes'.format(ontonote4ner_cn_path), 180)
#create_cliped_file(r'{}/dev.char.bmes'.format(ontonote4ner_cn_path), 180)
#create_cliped_file(r'{}/test.char.bmes'.format(ontonote4ner_cn_path), 180)
