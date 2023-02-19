import json
import os
import pandas as pd
import numpy as np
import re, html
from bs4 import BeautifulSoup as BS, NavigableString, SoupStrainer
import codecs
import random
import argparse
class answer():
    def __init__(self,answer_text=None, answer_start=None):
        self.answer_text = answer_text
        self.answer_start = answer_start
    
class qas():
    def __init__(self,question = None , answer = None, id_ = None):
        self.question = question
        self.answer = answer
        self.id_ = id_
     
class korquad():
    def __init__(self,context=None, title ="", qas=None):
        self.context = context
        self.title = title
        self.qas = qas #list of qass

def type_in_answer(context, text):
    answer_start = None
    find_start = 0
    while(1):
        ind = context.find(text,find_start)
        if ind == -1:
            print("no answer in context. type in the answer again!")
            break
        else:
            print(f"found answer in context at index {ind}")
            is_desired = input(f"is this position the desired position?[y/n] \n{context[ind-5:ind+len(text)+5]}")
            if is_desired=="y":
                answer_start = ind
                break
            else:
                find_start = ind+1
    return answer_start, text

def make_dataset(context_full, context_length = 900, title = "", id_ = "12", for_table = False):
    full_length = len(context_full)
    while(1):
        if for_table:
            context_start = context_full.find("<table>")
            if context_start<0:
                print(f"no table starting from 0")
                return
        else:
            context_start = int(random.random()* (full_length-context_length-1))
        context = context_full[context_start:context_start+context_length]
#         table_start = context.find("<table>")
#         table_end = context.find("</table>")
#         if table_start < table_end:
#             print(context[:table_start+1])
#             print(html_to_json(context[table_start:table_end+7]))
#             print(context[table_end:])
        print(context)
        a = input("confirm context? [y/n] (if file too short, press x)")
        if a == 'y':
            break
        elif a == "x":
            return None
        else:
            print("-------new context-------")
    question = input("type in a question for the above context")
    # answer_text = input("type in the answer for the above question")
    find_start = 0

    while(1):
        answer_text = input("type in the answer for the above question")
        answer_start,temp = type_in_answer(context,answer_text)
        if answer_start :
            answer_text = temp
            break

    new_answer = answer(answer_text=answer_text,answer_start=answer_start)
    new_qas = qas(question=question,answer=new_answer,id_ = id_)
    qas_list = []
    qas_list.append(new_qas)
    new_korquad = korquad(context=context,title = title, qas=qas_list) 
#     print("question : ", question)
#     print("answer :", answer_text)
#     print("cntext[answer_start]",context[answer_start+10] )
    return new_korquad
    

name_dict = {"지수":1, "수민":2,"정빈":3,"찬혁":4,"건우":5,"영한":6}

def main(args):
    
    print(f"------------------making custom data for {args.name}-------------")
    id_start = name_dict[args.name]*100000
    print(args.data_dir)
    data_dir = args.data_dir
    data_num = args.data_num
    for_table = args.for_table
    cnt = 0
    korquad_obj_list = []

    for file_name in os.listdir(data_dir):
        print(file_name)
        file_pth = os.path.join(data_dir,file_name)
        f=codecs.open(file_pth, 'r', encoding='UTF8')
        korquad_obj=make_dataset(f.read(),id_=id_start+cnt, for_table = for_table)
        if korquad_obj:
            korquad_obj_list.append(korquad_obj)
            cnt=+1

        if cnt == data_num:
            break





if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--data_dir", type=str, required=True)
    cli_parser.add_argument("--data_num", type=int, required=False, default = 10)
    cli_parser.add_argument("--context_len", type=int, required=False, default = 1200)
    cli_parser.add_argument("--name", type=str, required=True)
    cli_parser.add_argument("--for_table", type=bool, required=False, defult = False)
    cli_args = cli_parser.parse_args()

    main(cli_args)