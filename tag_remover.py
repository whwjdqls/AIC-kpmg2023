#Remove html tags
import os
import json
import argparse
from attrdict import AttrDict
from tqdm import tqdm
import re
import os.path
from zipfile import ZipFile
import glob
    
def _remove_tags(input_data, args, file_path):
    print("remove html tags from json")
    success_answer=0
    total_answer=0
    modified_context=0
    total_context=0
    
    #REMOVE TAGS
    TAGS = '<span>|<a>|</span>|</a>|<b>|</b>|<i>|</i>|<u>|</u>|<small>|</small>|<sup>|</sup>|<sub>|</sub>|<strong>|</strong>|<em>|</em>'
    
    #Load data
    for entry in input_data:
        title = entry["title"]
        context_text_origin = entry["context"]
        
        #Remove tags
        context_text = re.sub(TAGS, '', context_text_origin)
        entry['context'] = context_text #modified context
        
        total_context+=len(context_text_origin)
        modified_context += len(context_text)        
        i=0
        for qa in entry["qas"]:
            total_answer+=1            
            answer = qa["answer"]
            answer_text_origin=answer['text']
            start_position_origin = answer["answer_start"]

            #Remove tags
            answer_text = re.sub(TAGS,'',answer_text_origin) #modified answer_text
            context_text_bf_an = context_text_origin[:start_position_origin]
            context_text_bf_an = re.sub(TAGS,'',context_text_bf_an)
            start_position = len(context_text_bf_an) #modified start_position
                
            #Compare original answer and modified answer
            if answer_text==context_text[start_position:start_position+len(answer_text)]:
                success_answer+=1
            else:
                print('from: ',answer_text_origin)
                print('tooo: ',context_text[start_position:start_position+len(answer_text)])
            
            #MODIFYING    
            answer['text']=answer_text
            answer['answer_start']=start_position
        
    #SAVE
    input_data={"version": "KorQuAD_2.0_train", "data": input_data}
    with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(input_data, file)
    
    #Result            
    print('{}/{} successed'.format(success_answer, total_answer))
    print('removed from {} to {}'.format(total_context, modified_context))        
        
            
            
                

def main(cli_args):
    # Read from config file and make args
    with open(os.path.join(cli_args.config_dir, cli_args.task, cli_args.config_file)) as f:
        args = AttrDict(json.load(f))
    
    train_files = list(path for path in sorted(glob.glob(os.path.join(args.data_dir, args.task, 'KorQuAD', 'train')+'/*.json')))
    dev_files = list(path for path in sorted(glob.glob(os.path.join(args.data_dir, args.task, 'KorQuAD', 'dev')+'/*.json')))

    #for train_file in train_files[:4]:
    #    ZipFile(train_file).extractall(os.path.dirname(train_file))
    print(train_files)
    #Load json config file
    for train_file in train_files[0:9]:
        with open(train_file, "r", encoding="utf-8") as reader:
            input_data = json.load(reader)["data"]    
        file_path = os.path.join(args.data_dir, args.task, 'KorQuAD/train/new_') + os.path.basename(train_file)
        print('remove tags from {} to {}'.format(train_file, file_path))    
        _remove_tags(input_data, args, file_path)
        
    for dev_file in dev_files[0:2]:   
        with open(dev_file, "r", encoding="utf-8") as reader:
            input_data = json.load(reader)["data"]    
        file_path = os.path.join(args.data_dir, args.task, 'KorQuAD/dev/new_') + os.path.basename(dev_file)
        print('remove tags from {} to {}'.format(dev_file, file_path))    
        #_remove_tags(input_data, args, file_path)

    


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--task", type=str, required=True)
    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, required=True)

    cli_args = cli_parser.parse_args()

    main(cli_args)