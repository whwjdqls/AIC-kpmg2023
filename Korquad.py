import json
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import random
import numpy as np
from tqdm import tqdm
from transformers import( 
    DataProcessor
)

import transformers

class KorquadProcessor(DataProcessor):
    train_file = None
    dev_file = None

    def get_train_examples(self, data_dir, filename = None):
        """_summary_
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")  

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir, filename = None):
        """_summary_
        Returns the evaluation examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")  
            
        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev")
        
    def _create_examples(self,input_data, set_type,max_seq=900,overlap=0.1):
        is_training = set_type == "train"
        examples = []
        print("making examples from input_data")
        for entry in tqdm(input_data):
            title = entry["title"]
            context_text = entry["context"]
            for qa in entry["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position_character = None
                answer_text = None
                answers = []
                answers_2 = {}

                if "is_impossible" in qa:
                    is_impossible = qa["is_impossible"]
                else:
                    is_impossible = False

                if not is_impossible:
                    if is_training:
                        answer = qa["answer"]
                        answer_text = answer["text"]
                        start_position_character = answer["answer_start"]
                    else:
                        #answers = qa["answer"]
                        
                        #JISOO
                        answer = qa["answer"]
                        answers_2["text"] = answer["text"]
                        answers_2["answer_start"] = answer["answer_start"]
                        answers = [answers_2]
                        start_position_character = answer["answer_start"]
                        
                context_len = len(context_text)
                if context_len > max_seq:
                    stride = int(max_seq*(1-overlap))
                    for i in range(int((context_len - max_seq )/stride)+1):
                        truncated_context_text = context_text[stride*i:stride*i+max_seq]
                        example = None
                        if start_position_character == None:
                            continue
                        if start_position_character < stride*i or start_position_character > stride*i+max_seq:
                            if random.random() > 0.99:#make fake sample percentage : 1%
                                
                                #JISOO
                                if not is_training:
                                    start_position_character = None
                                    
                                example = transformers.SquadExample(
                                    qas_id=qas_id,
                                    question_text=question_text,
                                    context_text=truncated_context_text,
                                    answer_text=answer_text,
                                    start_position_character=start_position_character,
                                    title=title,
                                    is_impossible=True,
                                    answers=answers,
                                )
                        else:
                            new_start_position_character = start_position_character-(stride*i)
                            if new_start_position_character < 150 or new_start_position_character > 850:
                                continue
                            
                            #JISOO
                            if not is_training:
                                new_start_position_character = None
                                
                            example = transformers.SquadExample(
                            qas_id=qas_id,
                            question_text=question_text,
                            context_text=truncated_context_text,
                            answer_text=answer_text,
                            start_position_character=new_start_position_character,
                            title=title,
                            is_impossible=False,
                            answers=answers,
                        )
                        if example: 
                            examples.append(example)
        return examples

class KorquadV2Processor(KorquadProcessor):
    train_file = "korquad2.1_train_00.json"
    dev_file = "korquad2.1_dev_00.json"

    


# class SquadV2Processor(SquadProcessor):
#     train_file = "train-v2.0.json"
#     dev_file = "dev-v2.0.json"


