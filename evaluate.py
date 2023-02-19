""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""

import argparse
import json
import glob
import logging
import os
import random
import re
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from fastprogress.fastprogress import master_bar, progress_bar
from attrdict import AttrDict
####custom processor for korquad v2
from Korquad import KorquadV2Processor

from src import (
    eval_during_train,
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    MODEL_FOR_QUESTION_ANSWERING,
    init_logger,
    set_seed
)

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

logger = logging.getLogger(__name__)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def evaluate(args, model, tokenizer, global_step=None):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(global_step))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()
    batch_num = 0
    
    
    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        # print(batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],#(bs, 512)
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "distilkobert", "xlm-roberta"]:
                del inputs["token_type_ids"]

            example_indices = batch[3]

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            # output = [to_list(output[i]) for output in outputs]
            output = [to_list(outputs[0][i])]
            output.append(to_list(outputs[1][i]))

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
            tokenizer = TOKENIZER_CLASSES["koelectra-base-v3"].from_pretrained(
       "monologg/koelectra-base-v3-discriminator",
        do_lower_case=False,
    )
            print(tokenizer.decode(inputs['input_ids'][0]))
            # print(len(result.start_logits))
            start_position = np.array(result.start_logits).argmax()
            print("argmax : ", np.array(result.start_logits).argmax())
            # print(len(result.end_logits))
            end_position = np.array(result.end_logits).argmax()
            print("argmax : ", np.array(result.end_logits).argmax())
            print(tokenizer.decode(inputs['input_ids'][0][start_position:end_position+1]))
            
            all_results.append(result)
            if i == 5 :
                break
        if batch_num ==3:
            break
    return

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(global_step))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(global_step))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(global_step))
    else:
        output_null_log_odds_file = None

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    # Write the result
    # Write the evaluation result on file
    output_dir = os.path.join(args.output_dir, 'eval')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir, "eval_result_{}_{}.txt".format(list(filter(None, args.model_name_or_path.split("/"))).pop(),
                                                                               global_step))

    with open(output_eval_file, "w", encoding='utf-8') as f:
        official_eval_results = eval_during_train(args, step=global_step)
        results.update(official_eval_results)

    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            # processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            processor = KorquadV2Processor()
            if evaluate:
                examples = processor.get_dev_examples(os.path.join(args.data_dir, args.task),
                                                      filename=args.predict_file)
            else:
                examples = processor.get_train_examples(os.path.join(args.data_dir, args.task),
                                                        filename=args.train_file)
        print("examples length : ",len(examples))
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset


def main(cli_args):
    # Read from config file and make args
    with open(os.path.join(cli_args.config_dir, cli_args.task, cli_args.config_file)) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    init_logger()
    set_seed(args)

    logging.getLogger("transformers.data.metrics.squad_metrics").setLevel(logging.WARN)  # Reduce model loading logs

    # Load pretrained model and tokenizer
    config = CONFIG_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
    )
    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
    )
    model = MODEL_FOR_QUESTION_ANSWERING[args.model_type].from_pretrained(
        args.model_name_or_path,
        config=config,
    )
    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    checkpoint = "/home/kic/Desktop/KoELECTRA/finetune/ckpt/koelectra-base-v3-korquad-ckpt/checkpoint-40000"
    model = MODEL_FOR_QUESTION_ANSWERING[args.model_type].from_pretrained(checkpoint)
    model.to(args.device)
    evaluate(args, model, tokenizer, global_step=None)

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--task", type=str, required=True)
    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, required=True)

    cli_args = cli_parser.parse_args()

    main(cli_args)
