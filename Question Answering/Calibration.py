import argparse
import copy
import glob
import json
import logging
import os
from utils import post_process

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import timeit
import argparse, random, time, json, os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from common.config import register_args, load_pretrained_model, load_untrained_model
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from data.custom_squad_feature import custom_squad_convert_examples_to_features, SquadResult, SquadProcessor

from data.qa_metrics import (compute_predictions_logits, hotpot_evaluate, calib_evaluate, aggeragate_conf_from_nbest)
from itertools import chain
from utils import normalize_answer, compute_f1

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def compute_acceptance_of_nbest(nbest):
    count = 0
    predict_text = nbest[0]["text"]
    for i in range(1, len(nbest)):
        text = nbest[i]["text"]
        if len(text) <= 3:
            continue
        else:
            if text in predict_text or predict_text in text:
                count += 1
    acceptance = count
    return acceptance


def compute_correlation_of_nbest(all_nbest_json, sorted_f1, sorted_idx):
    all_nbest_json = [v for k, v in all_nbest_json.items()]
    all_nbest_json = np.array(all_nbest_json)[sorted_idx]
    results = []

    for nbest, f1 in zip(all_nbest_json, sorted_f1):
        acceptance = compute_acceptance_of_nbest(nbest)
        results.append([f1, acceptance])

    return np.array(results)


def find_first_less_than_confidence_threshold(matrix, threshold):
    for i, e in enumerate(matrix):
        if e > threshold:
            continue
        else:
            return i
    return i


def show_density(sorted_score, step=100):
    thresholds = [i / step for i in range(step)]
    confidence_rate_list = []

    for threshold in thresholds:
        num = find_first_less_than_confidence_threshold(sorted_score, threshold)
        confidence_rate_list.append(num)

    confidence_rate_list.append(0)
    confidence_rate_list = [abs(confidence_rate_list[i + 1] - confidence_rate_list[i]) for i in
                            range(len(confidence_rate_list) - 1)]
    confidence_rate_list = np.array(confidence_rate_list) / len(sorted_score)
    for i in confidence_rate_list:
        print(i)


def read_json(fname):
    with open(fname, encoding='utf-8') as f:
        return json.load(f)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def merge_predictions(dicts):
    return dict(chain(*[list(x.items()) for x in dicts]))


def remove_padding(batch, feature):
    new_batch = tuple(x[:, :len(feature.tokens)] for x in batch[:3]) + (batch[3],)
    return new_batch


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = "cached/"
    split_id = args.predict_file if evaluate else args.train_file
    split_id = os.path.basename(split_id).split('_')[-1].split(".")[0]
    cached_features_file = os.path.join(
        input_dir,
        "{}_{}_{}_{}".format(
            split_id,
            args.dataset,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )
    if args.use_model != "squad":
        cached_features_file = "cached/{}_squad_{}_roberta-base_512".format(args.use_model, args.use_model)

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        processor = SquadProcessor()
        if evaluate:
            examples = processor.get_dev_examples('.', filename=args.predict_file)
        else:
            examples = processor.get_train_examples('.', filename=args.train_file)
        features, dataset = custom_squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
            dataset=args.dataset,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args.do_debug:
        dataset, examples, features = torch.utils.data.TensorDataset(*dataset[:50]), examples[:50], features[:50]

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.disable_tqdm):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            # output = [to_list(output[i]) for output in outputs]
            output = [outputs[0][i], outputs[1][i]]

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

                # end_logits = torch.nn.functional.normalize(end_logits, p=2, dim=0)
                # start_logits = torch.nn.functional.normalize(start_logits, p=2, dim=0)

                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    predictions, all_nbest_json = compute_predictions_logits(
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
        args.dataset,
        output_probs=True
    )

    # predictions = read_json(
    # 	'baselines_Kamath_LIMECAL/calib_exp/data/predictions/dev_{}.json/predictions_.json'.format(
    # 		args.predict_file.split("_")[1].split(".")[0]))
    # Compute the F1 and exact scores.
    results, exact, f1 = hotpot_evaluate(examples, predictions)
    logger.info("Please check out point or nbest")
    FAUC, AUC, ACC = calib_evaluate(all_nbest_json, exact, f1)
    # FAUC, AUC, ACC = calib_seperate_point_evaluate(all_results, exact, f1)
    return results, FAUC, AUC, ACC


def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, confs, accs):
    filtered_tuples = [x for x in zip(accs, confs) if x[-1] > conf_thresh_lower and x[-1] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0, 0, 0
    else:
        correct = sum([x[0] for x in filtered_tuples if x[0] != 0])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[-1] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct) / len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin


def get_bin_info(confs, accs, bin_size=0.1):
    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)

    accuracies = []
    confidences = []
    bin_lengths = []

    for conf_thresh in upper_bounds:
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh - bin_size, conf_thresh, confs, accs)
        accuracies.append(acc)
        confidences.append(avg_conf)
        bin_lengths.append(len_bin)

    return accuracies, confidences, bin_lengths


def ECE(confs, accs, bin_size=0.1):
    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)  # Get bounds of bins

    n = len(confs)
    ece = 0  # Starting error
    overconfident_ece = 0
    underconfident_ece = 0
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh - bin_size, conf_thresh, confs, accs)
        ece += np.abs(acc - avg_conf) * len_bin / n  # Add weigthed difference to ECE
        if acc > avg_conf:
            underconfident_ece += np.abs(acc - avg_conf) * len_bin / n
        else:
            overconfident_ece += np.abs(acc - avg_conf) * len_bin / n

    return ece, underconfident_ece, overconfident_ece


def plot_reliable_diagram(ax, positions, gap, bin_size, outputs, name, xname, yname):
    gap_plt = ax.bar(positions, gap, width=bin_size, edgecolor="red", color="red", alpha=0.3, label="Gap", linewidth=2,
                     zorder=2)
    output_plt = ax.bar(positions, outputs, width=bin_size, edgecolor="black", color="blue", label="Outputs", zorder=3)
    # Line plot with center line.
    ax.set_aspect('equal')
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.legend(handles=[gap_plt, output_plt])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(name, fontsize=15)
    # ax.legend(fontsize=45)
    ax.set_xlabel(xname, fontsize=15, color="black")
    ax.set_ylabel(yname, fontsize=15, color="black")


def nbest_post_process(args, all_nbest_json):
    new_all_nbest_json = {}

    for i, (id, nbest) in enumerate(tqdm(all_nbest_json.items(), ascii=True, position=0, leave=False)):
        new_all_nbest_json[id] = post_process(nbest, args)

    return new_all_nbest_json


def get_bin_acc_confs_info(all_nbest_json, exact, qids=None, bin_size=0.1):
    accs, confs = [], []
    for qid, e in exact.items():
        if qids is not None and qid not in qids:
            continue
        nbest = all_nbest_json[qid]
        accs.append(e)
        confs.append(nbest[0]['probability'])

    accs = np.array(accs)
    confs = np.array(confs)

    bin_accs, bin_confs, bin_len_bins = get_bin_info(confs, accs, bin_size=0.1)
    ece, underconfident_ece, overconfident_ece = ECE(confs, accs, bin_size)
    acc_conf = np.column_stack([bin_accs, bin_confs])
    sorted_ids = np.argsort(bin_confs)
    acc_conf = acc_conf[sorted_ids]

    outputs = acc_conf[:, 0]
    gap = acc_conf[:, 1]
    positions = np.arange(0 + bin_size / 2, 1 + bin_size / 2, bin_size)
    return outputs, positions, gap, ece, underconfident_ece, overconfident_ece


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)


class TemperatureScaling():

    def __init__(self, temp=1, maxiter=50, solver="BFGS"):
        self.temp = 0

    # Find the temperature
    def fit(self, logits, true):
        best_temp = 0.3
        best_ece = 10000
        for i in range(2000):
            temp = 0.001 * i + 0.05
            scaled_probs = self.predict(logits, temp)[:, 0]
            if np.isnan(scaled_probs).sum() > 0:
                continue

            ece, underconfident_ece, overconfident_ece = ECE(scaled_probs, true, bin_size=0.1)
            if ece < best_ece:
                best_temp = temp
                best_ece = ece
                print("new best temp: {} ".format(best_temp))
                print("new best ece: {} ".format(best_ece))

        self.temp += best_temp

    def fit_process(self, logits, true, dev_nbest_json, dev_nbest_exact, dev_nbest_f1):
        best_temp = 0.3
        best_ece = 10000
        for i in range(300):
            temp = 0.03 * i + 1.05
            scaled_probs = self.predict(logits, temp)
            if np.isnan(scaled_probs).sum() > 0:
                continue

            for j, (qid, nbest) in enumerate(dev_nbest_json.items()):
                prob = scaled_probs[j]
                for nb, p in zip(nbest, prob):
                    nb["probability"] = p

            all_nbest_json = dev_nbest_json
            exact = dev_nbest_exact
            f1 = dev_nbest_f1

            all_nbest_json = nbest_post_process(None, all_nbest_json)
            outputs, positions, gap, ece, underconfident_ece, overconfident_ece = get_bin_acc_confs_info(all_nbest_json,
                                                                                                         exact,
                                                                                                         None)  # group_qids[i])

            if ece < best_ece:
                best_temp = temp
                best_ece = ece
                print("new best temp: {} ".format(best_temp))
                print("new best ece: {} ".format(best_ece))

        self.temp += best_temp

    def predict(self, logits, temp=None):
        for i, each in enumerate(logits):
            if len(each) != 20:
                logits[i] += [0] * (20 - len(each))
        logits = np.array(logits)
        if not temp:
            return softmax(logits / self.temp)
        else:
            return softmax(logits / temp)


def compute_correlation_intensity(all_nbest_json):
    label_correlation_intensity = {}
    for qid, nbest in tqdm(all_nbest_json.items(), ascii=True, position=0, leave=False):
        label_correlation_intensity[qid] = []
        best_candidate = normalize_answer(nbest[0]['text'])

        other_candidates = [normalize_answer(each['text']) for each in nbest][1:]

        for candidate in other_candidates:
            label_correlation_intensity[qid].append(compute_f1(best_candidate, candidate))
            break

    return label_correlation_intensity


def evaluate_show_order(args, model, tokenizer, prefix=""):
    args.use_model = args.dataset_name  # "hotpot, trivia
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    all_nbest_json = read_json(
        'predictions/train_{}/dev_{}.json/nbest_predictions_.json'.format(args.dataset_name, args.dataset_name))
    predictions = read_json(
        'predictions/train_{}/dev_{}.json/predictions_.json'.format(args.dataset_name, args.dataset_name))

    results, exact, f1 = hotpot_evaluate(examples, predictions)

    # These temperature is for MSP, you can train temperature for CoMe
    trained_temp = {"trivia": [1.49, 5.55, 0.926, 3.60], "hotpot": [1.629, 6.10, 0.741, 2.61],
                    "squad": [0.531, 1.48, 0.086, 1.29]}[args.dataset_name]
    show_grouped_by_correlation = False

    used_trained_TS = True
    chose_f1 = args.choose_f1auroc
    add_process = False if args.scoring_function not in ["sl", "sp"] else args.scoring_function
    use_TS = args.ts

    copy_all_nbest_json = copy.deepcopy(all_nbest_json)
    exact_this_turn = copy.deepcopy(f1) if chose_f1 else copy.deepcopy(exact)  # choose f1 or acc

    if use_TS:
        ######### TS
        dev_nbest_json, dev_nbest_exact, dev_nbest_f1, dev_logits, dev_label = {}, {}, {}, [], []
        test_nbest_json, test_nbest_exact, test_nbest_f1, test_logits = {}, {}, {}, []
        for i, (qid, nbest) in enumerate(all_nbest_json.items()):
            logits = []

            for candidate in nbest:
                logits.append(candidate["start_logit"] + candidate["end_logit"])

            if i % 10 == 0:
                dev_nbest_json[qid] = nbest
                dev_logits.append(logits)
                dev_label.append(exact_this_turn[qid])
                dev_nbest_exact[qid] = exact_this_turn[qid]
                dev_nbest_f1[qid] = f1[qid]
            else:
                test_nbest_json[qid] = nbest
                test_nbest_exact[qid] = exact_this_turn[qid]
                test_nbest_f1[qid] = f1[qid]
                test_logits.append(logits)

        for i, each in enumerate(dev_logits):
            if len(each) != 20:
                dev_logits[i] += [0] * (20 - len(each))
        dev_logits = np.array(dev_logits)
        dev_label = np.array(dev_label)
        model = TemperatureScaling()
        if used_trained_TS:
            if chose_f1:
                if add_process:
                    model.temp = trained_temp[-1]  # 3.60  # 1.29
                else:
                    model.temp = trained_temp[-2]  # 0.926  # 0.086 #
            else:
                if add_process:
                    model.temp = trained_temp[-3]  # 1.48 #
                else:
                    model.temp = trained_temp[-4]  # 1.49  # 0.531
        # 0.531 for extract, 1.48 for extract-process
        # 0.086 for f1, 1.29 for f1-process
        else:
            if add_process:
                model.fit_process(dev_logits, dev_label, dev_nbest_json, dev_nbest_exact, dev_nbest_f1)
            else:
                model.fit(dev_logits, dev_label)

        new_test_prob = model.predict(test_logits)
        for i, (qid, nbest) in enumerate(test_nbest_json.items()):
            prob = new_test_prob[i]
            for nb, p in zip(nbest, prob):
                nb["probability"] = p

        all_nbest_json = test_nbest_json
        exact_this_turn = test_nbest_exact

    if add_process:
        all_nbest_json = nbest_post_process(args, all_nbest_json)

    label_correlation_intensity = compute_correlation_intensity(all_nbest_json)
    group_qids = [[], [], [], [], []]
    group_ids = [[], [], [], [], []]
    for i, (qid, correlation) in enumerate(label_correlation_intensity.items()):
        if correlation[0] == 0:
            group_ids[0].append(i)
            group_qids[0].append(qid)
        else:
            # group_qids[int(correlation[0] // 0.25001) + 1].append(qid)
            # group_ids[int(correlation[0] // 0.25001) + 1].append(i)
            group_qids[int(0) + 1].append(qid)
            group_ids[int(0) + 1].append(i)
    plt.style.use('ggplot')

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(22.5, 4), sharex='col', sharey='row')
    bin_size = 0.1

    for i in range(2):
        if show_grouped_by_correlation:
            outputs, positions, gap, ece, underconfident_ece, overconfident_ece = get_bin_acc_confs_info(
                all_nbest_json,
                exact_this_turn,
                group_qids[i])
            data_num = len(group_ids[i])
        else:
            outputs, positions, gap, ece, underconfident_ece, overconfident_ece = get_bin_acc_confs_info(
                all_nbest_json,
                exact_this_turn,
                None)  # group_qids[i])
            data_num = "all"

        # for p, o in zip(positions, outputs):
        # 	print(p, '\t', o)
        # return

        ts = "ts" if use_TS else "ori"
        proc = "proc" if add_process else "ori"
        metric = "f1" if chose_f1 else "ext"
        name = "{} {} Reliability Diagram {} {} {}".format(data_num, args.use_model, ts, proc, metric)
        xname = "all/over/under ECE: {}‰, {}‰, {}‰".format(int(ece * 1000),
                                                           int(overconfident_ece * 1000),
                                                           int(underconfident_ece * 1000))
        yname = ""
        ax = axs[i]
        plot_reliable_diagram(ax, positions, gap, bin_size, outputs, name, xname, yname)
    plt.show()

    all_nbest_json = copy_all_nbest_json


def main():
    parser = argparse.ArgumentParser()

    register_args(parser)
    parser.add_argument('--id_dataset', default="squad", type=str, help='')
    parser.add_argument('--ts', action="store_true", help='temperature scaling')
    parser.add_argument('--choose_f1auroc', action="store_true", help='f or accuracy')
    parser.add_argument('--scoring_function', type=str, default="sl",
                        help='scoring functions, it should be in [sl, sp, msp]')

    parser.add_argument(
        "--do_debug",
        action="store_true",
        help="Use much less data to debug",
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_false", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=100, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Eval every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--disable_tqdm", action="store_true", help="Disable tqdm bar"
    )

    args = parser.parse_args()

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config, tokenizer, model = load_untrained_model(args)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
        checkpoints = [args.model_name_or_path]

        checkpoints = ["checkpoints/squad_roberta-base"]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = None
            # model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)  # , force_download=True)
            # model.to(args.device)

            # Evaluate
            id_dataset = {"squad": 0, "hotpot": 1, "trivia": 2}[args.id_dataset]
            for dataset_name in ["squad", "hotpot", "trivia"][id_dataset:id_dataset + 1]:
                logger.info(dataset_name)
                args.dataset_name = dataset_name
                args.predict_file = "outputs//dev_{}.json".format(dataset_name)
                args.output_dir = "predictions//dev_{}.json".format(dataset_name)

                evaluate_show_order(args, model, tokenizer, prefix=global_step)

    return results


if __name__ == "__main__":
    main()
