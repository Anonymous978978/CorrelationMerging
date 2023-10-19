import argparse
import os
import random, json

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import logging
import timeit
from sklearn.metrics import roc_auc_score

from torch.utils.data import DataLoader

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
from utils import post_process

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def ErrorRateAt95Recall1(labels, scores):
    recall_point = 0.95
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    # Sort label-score tuples by the score in descending order.
    indices = np.argsort(scores)[::-1]
    sorted_labels = labels[indices]
    sorted_scores = scores[indices]
    n_match = sum(sorted_labels)
    n_thresh = recall_point * n_match
    thresh_index = np.argmax(np.cumsum(sorted_labels) >= n_thresh)
    FP = np.sum(sorted_labels[:thresh_index] == 0)
    TN = np.sum(sorted_labels[thresh_index:] == 0)
    return float(FP) / float(FP + TN)


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
    sorted_score = sorted(sorted_score, reverse=True)

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
    input_dir = "./cached/"
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
        cached_features_file = "./cached/{}_squad_{}_roberta-base_512".format(args.use_model, args.use_model)

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
    """
    Get accuracy, confidence and elements in bin information for all the bins.

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?

    Returns:
        (acc, conf, len_bins): tuple containing all the necessary info for reliability diagrams.
    """

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
    ax.set_xlabel(xname, fontsize=15, color="black")
    ax.set_ylabel(yname, fontsize=15, color="black")


def nbest_post_process(args, all_nbest_json, alpha):
    new_all_nbest_json = {}

    for i, (id, nbest) in enumerate(tqdm(all_nbest_json.items(), ascii=True, position=0, leave=False)):
        new_all_nbest_json[id] = post_process(nbest, args, alpha)

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
    """
    Compute softmax values for each sets of scores in x.

    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)


class OODTemperatureScaling():

    def __init__(self, temp=1, maxiter=50, solver="BFGS"):
        self.temp = 0

    # Find the temperature
    def fit(self, dev_logits, ood_dev_logits):
        best_temp = 0.3
        best_auroc = 0
        for i in range(1000):
            temp = 0.1 * i + 0.05
            scaled_probs = 1 - self.predict(dev_logits, temp)[:, 0]
            ood_scaled_probs = 1 - self.predict(ood_dev_logits, temp)[:, 0]

            if np.isnan(scaled_probs).sum() > 0:
                print("there is nan element")
                continue

            y = np.concatenate((np.ones_like(ood_scaled_probs), np.zeros_like(scaled_probs)))
            scores = np.concatenate((ood_scaled_probs, scaled_probs))
            auroc = 100 * roc_auc_score(y, scores)

            if auroc > best_auroc:
                best_temp = temp
                best_auroc = auroc
                print("new best temp: {} ".format(best_temp))
                print("new best ece: {} ".format(best_auroc))

        self.temp += best_temp

    def fit_process(self, dev_logits, dev_nbest_json, ood_dev_logits, ood_dev_nbest_json):
        best_temp = 0.3
        best_auroc = 0
        for i in range(60):
            temp = 0.1 * i + 1
            scaled_probs = self.predict(dev_logits, temp)
            ood_scaled_probs = self.predict(ood_dev_logits, temp)

            if np.isnan(scaled_probs).sum() > 0:
                print("there is nan element")
                continue

            for i, (qid, nbest) in enumerate(dev_nbest_json.items()):
                prob = scaled_probs[i]
                for nb, p in zip(nbest, prob):
                    nb["probability"] = p

            dev_nbest_json = dev_nbest_json

            for i, (qid, nbest) in enumerate(ood_dev_nbest_json.items()):
                prob = ood_scaled_probs[i]
                for nb, p in zip(nbest, prob):
                    nb["probability"] = p

            ood_dev_nbest_json = ood_dev_nbest_json

            dev_nbest_json = nbest_post_process(None, dev_nbest_json)
            ood_dev_nbest_json = nbest_post_process(None, ood_dev_nbest_json)

            auroc = compute_auroc(dev_nbest_json, ood_dev_nbest_json)

            if auroc > best_auroc:
                best_temp = temp
                best_auroc = auroc
                print("new best temp: {} ".format(best_temp))
                print("new best ece: {} ".format(best_auroc))

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


def compute_auroc(all_nbest_json, ood_all_nbest_json, num=None):
    id_pps = np.array([1 - each[0]["probability"] for _, each in all_nbest_json.items()])
    ood_pps = np.array([1 - each[0]["probability"] for _, each in ood_all_nbest_json.items()])

    minest = min(min(id_pps), min(ood_pps))
    maxest = max(max(id_pps), max(ood_pps))
    if maxest > 1 or minest < 0:
        id_pps = (id_pps - minest) / (maxest - minest)
        ood_pps = (ood_pps - minest) / (maxest - minest)

    if num is not None:
        random.shuffle(id_pps)
        random.shuffle(ood_pps)
        id_pps, ood_pps = id_pps[:num], ood_pps[:num]

    y = np.concatenate((np.ones_like(ood_pps), np.zeros_like(id_pps)))
    scores = np.concatenate((ood_pps, id_pps))
    auroc = 100 * roc_auc_score(y, scores)
    fpr = 100 * ErrorRateAt95Recall1(y, scores)
    return auroc, fpr


def group_nbest_by_correlation(all_nbest_json):
    label_correlation_intensity = compute_correlation_intensity(all_nbest_json)
    group_nbest = [{}, {}, {}, {}, {}]
    for i, (qid, correlation) in enumerate(label_correlation_intensity.items()):
        if correlation[0] == 0:
            group_nbest[0][qid] = all_nbest_json[qid]
        else:
            group_nbest[int(correlation[0] // 0.25001) + 1][qid] = all_nbest_json[qid]

    return group_nbest


def compute_auroc_list(all_nbest_json, ood_all_nbest_json):
    group_id = group_nbest_by_correlation(all_nbest_json)
    group_ood = group_nbest_by_correlation(ood_all_nbest_json)
    num = min([len(each) for each in group_id + group_ood])

    ans = np.zeros((5, 5))
    for i in range(20):
        auroc_list = []

        for gi in group_id:
            aurocs = []
            for go in group_ood:
                aurocs.append(compute_auroc(gi, go, num))
            auroc_list.append(aurocs)
        ans += np.array(auroc_list)
    ans /= 20
    return ans


def plot_id_distribution_by_group(all_nbest_json):
    label_correlation_intensity = compute_correlation_intensity(all_nbest_json)
    group_qids = [[], [], [], [], []]
    group_ids = [[], [], [], [], []]
    for i, (qid, correlation) in enumerate(label_correlation_intensity.items()):
        if correlation[0] == 0:
            group_ids[0].append(i)
            group_qids[0].append(qid)
        else:
            group_qids[int(correlation[0] // 0.25001) + 1].append(qid)
            group_ids[int(correlation[0] // 0.25001) + 1].append(i)
            group_qids[1].append(qid)
            group_ids[1].append(i)

    probs_ind, probs_corr, probs_corr_all = [], [], []
    for qid, each in all_nbest_json.items():
        if qid in group_qids[0]:
            probs_ind.append(each[0]["probability"])
        elif qid in group_qids[-1]:
            probs_corr.append(each[0]["probability"])
        if qid in group_qids[1]:
            probs_corr_all.append(each[0]["probability"])


def evaluate_show_order(args, model, tokenizer, prefix=""):
    all_nbest_json = read_json(
        'predictions/train_{}/dev_{}.json/nbest_predictions_.json'.format(args.dataset_name, args.dataset_name))

    ood_all_nbest_json = read_json(
        'predictions/train_{}/dev_{}.json/nbest_predictions_.json'.format(args.dataset_name, args.ood_dataset))

    best_alpha = 1
    all_nbest_json = nbest_post_process(args, all_nbest_json, best_alpha)
    ood_all_nbest_json = nbest_post_process(args, ood_all_nbest_json, best_alpha)

    auroc, fpr = compute_auroc(all_nbest_json, ood_all_nbest_json)

    print("auroc  fpr: {}\t{} \n ".format(auroc, fpr))


def main():
    parser = argparse.ArgumentParser()

    register_args(parser)
    parser.add_argument('--scoring_function', type=str, default="sl",
                        help='scoring functions, it should be in [sl, sp, msp]')
    parser.add_argument('--id_dataset', default="squad", type=str, help='')
    parser.add_argument('--ood_dataset', default=None, type=str, help='')
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

        id_dataset_id = {"squad": 0, "hotpot": 1, "trivia": 2}[args.id_dataset]
        for dataset_name in ["squad", "hotpot", "trivia"][id_dataset_id:id_dataset_id + 1]:
            print("ID DATASET: ", dataset_name)
            print("Scoring_function: {}".format(args.scoring_function))
            checkpoints = ["checkpoints/{}_roberta-base".format(dataset_name)]
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                # Reload the model
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                model = None
                # model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)  # , force_download=True)
                # model.to(args.device)

                # Evaluate
                if args.ood_dataset is None:
                    start = 0
                    end = 8
                else:
                    ood_dataset_id = {"squad": 0, "trivia": 1, "hotpot": 2, "searchqa": 3, "textbookqa": 4,
                                      "newsqa": 5, "naturalqa": 6, "bioasq": 7}[args.ood_dataset]
                    start = ood_dataset_id
                    end = start + 1
                for ood_dataset in ["squad", "trivia", "hotpot", "searchqa", "textbookqa", "newsqa", "naturalqa",
                                    "bioasq"][start:end]:
                    if ood_dataset == dataset_name:
                        continue
                    logger.info(ood_dataset)
                    args.dataset_name = dataset_name
                    args.ood_dataset = ood_dataset
                    args.predict_file = "outputs//dev_{}.json".format(dataset_name)
                    args.output_dir = "predictions//dev_{}.json".format(dataset_name)

                    evaluate_show_order(args, model, tokenizer, prefix=global_step)
            break


if __name__ == "__main__":
    main()
