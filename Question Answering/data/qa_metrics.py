import torch
import collections
import json
import math
import re
import string
from tqdm import tqdm
import numpy as np
from transformers.models.bert.tokenization_bert import BasicTokenizer

from transformers.utils import logging
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

def auc_score(x, y):
    fpr, tpr, _ = roc_curve(y, x)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def f1auc_score(score, f1):
    score = np.ravel(score)
    f1 = np.ravel(f1)
    sorted_idx = np.argsort(-score)
    score = score[sorted_idx]
    f1 = f1[sorted_idx]
    num_test = f1.size
    segment = min(1000, score.size - 1)
    T = np.arange(segment) + 1
    T = T/segment # [0.001, 0.002, 0.003.....1.000]
    results = np.array([np.mean(f1[:int(num_test * t)])  for t in T])
    # print(results)
    return np.mean(results)

logger = logging.get_logger(__name__)


def normalize_answer(s):
	"""Lower text and remove punctuation, articles and extra whitespace."""
	
	def remove_articles(text):
		regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
		return re.sub(regex, " ", text)
	
	def white_space_fix(text):
		return " ".join(text.split())
	
	def remove_punc(text):
		exclude = set(string.punctuation)
		return "".join(ch for ch in text if ch not in exclude)
	
	def lower(text):
		return text.lower()
	
	return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
	if not s:
		return []
	return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
	return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
	gold_toks = get_tokens(a_gold)
	pred_toks = get_tokens(a_pred)
	common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
	num_same = sum(common.values())
	if len(gold_toks) == 0 or len(pred_toks) == 0:
		# If either is no-answer, then F1 is 1 if they agree, 0 otherwise
		return int(gold_toks == pred_toks)
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(pred_toks)
	recall = 1.0 * num_same / len(gold_toks)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1


def get_raw_scores(examples, preds):
	"""
	Computes the exact and f1 scores from the examples and the model predictions
	"""
	exact_scores = {}
	f1_scores = {}
	
	for example in examples:
		qas_id = example.qas_id
		gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]
		
		if not gold_answers:
			# For unanswerable questions, only correct answer is empty string
			gold_answers = [""]
		
		if qas_id not in preds:
			print("Missing prediction for %s" % qas_id)
			continue
		
		prediction = preds[qas_id]
		
		# print(gold_answers, prediction)
		exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
		f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)
	
	return exact_scores, f1_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
	new_scores = {}
	for qid, s in scores.items():
		pred_na = na_probs[qid] > na_prob_thresh
		if pred_na:
			new_scores[qid] = float(not qid_to_has_ans[qid])
		else:
			new_scores[qid] = s
	return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
	if not qid_list:
		total = len(exact_scores)
		return collections.OrderedDict(
			[
				("exact", 100.0 * sum(exact_scores.values()) / total),
				("f1", 100.0 * sum(f1_scores.values()) / total),
				("total", total),
			]
		)
	else:
		total = len(qid_list)
		return collections.OrderedDict(
			[
				("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
				("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
				("total", total),
			]
		)


def merge_eval(main_eval, new_eval, prefix):
	for k in new_eval:
		main_eval["%s_%s" % (prefix, k)] = new_eval[k]


def find_best_thresh_v2(preds, scores, na_probs, qid_to_has_ans):
	num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
	cur_score = num_no_ans
	best_score = cur_score
	best_thresh = 0.0
	qid_list = sorted(na_probs, key=lambda k: na_probs[k])
	for i, qid in enumerate(qid_list):
		if qid not in scores:
			continue
		if qid_to_has_ans[qid]:
			diff = scores[qid]
		else:
			if preds[qid]:
				diff = -1
			else:
				diff = 0
		cur_score += diff
		if cur_score > best_score:
			best_score = cur_score
			best_thresh = na_probs[qid]
	
	has_ans_score, has_ans_cnt = 0, 0
	for qid in qid_list:
		if not qid_to_has_ans[qid]:
			continue
		has_ans_cnt += 1
		
		if qid not in scores:
			continue
		has_ans_score += scores[qid]
	
	return 100.0 * best_score / len(scores), best_thresh, 1.0 * has_ans_score / has_ans_cnt


def find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
	best_exact, exact_thresh, has_ans_exact = find_best_thresh_v2(preds, exact_raw, na_probs, qid_to_has_ans)
	best_f1, f1_thresh, has_ans_f1 = find_best_thresh_v2(preds, f1_raw, na_probs, qid_to_has_ans)
	main_eval["best_exact"] = best_exact
	main_eval["best_exact_thresh"] = exact_thresh
	main_eval["best_f1"] = best_f1
	main_eval["best_f1_thresh"] = f1_thresh
	main_eval["has_ans_exact"] = has_ans_exact
	main_eval["has_ans_f1"] = has_ans_f1


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
	num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
	cur_score = num_no_ans
	best_score = cur_score
	best_thresh = 0.0
	qid_list = sorted(na_probs, key=lambda k: na_probs[k])
	for _, qid in enumerate(qid_list):
		if qid not in scores:
			continue
		if qid_to_has_ans[qid]:
			diff = scores[qid]
		else:
			if preds[qid]:
				diff = -1
			else:
				diff = 0
		cur_score += diff
		if cur_score > best_score:
			best_score = cur_score
			best_thresh = na_probs[qid]
	return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
	best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
	best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
	
	main_eval["best_exact"] = best_exact
	main_eval["best_exact_thresh"] = exact_thresh
	main_eval["best_f1"] = best_f1
	main_eval["best_f1_thresh"] = f1_thresh


def squad_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
	qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
	has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
	no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]
	
	if no_answer_probs is None:
		no_answer_probs = {k: 0.0 for k in preds}
	
	exact, f1 = get_raw_scores(examples, preds)
	
	exact_threshold = apply_no_ans_threshold(
		exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
	)
	f1_threshold = apply_no_ans_threshold(f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold)
	
	evaluation = make_eval_dict(exact_threshold, f1_threshold)
	
	if has_answer_qids:
		has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=has_answer_qids)
		merge_eval(evaluation, has_ans_eval, "HasAns")
	
	if no_answer_qids:
		no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=no_answer_qids)
		merge_eval(evaluation, no_ans_eval, "NoAns")
	
	if no_answer_probs:
		find_all_best_thresh(evaluation, preds, exact, f1, no_answer_probs, qas_id_to_has_answer)
	
	return evaluation


def aggeragate_conf(start_confidence, end_confidence):
	max_start_confidence = np.max(start_confidence, axis=1)
	max_end_confidence = np.max(end_confidence, axis=1)
	
	aggeragate_confidence = (max_start_confidence + max_end_confidence) / 2
	
	return aggeragate_confidence


def aggeragate_conf_from_nbest(all_nbest_json):
	all_nbest_aggeragate_confidence = []
	for k, nbest in all_nbest_json.items():
		aggeragate_confidence = []
		for prob in nbest:
			aggeragate_confidence.append(prob["probability"])
		all_nbest_aggeragate_confidence.append(aggeragate_confidence)
	best_aggeragate_confidence = [each[0] for each in all_nbest_aggeragate_confidence]
	best_aggeragate_confidence = np.array(best_aggeragate_confidence)
	all_nbest_aggeragate_confidence = np.array(all_nbest_aggeragate_confidence)
	return best_aggeragate_confidence, all_nbest_aggeragate_confidence


def compute_acc(best_aggeragate_confidence, threshold, exact):
	return np.sum((best_aggeragate_confidence > threshold) == exact) / len(exact)


def calib_evaluate(all_nbest_json, exact, f1):
	# logit
	best_aggeragate_logits = [each[0]["start_logit"]+each[0]["end_logit"] for _, each in all_nbest_json.items()]
	best_aggeragate_logits = (np.array(best_aggeragate_logits) - min(best_aggeragate_logits)) / (max(best_aggeragate_logits)-min(best_aggeragate_logits))
	f1 = np.array([v for k, v in f1.items()])
	exact = np.array([v for k, v in exact.items()])
	FAUC = f1auc_score(best_aggeragate_logits, f1)
	AUC = auc_score(best_aggeragate_logits, exact)
	threshold = 0.5
	ACC = compute_acc(best_aggeragate_logits, threshold, exact)
	return FAUC, AUC, ACC


	best_aggeragate_confidence, all_nbest_aggeragate_confidence = aggeragate_conf_from_nbest(all_nbest_json)

	f1 = np.array([v for k, v in f1.items()])
	exact = np.array([v for k, v in exact.items()])
	FAUC = f1auc_score(best_aggeragate_confidence, f1)
	AUC = auc_score(best_aggeragate_confidence, exact)
	threshold = 0.5
	ACC = compute_acc(best_aggeragate_confidence, threshold, exact)
	return FAUC, AUC, ACC


def calib_seperate_point_evaluate(all_results, exact, f1):
	start_confidence, end_confidence = [], []
	for result in all_results:
		start_confidence.append(torch.softmax(result.start_logits, dim=0).detach().cpu().numpy())
		end_confidence.append(torch.softmax(result.end_logits, dim=0).detach().cpu().numpy())
	start_confidence = np.array(start_confidence)
	end_confidence = np.array(end_confidence)
	aggeragate_confidence = aggeragate_conf(start_confidence, end_confidence)
	
	f1 = np.array([v for k, v in f1.items()])
	exact = np.array([v for k, v in exact.items()])
	FAUC = f1auc_score(aggeragate_confidence, f1)
	AUC = auc_score(aggeragate_confidence, exact)
	threshold = 0.5
	ACC = compute_acc(aggeragate_confidence, threshold, exact)
	return FAUC, AUC, ACC


def hotpot_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
	qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
	has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
	
	exact, f1 = get_raw_scores(examples, preds)
	
	evaluation = make_eval_dict(exact, f1)
	
	return evaluation, exact, f1


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
	def _strip_spaces(text):
		ns_chars = []
		ns_to_s_map = collections.OrderedDict()
		for (i, c) in enumerate(text):
			if c == " ":
				continue
			ns_to_s_map[len(ns_chars)] = i
			ns_chars.append(c)
		ns_text = "".join(ns_chars)
		return (ns_text, ns_to_s_map)

	tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
	
	tok_text = " ".join(tokenizer.tokenize(orig_text))
	
	start_position = tok_text.find(pred_text)
	if start_position == -1:
		if verbose_logging:
			logger.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
		return orig_text
	end_position = start_position + len(pred_text) - 1
	
	(orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
	(tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)
	
	if len(orig_ns_text) != len(tok_ns_text):
		if verbose_logging:
			logger.info("Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text)
		return orig_text

	tok_s_to_ns_map = {}
	for (i, tok_index) in tok_ns_to_s_map.items():
		tok_s_to_ns_map[tok_index] = i
	
	orig_start_position = None
	if start_position in tok_s_to_ns_map:
		ns_start_position = tok_s_to_ns_map[start_position]
		if ns_start_position in orig_ns_to_s_map:
			orig_start_position = orig_ns_to_s_map[ns_start_position]
	
	if orig_start_position is None:
		if verbose_logging:
			logger.info("Couldn't map start position")
		return orig_text
	
	orig_end_position = None
	if end_position in tok_s_to_ns_map:
		ns_end_position = tok_s_to_ns_map[end_position]
		if ns_end_position in orig_ns_to_s_map:
			orig_end_position = orig_ns_to_s_map[ns_end_position]
	
	if orig_end_position is None:
		if verbose_logging:
			logger.info("Couldn't map end position")
		return orig_text
	
	output_text = orig_text[orig_start_position: (orig_end_position + 1)]
	return output_text


def _get_best_indexes(logits, n_best_size):
	best_indexes = torch.topk(logits, k=n_best_size)[1].tolist()
	return best_indexes


def _compute_softmax(scores):
	"""Compute softmax probability over raw logits."""
	if not scores:
		return []
	
	max_score = None
	for score in scores:
		if max_score is None or score > max_score:
			max_score = score
	
	exp_scores = []
	total_sum = 0.0
	for score in scores:
		x = math.exp(score - max_score)
		exp_scores.append(x)
		total_sum += x
	
	probs = []
	for score in exp_scores:
		probs.append(score / total_sum)
	return probs


def compute_predictions_logits(
		all_examples,
		all_features,
		all_results,
		n_best_size,
		max_answer_length,
		do_lower_case,
		output_prediction_file,
		output_nbest_file,
		output_null_log_odds_file,
		verbose_logging,
		version_2_with_negative,
		null_score_diff_threshold,
		tokenizer,
		dataset='hpqa',
		output_probs=False
):
	example_index_to_features = collections.defaultdict(list)
	for feature in all_features:
		example_index_to_features[feature.example_index].append(feature)
	
	unique_id_to_result = {}
	for result in all_results:
		unique_id_to_result[result.unique_id] = result
	
	start_logits_list, end_logits_list = None, None
	for result in all_results:
		if start_logits_list is None:
			start_logits_list, end_logits_list = result.start_logits.unsqueeze(0), result.end_logits.unsqueeze(0)
		else:
			start_logits_list = torch.cat([start_logits_list, result.start_logits.unsqueeze(0)], dim=0)
			end_logits_list = torch.cat([end_logits_list, result.end_logits.unsqueeze(0)], dim=0)
	start_logits_n_best = torch.topk(start_logits_list, k=n_best_size, dim=-1).indices.tolist()
	end_logits_n_best = torch.topk(end_logits_list, k=n_best_size, dim=-1).indices.tolist()
	
	_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
		"PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
	)
	
	all_predictions = collections.OrderedDict()
	all_nbest_json = collections.OrderedDict()
	scores_diff_json = collections.OrderedDict()
	
	for (example_index, example) in enumerate(tqdm(all_examples, ascii=True, position=0, leave=False)):
		features = example_index_to_features[example_index]
		
		prelim_predictions = []
		# keep track of the minimum score of null start+end of position 0
		score_null = 1000000  # large and positive
		min_null_feature_index = 0  # the paragraph slice with min null score
		null_start_logit = 0  # the start logit at the slice with min null score
		null_end_logit = 0  # the end logit at the slice with min null score
		for (feature_index, feature) in enumerate(features):
			result = unique_id_to_result[feature.unique_id]
			"""start_indexes = _get_best_indexes(result.start_logits, n_best_size)
			end_indexes = _get_best_indexes(result.end_logits, n_best_size)"""
			start_indexes = start_logits_n_best[example_index]
			end_indexes = end_logits_n_best[example_index]
			# if we could have irrelevant answers, get the min score of irrelevant
			if version_2_with_negative:
				feature_null_score = result.start_logits[0] + result.end_logits[0]
				if feature_null_score < score_null:
					score_null = feature_null_score
					min_null_feature_index = feature_index
					null_start_logit = result.start_logits[0]
					null_end_logit = result.end_logits[0]
			for start_index in start_indexes:
				for end_index in end_indexes:
					if start_index >= len(feature.tokens):
						continue
					if end_index >= len(feature.tokens):
						continue
					if start_index not in feature.token_to_orig_map:
						continue
					if end_index not in feature.token_to_orig_map:
						continue
					if not feature.token_is_max_context.get(start_index, False):
						continue
					if end_index < start_index:
						continue
					length = end_index - start_index + 1
					if length > max_answer_length:
						continue
					prelim_predictions.append(
						_PrelimPrediction(
							feature_index=feature_index,
							start_index=start_index,
							end_index=end_index,
							start_logit=result.start_logits[start_index],
							end_logit=result.end_logits[end_index],
						)
					)
		if version_2_with_negative:
			prelim_predictions.append(
				_PrelimPrediction(
					feature_index=min_null_feature_index,
					start_index=0,
					end_index=0,
					start_logit=null_start_logit,
					end_logit=null_end_logit,
				)
			)
		prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
		
		_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
			"NbestPrediction", ["text", "start_logit", "end_logit", "start_index", "end_index"]
		)
		
		seen_predictions = {}
		nbest = []
		
		prefix_tokens = ['yes', 'no', 'unk', tokenizer.sep_token] if dataset == 'hpqa' else []
		ex_doc_tokens = prefix_tokens + example.doc_tokens
		for pred in prelim_predictions:
			if len(nbest) >= n_best_size:
				break
			feature = features[pred.feature_index]
			if pred.start_index > 0:  # this is a non-null prediction
				tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
				orig_doc_start = feature.token_to_orig_map[pred.start_index]
				orig_doc_end = feature.token_to_orig_map[pred.end_index]
				orig_tokens = ex_doc_tokens[orig_doc_start: (orig_doc_end + 1)]
				
				tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

				tok_text = tok_text.strip()
				tok_text = " ".join(tok_text.split())
				orig_text = " ".join(orig_tokens)
				
				final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
				if final_text in seen_predictions:
					continue
				
				seen_predictions[final_text] = True
			else:
				final_text = ""
				seen_predictions[final_text] = True
			
			nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit,
			                              start_index=pred.start_index, end_index=pred.end_index))
		# if we didn't include the empty option in the n-best, include it
		if version_2_with_negative:
			if "" not in seen_predictions:
				nbest.append(
					_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit,
					                 start_index=0,
					                 end_index=0))

			if len(nbest) == 1:
				nbest.insert(0,
				             _NbestPrediction(text="empty", start_logit=torch.tensor(0.0), end_logit=torch.tensor(0.0),
			                              start_index=0,
			                              end_index=0))
		if not nbest:
			nbest.append(_NbestPrediction(text="empty", start_logit=torch.tensor(0.0), end_logit=torch.tensor(0.0),
			                              start_index=0,
			                              end_index=0))
		
		assert len(nbest) >= 1, "No valid predictions"
		
		total_scores = []
		best_non_null_entry = None
		for entry in nbest:
			total_scores.append(entry.start_logit + entry.end_logit)
			if not best_non_null_entry:
				if entry.text:
					best_non_null_entry = entry
		
		probs = _compute_softmax(total_scores)
		
		nbest_json = []
		for (i, entry) in enumerate(nbest):
			output = collections.OrderedDict()
			output["text"] = entry.text
			output["probability"] = probs[i]
			output["start_logit"] = entry.start_logit if type(entry.start_logit) is float else entry.start_logit.item()
			output["end_logit"] = entry.end_logit if type(entry.end_logit) is float else entry.end_logit.item()
			output["start_index"] = entry.start_index
			output["end_index"] = entry.end_index
			nbest_json.append(output)
		
		assert len(nbest_json) >= 1, "No valid predictions"
		
		if not version_2_with_negative:
			all_predictions[example.qas_id] = nbest_json[0]["text"]
		else:
			# predict "" iff the null score - the score of best non-null > threshold
			score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
			scores_diff_json[example.qas_id] = score_diff
			if score_diff > null_score_diff_threshold:
				all_predictions[example.qas_id] = ""
			else:
				all_predictions[example.qas_id] = best_non_null_entry.text
		all_nbest_json[example.qas_id] = nbest_json
	
	if output_prediction_file:
		with open(output_prediction_file, "w") as writer:
			writer.write(json.dumps(all_predictions, indent=4) + "\n")
	
	if output_nbest_file:
		with open(output_nbest_file, "w") as writer:
			writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
	
	if output_null_log_odds_file and version_2_with_negative:
		with open(output_null_log_odds_file, "w") as writer:
			writer.write(json.dumps(scores_diff_json, indent=4) + "\n")
			
	if output_probs:
		return all_predictions, all_nbest_json
	return all_predictions
