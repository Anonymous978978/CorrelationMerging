import argparse
import glob
import json
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import timeit
from os.path import join
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

from data.qa_metrics import (compute_predictions_logits, hotpot_evaluate, calib_evaluate, aggeragate_conf_from_nbest,
                             f1auc_score, auc_score, normalize_answer, compute_f1)
from itertools import chain


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def get_semnatic_similarity(question_key_words, evidence_key_words, model, tokenizer):
	top5_question_keywords = question_key_words[:5]
	top5_question_keywords = [each[0] for each in top5_question_keywords]
	evidence_key_words = [each[0] for each in evidence_key_words]
	
	question_input_ids = tokenizer(top5_question_keywords).data['input_ids']
	evidence_input_ids = tokenizer(evidence_key_words).data['input_ids']
	
	encoder = model.roberta
	question_embeddings = [encoder(torch.tensor([input_ids]).cuda())[0][0] for input_ids in question_input_ids]
	question_embeddings = torch.tensor([each[1:-1].mean(dim=0).tolist() for each in question_embeddings])
	evidence_embeddings = [encoder(torch.tensor([input_ids]).cuda())[0][0] for input_ids in evidence_input_ids]
	evidence_embeddings = torch.tensor([each[1:-1].mean(dim=0).tolist() for each in evidence_embeddings])
	
	dot_product = torch.matmul(question_embeddings, evidence_embeddings.T)
	norm_length = question_embeddings.norm(dim=1).unsqueeze(1) * evidence_embeddings.norm(dim=1).unsqueeze(0)
	acceptance_matrix = (dot_product / norm_length).max(dim=-1)[0]
	
	return acceptance_matrix


def extract_keyword(kw_model, nlp_text):
	try:
		keywords = kw_model.extract_keywords(docs=nlp_text.text, vectorizer=KeyphraseCountVectorizer())
	except:
		return []
	# rake = RAKE.Rake(RAKE.SmartStopList())
	# keywords = rake.run(nlp_text.text)
	print('\n', nlp_text.text)
	print(keywords, '\n')
	
	return keywords


def compute_confidence(args, kw_model, question, index_to_sentence, sentences, nbest, tokenizer, e, f, question_tokens,
                       model, example):
	start_position, end_position = (nbest[0]["start_index"], nbest[0]["end_index"])
	start_sentence_id, end_sentence_id = index_to_sentence[start_position], index_to_sentence[end_position]
	
	# if start_sentence_id != end_sentence_id:
	# 	return [0, 0, 0, 0, 0]
	
	evidence_sentence = sentences[start_sentence_id - 1]  # the first sentence is question
	# question_key_words = extract_keyword(kw_model, question)
	# evidence_key_words = extract_keyword(kw_model, evidence_sentence)
	
	if f == 1:
		gold_label = 2
	elif f < 1 and f > 0:
		gold_label = 1
	else:
		gold_label = 0
	args.nli_data.append({"pairID": example.qas_id,
	                      "sentence1": "answer : " + nbest[0]['text'] + " | " + " question : " + question.text,
	                      "sentence2": evidence_sentence.text,
	                      "gold_label": gold_label,
	                      "confidence": nbest[0]["probability"],
	                      "f1_score": f})
	
	# if nbest[0]["probability"] > 0.75:
	# 	print("confidence: {}   e: {}, f:{}".format(nbest[0]["probability"], e, f))
	# 	print("prediction: {}   answer: {}".format(nbest[0]['text'], example.answers[0]['text']))
	# 	print("question: {}".format(question.text))
	# 	print("evidence: {}".format(evidence_sentence.text), '\n')
	return None
	
	if len(question_key_words) * len(evidence_key_words) == 0:
		print(question.text)
		return [0, 0, 0, 0, 0]
	
	acceptance_matrix = get_semnatic_similarity(question_key_words, evidence_key_words, model, tokenizer)
	
	# if f == 0 or f == 1:
	# print('\n', e, f, question.text)
	print(acceptance_matrix)
	# print(evidence_key_words)
	# print([(token, int(attr.item() * 100)) for token, attr in zip(question_tokens, attribution)])
	# print("SUM:", sum(attribution), "Mean:", attribution.mean())
	return acceptance_matrix


def shortcut_validation(args, all_nbest_json, examples, features, exact, f1, model):
	tokenizer = args.tokenizer
	
	# nlp = args.nlp
	
	kw_model = None #KeyBERT(model='all-mpnet-base-v2')
	
	saved_nlp = torch.load("stanza_data/{}_saved_nlp".format(args.dataset_name))
	# saved_nlp = {}
	args.nli_data = []
	acceptance_matrix_list = []
	for step, (example, feature) in enumerate(tqdm(zip(examples, features), position=0, ascii=True, ncols=100)):
		# if step % 5 != 0:
		#  	continue
		qid = example.qas_id
		# attribution = torch.load(args.interp_dict[qid])["attribution"]
		
		question_tokens = feature.tokens[1:len(feature.tokens) - len(feature.token_is_max_context) - 4 + 1]
		nbest, e, f = all_nbest_json[qid], exact[qid], f1[qid]
		tokens, input_ids = feature.tokens, feature.input_ids
		predict_answer_span = (nbest[0]["start_index"], nbest[0]["end_index"])
		predict_answer = nbest[0]["text"]
		context = example.context_text
		question = example.question_text
		# if predict_answer!=tokenizer.decode(
		# 	input_ids[predict_answer_span[0]: predict_answer_span[1]+1]).strip():
		# 	print('\n', predict_answer)
		# 	print(tokenizer.decode(input_ids[predict_answer_span[0]: predict_answer_span[1]+1]).strip())
		
		all_ids = []
		question_ids = tokenizer(question)
		all_ids = all_ids + question_ids.data["input_ids"] + [2]
		index_to_sentence = [0] * len(all_ids)
		
		# sentences = nlp(context).sentences
		# question = nlp(question).sentences[-1]
		# saved_nlp[qid] = {"sentence": sentences, "question": question}
		# assert len(question) == 1
		# continue
		
		sentences = saved_nlp[qid]["sentence"]
		question = saved_nlp[qid]["question"]
		
		for i, sentence in enumerate(sentences):
			sentence.sentence_ids = tokenizer(" " + sentence.text)
			all_ids = all_ids + sentence.sentence_ids["input_ids"][1:-1]
			index_to_sentence += [i + 1] * len(sentence.sentence_ids["input_ids"][1:-1])
		
		# compare_length = len(all_ids) if len(all_ids) < len(input_ids) else len(input_ids)-1
		# if int(sum(np.array(all_ids[:compare_length]) != np.array(input_ids[:compare_length]))) != 0:
		# 	difference1 = np.array(all_ids[:compare_length])[np.array(all_ids[:compare_length])!=np.array(input_ids[:compare_length])]
		# 	difference2 = np.array(input_ids[:compare_length])[np.array(all_ids[:compare_length])!=np.array(input_ids[:compare_length])]
		# 	print('\n',tokenizer.decode(difference1))
		# 	print(tokenizer.decode(difference2))
		
		acceptance_matrix = compute_confidence(args, kw_model, question, index_to_sentence, sentences, nbest, tokenizer,
		                                       e, f,
		                                       question_tokens, model, example)
		acceptance_matrix_list.append({qid: acceptance_matrix})
	
	# torch.save(saved_nlp, "stanza_data/{}_saved_nlp".format(args.dataset_name))
	
	filename = "qa_to_nli_data/{}_nli.jsonl".format(args.dataset_name)
	with open(filename, 'w', encoding="UTF-8") as f:
		for e in args.nli_data:
			json_str = json.dumps(e)
			f.write(json_str)
			f.write('\n')
	
	return acceptance_matrix_list


def build_file_dict(args):
	fnames = os.listdir(join('interpretations', "question_lime", args.dataset_name))
	if args.dataset == "news":
		qa_ids = [x for x in fnames]
	else:
		qa_ids = [x.split('-', 1)[1].split('.')[0] for x in fnames]
	fullnames = [join('interpretations', "question_lime", args.dataset_name, x) for x in fnames]
	return dict(zip(qa_ids, fullnames))


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


def post_process_by_embedding(args, nbest, match_score):
	best = nbest[0]
	best_answer, best_probability = best["text"], best["probability"]
	# print("probability before: {}".format(best_probability))
	
	best_answer = normalize_answer(best_answer)
	best_answer_words = best_answer.split()
	
	for i in range(1, len(nbest)):
		candidate = nbest[i]
		candidate_answer, candidate_probability = candidate["text"], candidate["probability"]
		candidate_answer = normalize_answer(candidate_answer)
		candidate_words = candidate_answer.split()
		
		relevance = match_score[0][i] / len(best_answer_words)
		
		if relevance > 0.99:
			best_probability += candidate_probability
	
	best["probability"] = best_probability
	nbest[0] = best
	
	# print("probability after: {}".format(best_probability))
	
	return nbest


def post_process(nbest,args, alpha=0):
	remove_char_list = ["(", ")", ",", ".", "#", "'", '"', "@", "|", ]  # "had bi  ".strip()
	best = nbest[0]
	best_answer, best_probability = best["text"], best["probability"]
	# print("probability before: {}".format(best_probability))
	
	best_answer = normalize_answer(best_answer)
	best_answer_words = best_answer.split()

	logits =  [nbest[0]['start_logit']+ nbest[0]['end_logit']]
	superclass_logits = 0

	superclass_probability = 0
	bro_num = 0

	ids = [k for k in range(1, 20)]
	random.shuffle(ids)
	ids = ids[:20]
	for i in range(1, len(nbest)):
		candidate = nbest[i]
		candidate_answer, candidate_probability = candidate["text"], candidate["probability"]
		candidate_answer = normalize_answer(candidate_answer)
		candidate_words = candidate_answer.split()
		# if len(candidate_words) > len(best_answer_words) * 2 + 2:
		# 	continue
		# if len(best_answer_words) > len(candidate_words) * 2 + 2:
		# 	continue
		logit_candidate = candidate["start_logit"] + candidate["end_logit"]
		logits.append(logit_candidate)

		relevance = compute_f1(candidate_answer, best_answer)
		# randomly construct group
		random_group = not True
		if random_group:
			# print("this is randomSuper")
			if i in ids:
				superclass_probability += candidate_probability
				superclass_logits += logit_candidate
				bro_num += 1
		else:
			if len(best_answer_words) <= 2:
				if relevance >= 0.3:
					superclass_probability += candidate_probability
					superclass_logits += logit_candidate
					bro_num += 1
			elif len(best_answer_words) >= 6:
				if relevance >= 0.9:
					superclass_probability += candidate_probability
					superclass_logits += logit_candidate
					bro_num += 1
			else:
				if relevance >= 0.6:
					superclass_probability += candidate_probability
					superclass_logits += logit_candidate
					bro_num += 1


	# logits[0] += alpha * superclass_logits /max(bro_num, 1.)
	# best["start_logit"] += alpha * superclass_logits /max(bro_num, 1.)
	if args.scoring_function=="sp":
		best["probability"] += superclass_probability
	elif args.scoring_function=="sl":
		logits[0] += alpha * superclass_logits / max(bro_num, 1.)
		best["probability"] = logits[0]
	else:
		pass

	nbest[0] = best
	return nbest



def nbest_post_process(args, all_nbest_json):
	new_all_nbest_json = {}
	
	all_match_score = None
	if not os.path.isfile("{}_all_match_scores".format(args.dataset_name)):
		all_match_score = []
		for id, nbest in tqdm(all_nbest_json.items(), ascii=True, position=0, leave=False):
			encoder, tokenizer = args.model.roberta, args.tokenizer
			nbest_texts = [each['text'] for each in nbest]
			nbest_input_ids = tokenizer(nbest_texts).data['input_ids']
			nbest_embeddings = [encoder(torch.tensor([input_ids]).cuda())[0][0] for input_ids in nbest_input_ids]
			nbest_embeddings = torch.tensor([each[1:-1].mean(dim=0).tolist() for each in nbest_embeddings])
			best_embedding = nbest_embeddings[0:1]
			dot_product = torch.matmul(best_embedding, nbest_embeddings.T)
			norm_length = best_embedding.norm(dim=1).unsqueeze(1) * nbest_embeddings.norm(dim=1).unsqueeze(0)
			match_score = (dot_product / norm_length)
			all_match_score.append({id: match_score})
		torch.save(all_match_score, "{}_all_match_scores".format(args.dataset_name))
	else:
		all_match_score = torch.load("{}_all_match_scores".format(args.dataset_name))
	
	for i, (id, nbest) in enumerate(tqdm(all_nbest_json.items(), ascii=True, position=0, leave=False)):
		if all_match_score is not None:
			assert id in all_match_score[i].keys()
			match_score = all_match_score[i][id]
		else:
			match_score = None
		new_all_nbest_json[id] = post_process_by_embedding(args, nbest, match_score)
	
	return new_all_nbest_json


def evaluate_show_order(args, model, tokenizer, prefix=""):
	dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
	
	all_nbest_json = read_json(
		'baselines_Kamath_LIMECAL/calib_exp/data/predictions/dev_{}.json/nbest_predictions_.json'.format(
			args.predict_file.split("_")[1].split(".")[0]))
	
	predictions = read_json(
		'baselines_Kamath_LIMECAL/calib_exp/data/predictions/dev_{}.json/predictions_.json'.format(
			args.predict_file.split("_")[1].split(".")[0]))
	
	args.tokenizer = tokenizer
	# Compute the F1 and exact scores.
	results, exact, f1 = hotpot_evaluate(examples, predictions)
	
	args.model = model
	f1 = np.array([v for k, v in f1.items()])
	exact = np.array([v for k, v in exact.items()])
	best_aggeragate_confidence, _ = aggeragate_conf_from_nbest(all_nbest_json)
	
	score = np.ravel(best_aggeragate_confidence)
	f1 = np.ravel(f1)
	sorted_idx = np.argsort(-score)
	sorted_score = score[sorted_idx]
	sorted_f1 = f1[sorted_idx]
	sorted_exact = exact[sorted_idx]
	
	prediction_list = [(n, p) for n, p in predictions.items()]
	sorted_examples = np.array(examples)[sorted_idx].tolist()
	sorted_predictions = np.array(prediction_list)[sorted_idx].tolist()
	
	def attribution_validation():
		for i, (id, prediction) in enumerate(sorted_predictions):
			if i <= 1300:
				continue
			if i == 5500:
				break
			
			if "news" in args.dataset_name:
				interp_id = id.split('/')[-1].split(".")[0]
			else:
				interp_id = id
			if interp_id not in args.interp_dict.keys():
				continue
			
			e = sorted_exact[i]
			f = sorted_f1[i]
			if f != 0 and f != 1:
				continue
			interp = torch.load(args.interp_dict[interp_id])
			example = sorted_examples[i]
			nbest = all_nbest_json[id]
			
			question_tokens = interp['feature'].tokens[1:len(interp['attribution']) + 1]
			context = interp['example'].context_text
			print(nbest[0])
			print(e, f)
			print(example.question_text)
			print([(int(att.item() * 1000), token) for att, token in zip(interp['attribution'], question_tokens)])
			start = max(0, interp['prelim_result']['start_index'] - 20)
			end = min(interp['prelim_result']['end_index'] + 20, len(interp['feature'].tokens))
			print(interp['prelim_result'], interp['example'].answers)
			print(context)
			
			if f == 0:
				print("hello word, can you see me, this is file: ", __name__, " line: ")
			print('\n')
	
	# show_density(sorted_score[sorted_exact == 0], step=50)
	# print("#"*50)
	# show_density(sorted_score[sorted_exact == 1], step=50)

	
	FAUC = f1auc_score(best_aggeragate_confidence, f1)
	AUC = auc_score(best_aggeragate_confidence, exact)
	ACC = None
	# FAUC, AUC, ACC = calib_seperate_point_evaluate(all_results, exact, f1)
	return results, FAUC, AUC, ACC


def main():
	parser = argparse.ArgumentParser()
	
	register_args(parser)
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
	parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
	parser.add_argument(
		"--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
	)
	
	parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
	parser.add_argument(
		"--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
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
			model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)  # , force_download=True)
			model.to(args.device)
			
			# Evaluate
			ids = 2
			
			# nlp = stanza.Pipeline('en', processors='tokenize,pos')
			# args.nlp = nlp
			
			for dataset_name in ["bioasq", "hotpot", "naturalqa", "newsqa", "searchqa", "textbookqa",
			                     "trivia", "squad"][-1:]:  # [ids:ids + 1]:
				args.dataset_name = dataset_name
				logger.info(dataset_name)
				args.predict_file = "outputs//dev_{}.json".format(dataset_name)
				args.output_dir = "predictions//dev_{}.json".format(dataset_name)
				
				# interp_dict = build_file_dict(args)
				# args.interp_dict = interp_dict
				
				result, FAUC, AUC, ACC = evaluate_show_order(args, model, tokenizer, prefix=global_step)
				
				result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
				results.update(result)
				logger.info("{}   FAUC: {}   AUC: {}     ACC: {}".format(dataset_name, FAUC, AUC, ACC))
	
	logger.info("Results: {}".format(results))
	
	return results

