import numpy as np
import torch
import random
from sklearn.metrics import roc_auc_score
import copy



def find_first_less_than_confidence_threshold(matrix, threshold):
	for i, e in enumerate(matrix):
		if e > threshold:
			continue
		else:
			return i
	return i


def show_density(sorted_score, step=100, plot=False):
    sorted_score = sorted(sorted_score, reverse=True)

    thresholds = [i / step for i in range(0, step)]
    confidence_rate_list = []

    for threshold in thresholds:
        num = find_first_less_than_confidence_threshold(sorted_score, threshold)
        confidence_rate_list.append(num)

    confidence_rate_list.append(0)
    confidence_rate_list = [abs(confidence_rate_list[i + 1] - confidence_rate_list[i]) for i in
                            range(len(confidence_rate_list) - 1)]
    confidence_rate_list = np.array(confidence_rate_list) / len(sorted_score)
    if plot:
        for i in confidence_rate_list:
            print(i)

    return confidence_rate_list


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


np.random.seed(0)
random.seed(0)

torch.random.seed()

def vision_toy_example():
    mul = 10
    sigma = 1
    total_dim = 2
    alpha = 0.5
    iterations = 30
    aurocs = [[] for _ in range(iterations)]
    fpr95s = [[] for _ in range(iterations)]

    superlogits_aurocs = [[] for _ in range(iterations)]
    superlogits_fpr95s = [[] for _ in range(iterations)]
    for iters in range(iterations):
        for m in [0]:

            if m == 1:
                u_class0 = [0, mul]
                u_class1 = [mul, mul]
            else:
                u_class0 = [mul, 0]
                u_class1 = [0, mul]

            for _ in range(1):
                # id
                u_id = [np.array(u_class0), np.array(u_class1)]
                generate_id_data = []
                for _ in range(2000):
                    label = np.random.randint(0, 2)
                    generate_x = np.array([np.random.normal(u, sigma) for u in u_id[label]])
                    generate_id_data.append(generate_x)
                generate_id_data = np.stack(generate_id_data)
                for each in generate_id_data:
                    print(each[0], each[1])

                # ood
                distance = 2
                u_ood = (np.array(u_class0) + np.array(u_class1)) / (2 + distance)
                generate_ood_data = []
                for _ in range(2000):
                    generate_x = np.array(
                        [np.random.normal(u, sigma) for u in u_ood])  # np.random.randn(total_dim) * u_ood
                    generate_ood_data.append(generate_x)
                generate_ood_data = np.stack(generate_ood_data)
                for each in generate_ood_data:
                    print(each[0], each[1])

                # compute_prob
                id_logit0 = np.sum(W0 * generate_id_data, axis=-1).reshape(-1, 1)
                id_logit1 = np.sum(W1 * generate_id_data, axis=-1).reshape(-1, 1)
                id_logits = np.concatenate([id_logit0, id_logit1], axis=-1)
                id_logits = torch.from_numpy(id_logits)
                id_probs = torch.softmax(id_logits, dim=-1)
                id_msp = torch.max(id_probs, dim=-1)[0]

                ood_logit0 = np.sum(W0 * generate_ood_data, axis=-1).reshape(-1, 1)
                ood_logit1 = np.sum(W1 * generate_ood_data, axis=-1).reshape(-1, 1)
                ood_logits = np.concatenate([ood_logit0, ood_logit1], axis=-1)
                ood_logits = torch.from_numpy(ood_logits)
                ood_probs = torch.softmax(ood_logits, dim=-1)
                ood_msp = torch.max(ood_probs, dim=-1)[0]

                y = np.concatenate((np.ones_like(ood_msp), np.zeros_like(id_msp)))
                scores = np.concatenate((ood_msp, id_msp))
                fpr_tpr95 = ErrorRateAt95Recall1(y, 1 - scores) * 100
                auroc = 100 * roc_auc_score(y, 1 - scores)
                print("shared feaure/distance: ", m, "FPR95: ", fpr_tpr95, "AUROC: ", auroc)
                aurocs[iters].append(auroc)
                fpr95s[iters].append(fpr_tpr95)

                copy_id_logits = copy.deepcopy(id_logits)
                copy_ood_logits = copy.deepcopy(ood_logits)

                eps = 2
                for i, logits in enumerate(copy_id_logits):
                    if logits[0] > logits[1]:
                        copy_id_logits[i][0] += logits[1] / eps
                    else:
                        copy_id_logits[i][1] += logits[0] / eps
                id_probs = torch.softmax(copy_id_logits, dim=-1)
                id_msp = torch.max(id_probs, dim=-1)[0]

                for i, logits in enumerate(copy_ood_logits):
                    if logits[0] > logits[1]:
                        copy_ood_logits[i][0] += logits[1] / eps
                    else:
                        copy_ood_logits[i][1] += logits[0] / eps
                ood_probs = torch.softmax(copy_ood_logits, dim=-1)
                ood_msp = torch.max(ood_probs, dim=-1)[0]

                y = np.concatenate((np.ones_like(ood_msp), np.zeros_like(id_msp)))
                scores = np.concatenate((ood_msp, id_msp))
                fpr_tpr95 = ErrorRateAt95Recall1(y, 1 - scores) * 100
                auroc = 100 * roc_auc_score(y, 1 - scores)
                print("SuperLogit------------------", "FPR95: ", fpr_tpr95, "AUROC: ", auroc)
                superlogits_aurocs[iters].append(auroc)
                superlogits_fpr95s[iters].append(fpr_tpr95)

    aurocs = np.array(aurocs).mean(axis=0)
    fpr95s = np.array(fpr95s).mean(axis=0)
    superlogits_aurocs = np.array(superlogits_aurocs).mean(axis=0)
    superlogits_fpr95s = np.array(superlogits_fpr95s).mean(axis=0)

    for aur, fpr, saur, sfpr in zip(aurocs, fpr95s, superlogits_aurocs, superlogits_fpr95s):
        print(aur, '\t', fpr, '\t', saur, '\t', sfpr)

# vision_toy_example()

mul = 1
sigma = 2
total_dim = 100
alpha = 0.5
distance = 2
iterations = 20
aurocs = [[] for _ in range(iterations)]
fpr95s = [[] for _ in range(iterations)]

superlogits_aurocs = [[] for _ in range(iterations)]
superlogits_fpr95s = [[] for _ in range(iterations)]

confidence_dict = {}
from tqdm import tqdm
for iters in tqdm(range(iterations), ascii=True):
    for m in range(0, total_dim):
        u_share = [mul for _ in range(m)]
        u_class0 = u_share + [mul for _ in range((total_dim-m)//2)] + [0 for _ in range((total_dim-m)//2)]
        u_class1 = u_share + [0 for _ in range((total_dim-m)//2)] + [mul for _ in range((total_dim-m)//2)]
        u_class0 = u_class0 + (total_dim-len(u_class0))*[0]
        u_class1 = u_class1 + (total_dim-len(u_class1))*[mul]


        W0 = np.zeros_like(u_class0)
        for i, mean in enumerate(u_class0):
            if mean == 0:
                W0[i] = 0
            else:
                W0[i] = 1

        W1 = np.zeros_like(u_class1)
        for i, mean in enumerate(u_class1):
            if mean == 0:
                W1[i] = 0
            else:
                W1[i] = 1

        # id
        u_id = [np.array(u_class0), np.array(u_class1)]
        generate_id_data = []
        for _ in range(2000):
            label = np.random.randint(0, 2)
            generate_x = np.array([np.random.normal(u, sigma) for u in u_id[label]])
            generate_id_data.append(generate_x)
        generate_id_data = np.stack(generate_id_data)

        # ood
        u_ood = (np.array(u_class0) + np.array(u_class1)) / (2 + distance)
        generate_ood_data = []
        for _ in range(2000):
            generate_x = np.array([np.random.normal(u, sigma) for u in u_ood]) # np.random.randn(total_dim) * u_ood
            generate_ood_data.append(generate_x)
        generate_ood_data = np.stack(generate_ood_data)

        # compute_prob
        id_logit0 = np.sum(W0*generate_id_data, axis=-1).reshape(-1,1)
        id_logit1 = np.sum(W1*generate_id_data, axis=-1).reshape(-1,1)
        id_logits = np.concatenate([id_logit0, id_logit1], axis=-1)
        id_logits = torch.from_numpy(id_logits)
        id_probs = torch.softmax(id_logits, dim=-1)
        id_msp = torch.max(id_probs, dim=-1)[0]
        print("m is {}     corr intensity is {}".format(m, (m/(50+0.5*m))**2))
        # if str(m) not in confidence_dict.keys():
        #     confidence_dict[str(m)] = []
        # cd = show_density(id_msp)
        # confidence_dict[str(m)].append(cd)
        # # show_density(id_msp)

        ood_logit0 = np.sum(W0*generate_ood_data, axis=-1).reshape(-1,1)
        ood_logit1 = np.sum(W1*generate_ood_data, axis=-1).reshape(-1,1)
        ood_logits = np.concatenate([ood_logit0, ood_logit1], axis=-1)
        ood_logits = torch.from_numpy(ood_logits)
        ood_probs = torch.softmax(ood_logits, dim=-1)
        ood_msp = torch.max(ood_probs, dim=-1)[0]
        y = np.concatenate((np.ones_like(ood_msp), np.zeros_like(id_msp)))
        scores = np.concatenate((ood_msp, id_msp))
        fpr_tpr95 = ErrorRateAt95Recall1(y, 1 - scores) * 100
        auroc = 100 * roc_auc_score(y, 1 - scores)
        print("shared feaure: ", m, "FPR95: ", fpr_tpr95, "AUROC: ", auroc)

        aurocs[iters].append(auroc)
        fpr95s[iters].append(fpr_tpr95)
        copy_id_logits = copy.deepcopy(id_logits)
        copy_ood_logits = copy.deepcopy(ood_logits)
        eps = 2
        # for i, logits in enumerate(copy_id_logits):
        #     if logits[0] > logits[1]:
        #         copy_id_logits[i][0] += logits[1]/eps
        #     else:
        #         copy_id_logits[i][1] += logits[0]/eps
        id_probs = copy_id_logits # torch.softmax(copy_id_logits, dim=-1)
        id_msp = torch.max(id_probs, dim=-1)[0]
        # for i, logits in enumerate(copy_ood_logits):
        #     if logits[0] > logits[1]:
        #         copy_ood_logits[i][0] += logits[1]/eps
        #     else:
        #         copy_ood_logits[i][1] += logits[0]/eps
        ood_probs = copy_ood_logits # torch.softmax(copy_ood_logits, dim=-1)
        ood_msp = torch.max(ood_probs, dim=-1)[0]
        y = np.concatenate((np.ones_like(ood_msp), np.zeros_like(id_msp)))
        scores = np.concatenate((ood_msp, id_msp))
        fpr_tpr95 = ErrorRateAt95Recall1(y, 1 - scores) * 100
        auroc = 100 * roc_auc_score(y, 1 - scores)
        print("CoMe: ", "FPR95: ", fpr_tpr95, "AUROC: ", auroc)
        superlogits_aurocs[iters].append(auroc)
        superlogits_fpr95s[iters].append(fpr_tpr95)

for name, distribution in confidence_dict.items():
    print("m=", name, "#"*50)
    distribution = np.array(distribution).mean(axis=0)
    for each in distribution:
        print(each)

aurocs = np.array(aurocs).mean(axis=0)
fpr95s = np.array(fpr95s).mean(axis=0)
superlogits_aurocs = np.array(superlogits_aurocs).mean(axis=0)
superlogits_fpr95s = np.array(superlogits_fpr95s).mean(axis=0)

for aur, fpr, saur, sfpr in zip(aurocs, fpr95s, superlogits_aurocs, superlogits_fpr95s):
    print(aur, '\t', fpr, '\t', saur, '\t', sfpr)

