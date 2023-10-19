import torch
import numpy as np
import copy
import random
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


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


saved_data = torch.load("saved_data.pth")

logit_test = saved_data["logit_test"]
labels_test = saved_data["label_test"]
prob_test = saved_data["prob_test"]
new_label_list = saved_data["new_label_list"]
print("hello world")

new_label_list = np.array(new_label_list).reshape(10, 15)
fprs = [[], [], []]
aurocs = [[], [], []]

super_fprs = [[], [], []]
super_aurocs = [[], [], []]
iterations = 10
for _ in tqdm(range(iterations), ascii=True):
    copy_label_list = copy.deepcopy(new_label_list)

    domain_ids = [i for i in range(10)]
    random.shuffle(domain_ids)
    copy_label_list = copy_label_list[domain_ids]
    correlated = copy_label_list[:10].reshape(-1)

    for i in range(10):
        np.random.shuffle(copy_label_list[i])
    indep = copy_label_list[:, :3].reshape(-1)

    for i in range(10):
        np.random.shuffle(copy_label_list[i])
    mid = copy_label_list[:5, :6].reshape(-1)

    for order, class_ids in enumerate([correlated]):  # [indep, mid, correlated]):
        select_vector = torch.ones_like(labels_test) == 0
        for i, each in enumerate(labels_test):
            if each.item() in class_ids or each.item() == 150:
                select_vector[i] = True
            else:
                select_vector[i] = False
        logit_test_part = logit_test[select_vector]
        labels_test_part = labels_test[select_vector]
        id_logit_test = logit_test_part[labels_test_part != 150]
        ood_logit_test = logit_test_part[labels_test_part == 150]

        id_logit_part = id_logit_test[:, class_ids]
        ood_logit_part = ood_logit_test[:, class_ids]

        id_prob_part = torch.softmax(id_logit_part, dim=-1)
        ood_prob_part = torch.softmax(ood_logit_part, dim=-1)

        oos_msp = ood_prob_part.max(dim=-1)[0].detach().cpu().numpy()
        id_msp = id_prob_part.max(dim=-1)[0].detach().cpu().numpy()
        y = np.concatenate((np.ones_like(oos_msp), np.zeros_like(id_msp)))
        scores = np.concatenate((oos_msp, id_msp))
        fpr_tpr95 = ErrorRateAt95Recall1(y, 1 - scores) * 100
        auroc = 100 * roc_auc_score(y, 1 - scores)
        print("MSP")
        print(fpr_tpr95, auroc)
        fprs[order].append(fpr_tpr95)
        aurocs[order].append(auroc)

        print("SuperProb")
        oos_msp_ids = ood_prob_part.max(dim=-1)[1].detach().cpu().numpy()
        id_msp_ids = id_prob_part.max(dim=-1)[1].detach().cpu().numpy()
        oos_msp = np.array(
            [sum(op[(oid // 15) * 15:(oid // 15 + 1) * 15]).item() for op, oid in zip(ood_prob_part, oos_msp_ids)])
        id_msp = np.array(
            [sum(op[(oid // 15) * 15:(oid // 15 + 1) * 15]).item() for op, oid in zip(id_prob_part, id_msp_ids)])
        y = np.concatenate((np.ones_like(oos_msp), np.zeros_like(id_msp)))
        scores = np.concatenate((oos_msp, id_msp))
        fpr_tpr95 = ErrorRateAt95Recall1(y, 1 - scores) * 100
        auroc = 100 * roc_auc_score(y, 1 - scores)
        print(fpr_tpr95, auroc)
        super_fprs[order].append(fpr_tpr95)
        super_aurocs[order].append(auroc)

print("hello")
for i in range(3):
    print(sum(aurocs[i]) / iterations, sum(fprs[i]) / iterations)
    print(sum(super_aurocs[i]) / iterations, sum(super_fprs[i]) / iterations)
