import copy

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import pandas as pd
import time
from sklearn.metrics import log_loss, brier_score_loss
from os.path import join
import sklearn.metrics as metrics
# Imports to get "utility" package
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath("utility"))))
from utility.unpickle_probs import unpickle_probs
from utility.evaluation import ECE, MCE
from sklearn.metrics import roc_auc_score


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)


class HistogramBinning():
    def __init__(self, M=15):
        self.bin_size = 1. / M  # Calculate bin size
        self.conf = []  # Initiate confidence list
        self.upper_bounds = np.arange(self.bin_size, 1 + self.bin_size, self.bin_size)  # Set bin bounds for intervals

    def _get_conf(self, conf_thresh_lower, conf_thresh_upper, probs, true):
        filtered = [x[0] for x in zip(true, probs) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
        nr_elems = len(filtered)  # Number of elements in the list.

        if nr_elems < 1:
            return 0
        else:
            conf = sum(filtered) / nr_elems  # Sums positive classes
            return conf

    def fit(self, probs, true):
        conf = []
        for conf_thresh in self.upper_bounds:
            temp_conf = self._get_conf((conf_thresh - self.bin_size), conf_thresh, probs=probs, true=true)
            conf.append(temp_conf)

        self.conf = conf

    def predict(self, probs):
        for i, prob in enumerate(probs):
            idx = np.searchsorted(self.upper_bounds, prob)
            probs[i] = self.conf[idx]

        return probs


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


def logits_postprocess(logit, alpha):
    max_ids = np.argmax(logit)
    left = max_ids % 5
    logit_sum = 0

    superclass = True
    count = 0
    if not superclass:
        for i in range(20):  # (max_ids - left, max_ids - left + 5):# range(20):
            if max_ids - left > i or max_ids - left + 5 < i:
                logit_sum += logit[i]
                count += 1
                if count == 5:
                    break
        # if i != max_ids and logit[i] > 0:
        #     logit_sum += logit[i]
    else:
        for i in range(max_ids - left, max_ids - left + 5):  # range(20):
            if i != max_ids and logit[i] > 0:
                logit_sum += logit[i]

    logit[max_ids] += alpha * logit_sum
    logit[max_ids] = min(100.0, logit[max_ids])

    if logit[max_ids] != max(logit):
        logit[max_ids] = max(logit) + 0.000001
    # logit[max_ids] = logit(1., prob[max_ids])
    # logit[max_ids] = logit(0., prob[max_ids])
    return logit


class TemperatureScaling():
    def __init__(self, temp=1, maxiter=50, solver="BFGS"):
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
        self.alpha = 0
        self.alpha_fd = 0.25
        self.alpha_ood = 0.25

    def fit_ood(self, id_logits, ood_logits):
        best_alpha = 0
        best_auroc = 0

        for i in range(125):
            try:
                alpha = -0.5 + i * 0.01
                id_logits_subset = copy.deepcopy(id_logits)
                for prob in id_logits_subset:
                    prob = logits_postprocess(prob, alpha)
                ood_logits_subset = copy.deepcopy(ood_logits)
                for prob in ood_logits_subset:
                    prob = logits_postprocess(prob, alpha)

                id_probs = softmax(id_logits_subset)
                ood_probs = softmax(ood_logits_subset)

                idp = np.max(id_probs, axis=1)
                oodp = np.max(ood_probs, axis=1)

                y = np.concatenate((np.ones_like(oodp), np.zeros_like(idp)))
                scores = np.concatenate((oodp, idp))
                # fpr_tpr95 = ErrorRateAt95Recall1(y, 1 - scores)
                auroc = 100 * roc_auc_score(y, 1 - scores)
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_alpha = alpha
                    print("best auroc: {}, best alpha: {}".format(best_auroc, best_alpha))
            except:
                pass
        self.alpha_ood = best_alpha

    def _loss_fun_ood(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        probs_copy = copy.deepcopy(probs)
        for prob in probs_copy:
            prob = logits_postprocess(prob, x)
        scaled_probs = self.predict(probs_copy, 1)
        probs_pred = np.max(scaled_probs, axis=1).reshape(-1, 1)
        probs = np.concatenate([1 - probs_pred, probs_pred], axis=-1)
        loss = log_loss(y_true=true, y_pred=probs)
        return loss

    def transform_logits_ood(self, logits):
        for prob in logits:
            prob = logits_postprocess(prob, self.alpha_ood)
        return logits

    def fit_fd(self, logits, accuracy):
        best_auroc = -100000
        best_alpha = 0

        for i in range(200):
            alpha = -0.3 + i * 0.003

            logits_subset = copy.deepcopy(logits)
            for prob in logits_subset:
                prob = logits_postprocess(prob, alpha)
            probs_subset = softmax(logits_subset)

            confs, accs = np.array([max(prob) for prob in probs_subset]), accuracy
            labels = 1 - np.array(accs)
            scores = 1 - np.array(confs)
            auroc = 100 * roc_auc_score(labels, scores)

            if auroc > best_auroc:
                best_auroc = auroc
                best_alpha = alpha

        self.alpha_fd = best_alpha

    def _loss_fun_fd(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        probs_copy = copy.deepcopy(probs)
        for prob in probs_copy:
            prob = logits_postprocess(prob, x)
        scaled_probs = self.predict(probs_copy, 1)
        probs_pred = np.max(scaled_probs, axis=1).reshape(-1, 1)
        probs = np.concatenate([1 - probs_pred, probs_pred], axis=-1)
        loss = log_loss(y_true=true, y_pred=probs)
        return loss

    def transform_logits_fd(self, logits):
        for prob in logits:
            prob = logits_postprocess(prob, self.alpha_fd)
        return logits

    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x)
        loss = log_loss(y_true=true, y_pred=scaled_probs)
        return loss

    def _loss_fun_my(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        probs_copy = copy.deepcopy(probs)
        for prob in probs_copy:
            prob = logits_postprocess(prob, x[0])
        scaled_probs = self.predict(probs_copy)
        loss = log_loss(y_true=true, y_pred=scaled_probs)
        return loss

    # Find the temperature
    def fit(self, logits, true):
        true = true.flatten()  # Flatten y_val
        opt = minimize(self._loss_fun, x0=1, args=(logits, true), options={'maxiter': self.maxiter}, method=self.solver)
        self.temp = opt.x[0]

        return opt

    def fit_my(self, logits, true, accuracy_subset):
        min_ece = 100000
        best_alpha = 0
        best_temp = 0

        for j in range(50):
            temp = 2 + j * 0.04
            for i in range(10):
                alpha = -0.1 + i * 0.02

                logits_subset = copy.deepcopy(logits)
                for prob in logits_subset:
                    prob = logits_postprocess(prob, alpha)
                probs_subset = self.predict(logits_subset, temp)

                confs, accs = np.array([max(prob) for prob in probs_subset]), accuracy_subset
                ece, _, _ = ECE(confs, accs)
                if ece < min_ece:
                    min_ece = ece
                    best_alpha = alpha
                    best_temp = temp
        self.alpha = best_alpha
        self.temp = best_temp
        print("alpha: {}   temp: {}".format(best_alpha, best_temp))
        return None

    def transform_logits(self, logits):
        for prob in logits:
            prob = logits_postprocess(prob, self.alpha)
        return logits

    def fit_postprocess(self, logits, true, postprocess, accuracy_subset):
        true = true.flatten()  # Flatten y_val
        min_ece = 100000
        best_temp = 0

        for i in range(300):
            temp = 1 + 0.03 * i
            scaled_probs = self.predict(logits, temp)

            for prob in scaled_probs:
                prob = postprocess(prob)

            confs, accs = np.array([max(prob) for prob in scaled_probs]), accuracy_subset
            ece, _, _ = ECE(confs, accs)
            if ece < min_ece:
                min_ece = ece
                best_temp = temp
                # print("new best temp/ece: {} {}".format(temp, ece))
        self.temp = best_temp

    def _loss_fun_postprocess(self, x, probs, true, postprocess):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x)
        for i, prob in enumerate(scaled_probs):
            scaled_probs[i] = postprocess(prob)
        loss = log_loss(y_true=true, y_pred=scaled_probs)
        return loss

    def predict(self, logits, temp=None):
        if not temp:
            return softmax(logits / self.temp)
        else:
            return softmax(logits / temp)

    def predict_my(self, logits):
        for prob in logits:
            prob = logits_postprocess(prob, self.alpha)
        return softmax(logits / self.temp)


class TemperatureScalingImageNet():
    def __init__(self, temp=1, maxiter=50, solver="BFGS"):
        self.temp = 0

    # Find the temperature
    def fit(self, logits, accs):
        best_temp = 0.3
        best_ece = 10000
        for i in range(50):
            temp = 0.01 * i + 1
            scaled_probs = self.predict(logits, temp)
            if np.isnan(scaled_probs).sum() > 0:
                continue

            scaled_probs = np.array([max(prob) for prob in scaled_probs])
            ece, underconfident_ece, overconfident_ece = ECE_of_mine(scaled_probs, accs, bin_size=0.1)
            if ece < best_ece:
                best_temp = temp
                best_ece = ece
            # print("new best temp: {} ".format(best_temp))
            # print("new best ece: {} ".format(best_ece))

        self.temp += best_temp

    def predict(self, logits, temp=None):
        """
        Scales logits based on the temperature and returns calibrated probabilities

        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.

        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """

        if not temp:
            return softmax(logits / self.temp)
        else:
            return softmax(logits / temp)


def evaluate(probs, y_true, verbose=False, normalize=False, bins=15):
    preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction

    if normalize:
        confs = np.max(probs, axis=1) / np.sum(probs, axis=1)
    else:
        confs = np.max(probs, axis=1)  # Take only maximum confidence

    accuracy = metrics.accuracy_score(y_true, preds) * 100
    error = 100 - accuracy

    # Calculate ECE
    ece = ECE(confs, preds, y_true, bin_size=1 / bins)
    # Calculate MCE
    mce = MCE(confs, preds, y_true, bin_size=1 / bins)

    loss = log_loss(y_true=y_true, y_pred=probs)

    y_prob_true = np.array([probs[i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class
    brier = brier_score_loss(y_true=y_true, y_prob=y_prob_true)  # Brier Score (MSE)

    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("MCE:", mce)
        print("Loss:", loss)
        print("brier:", brier)

    return (error, ece, mce, loss, brier)


def cal_results(fn, path, files, m_kwargs={}, approach="all"):
    df = pd.DataFrame(columns=["Name", "Error", "ECE", "MCE", "Loss", "Brier"])

    total_t1 = time.time()

    for i, f in enumerate(files):

        name = "_".join(f.split("_")[1:-1])
        print(name)
        t1 = time.time()

        FILE_PATH = join(path, f)
        (logits_val, y_val), (logits_test, y_test) = unpickle_probs(FILE_PATH)

        if approach == "all":

            y_val = y_val.flatten()

            model = fn(**m_kwargs)

            model.fit(logits_val, y_val)

            probs_val = model.predict(logits_val)
            probs_test = model.predict(logits_test)

            error, ece, mce, loss, brier = evaluate(softmax(logits_test), y_test, verbose=True)  # Test before scaling
            error2, ece2, mce2, loss2, brier2 = evaluate(probs_test, y_test, verbose=False)

            print("Error %f; ece %f; mce %f; loss %f, brier %f" % evaluate(probs_val, y_val, verbose=False,
                                                                           normalize=True))


        else:  # 1-vs-k models
            probs_val = softmax(logits_val)  # Softmax logits
            probs_test = softmax(logits_test)
            K = probs_test.shape[1]

            # Go through all the classes
            for k in range(K):
                # Prep class labels (1 fixed true class, 0 other classes)
                y_cal = np.array(y_val == k, dtype="int")[:, 0]

                # Train model
                model = fn(**m_kwargs)
                model.fit(probs_val[:, k], y_cal)  # Get only one column with probs for given class "k"

                probs_val[:, k] = model.predict(probs_val[:, k])  # Predict new values based on the fittting
                probs_test[:, k] = model.predict(probs_test[:, k])

                # Replace NaN with 0, as it should be close to zero  # TODO is it needed?
                idx_nan = np.where(np.isnan(probs_test))
                probs_test[idx_nan] = 0

                idx_nan = np.where(np.isnan(probs_val))
                probs_val[idx_nan] = 0

            # Get results for test set
            error, ece, mce, loss, brier = evaluate(softmax(logits_test), y_test, verbose=True, normalize=False)
            error2, ece2, mce2, loss2, brier2 = evaluate(probs_test, y_test, verbose=False, normalize=True)

            print("Error %f; ece %f; mce %f; loss %f, brier %f" % evaluate(probs_val, y_val, verbose=False,
                                                                           normalize=True))

        df.loc[i * 2] = [name, error, ece, mce, loss, brier]
        df.loc[i * 2 + 1] = [(name + "_calib"), error2, ece2, mce2, loss2, brier2]

        t2 = time.time()
        print("Time taken:", (t2 - t1), "\n")

    total_t2 = time.time()
    print("Total time taken:", (total_t2 - total_t1))

    return df
