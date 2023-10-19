import argparse
import warnings

warnings.filterwarnings('ignore')
from tqdm import tqdm
import copy
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from cal_methods import TemperatureScaling
from sklearn.metrics import roc_auc_score

coarse_to_fine_dict = {
    "aquatic mammals": ["beaver", "dolphin", "otter", 'seal', 'whale'],
    "fish": ["aquarium_fish", "flatfish", "ray", 'shark', 'trout'],
    "flowers": ["orchid", "poppy", "rose", 'sunflower', 'tulip'],
    "food containers": ["bottle", "bowl", "can", 'cup', 'plate'],
    "fruit and vegetables": ["apple", "mushroom", "orange", "pear", 'sweet_pepper'],
    "household electrical devices": ["clock", "keyboard", "lamp", 'telephone', 'television'],
    "household furniture": ["bed", "chair", "couch", 'table', 'wardrobe'],
    "insects": ["bee", "beetle", "butterfly", 'caterpillar', 'cockroach'],
    "large carnivores": ["bear", "leopard", "lion", 'tiger', 'wolf'],
    "large man-made outdoor things": ["bridge", "castle", "house", 'road', 'skyscraper'],
    "large natural outdoor scenes": ["cloud", "forest", "mountain", 'plain', 'sea'],
    "large omnivores and herbivores": ["camel", "cattle", "chimpanzee", 'elephant', 'kangaroo'],
    "medium-sized mammals": ["fox", "porcupine", "possum", 'raccoon', 'skunk'],
    "non-insect invertebrates": ["crab", "lobster", "snail", 'spider', 'worm'],
    "people": ["baby", "boy", "girl", 'man', 'woman'],
    "reptiles": ["crocodile", "dinosaur", "lizard", 'snake', 'turtle'],
    "small mammals": ["hamster", "mouse", "rabbit", 'shrew', 'squirrel'],
    "trees": ["maple_tree", "oak_tree", "palm_tree", 'pine_tree', 'willow_tree'],
    "vehicles 1": ["bicycle", "bus", "motorcycle", 'pickup_truck', 'train'],
    "vehicles 2": ["lawn_mower", "rocket", "streetcar", 'tank', 'tractor']
}


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


def filter_to_subset(logits, ys, reserve_class):
    logits_subset, ys_subset = [], []
    for logit, y in zip(logits, ys):
        if y in reserve_class:
            logits_subset.append(logit.reshape(100, 1)[reserve_class].flatten())
            ys_subset.append(reserve_class.index(y))
    return np.array(logits_subset), np.array(ys_subset)


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


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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
    return confidence_rate_list


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):

        out = self.conv1(self.relu(self.bn1(x)))

        torch_model.record(out)

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)

        out = self.conv2(self.relu(self.bn2(out)))
        torch_model.record(out)

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        torch_model.record(out)

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(in_planes + i * growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        t = self.layer(x)
        torch_model.record(t)
        return t


class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(DenseNet3, self).__init__()

        self.collecting = False

        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n / 2
            block = BottleneckBlock
        else:
            block = BasicBlock
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)

    def load(self, path="saved data/densenet_cifar100.pth"):
        tm = torch.load(path, map_location="cpu")
        self.load_state_dict(tm.state_dict(), strict=False)

    def record(self, t):
        if self.collecting:
            self.gram_feats.append(t)

    def gram_feature_list(self, x):
        self.collecting = True
        self.gram_feats = []
        self.forward(x)
        self.collecting = False
        temp = self.gram_feats
        self.gram_feats = []
        return temp

    def get_min_max(self, data, power):
        mins = []
        maxs = []

        for i in range(0, len(data), 64):
            batch = data[i:i + 64].cuda()
            feat_list = self.gram_feature_list(batch)
            for L, feat_L in enumerate(feat_list):
                if L == len(mins):
                    mins.append([None] * len(power))
                    maxs.append([None] * len(power))

                for p, P in enumerate(power):
                    g_p = G_p(feat_L, P)

                    current_min = g_p.min(dim=0, keepdim=True)[0]
                    current_max = g_p.max(dim=0, keepdim=True)[0]

                    if mins[L][p] is None:
                        mins[L][p] = current_min
                        maxs[L][p] = current_max
                    else:
                        mins[L][p] = torch.min(current_min, mins[L][p])
                        maxs[L][p] = torch.max(current_max, maxs[L][p])

        return mins, maxs

    def get_deviations(self, data, power, mins, maxs):
        deviations = []

        for i in range(0, len(data), 64):
            batch = data[i:i + 64].cuda()
            feat_list = self.gram_feature_list(batch)
            batch_deviations = []
            for L, feat_L in enumerate(feat_list):
                dev = 0
                for p, P in enumerate(power):
                    g_p = G_p(feat_L, P)

                    dev += (F.relu(mins[L][p] - g_p) / torch.abs(mins[L][p] + 10 ** -6)).sum(dim=1, keepdim=True)
                    dev += (F.relu(g_p - maxs[L][p]) / torch.abs(maxs[L][p] + 10 ** -6)).sum(dim=1, keepdim=True)
                batch_deviations.append(dev.cpu().detach().numpy())
            batch_deviations = np.concatenate(batch_deviations, axis=1)
            deviations.append(batch_deviations)
        deviations = np.concatenate(deviations, axis=0)

        return deviations


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        t = self.conv1(x)
        out = F.relu(self.bn1(t))
        torch_model.record(t)
        torch_model.record(out)
        t = self.conv2(out)
        out = self.bn2(self.conv2(out))
        torch_model.record(t)
        torch_model.record(out)
        t = self.shortcut(x)
        out += t
        torch_model.record(t)
        out = F.relu(out)
        torch_model.record(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.collecting = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y

    def record(self, t):
        if self.collecting:
            self.gram_feats.append(t)

    def gram_feature_list(self, x):
        self.collecting = True
        self.gram_feats = []
        self.forward(x)
        self.collecting = False
        temp = self.gram_feats
        self.gram_feats = []
        return temp

    def load(self, path="saved data/resnet_cifar100.pth"):
        tm = torch.load(path, map_location="cpu")
        self.load_state_dict(tm)

    def get_min_max(self, data, power):
        mins = []
        maxs = []

        for i in range(0, len(data), 128):
            batch = data[i:i + 128].cuda()
            feat_list = self.gram_feature_list(batch)
            for L, feat_L in enumerate(feat_list):
                if L == len(mins):
                    mins.append([None] * len(power))
                    maxs.append([None] * len(power))

                for p, P in enumerate(power):
                    g_p = G_p(feat_L, P)

                    current_min = g_p.min(dim=0, keepdim=True)[0]
                    current_max = g_p.max(dim=0, keepdim=True)[0]

                    if mins[L][p] is None:
                        mins[L][p] = current_min
                        maxs[L][p] = current_max
                    else:
                        mins[L][p] = torch.min(current_min, mins[L][p])
                        maxs[L][p] = torch.max(current_max, maxs[L][p])

        return mins, maxs

    def get_deviations(self, data, power, mins, maxs):
        deviations = []

        for i in range(0, len(data), 128):
            batch = data[i:i + 128].cuda()
            feat_list = self.gram_feature_list(batch)
            batch_deviations = []
            for L, feat_L in enumerate(feat_list):
                dev = 0
                for p, P in enumerate(power):
                    g_p = G_p(feat_L, P)

                    dev += (F.relu(mins[L][p] - g_p) / torch.abs(mins[L][p] + 10 ** -6)).sum(dim=1, keepdim=True)
                    dev += (F.relu(g_p - maxs[L][p]) / torch.abs(maxs[L][p] + 10 ** -6)).sum(dim=1, keepdim=True)
                batch_deviations.append(dev.cpu().detach().numpy())
            batch_deviations = np.concatenate(batch_deviations, axis=1)
            deviations.append(batch_deviations)
        deviations = np.concatenate(deviations, axis=0)

        return deviations


torch_model = DenseNet3(100, num_classes=100)

torch_model.load()
torch_model.cuda()
torch_model.params = list(torch_model.parameters())
torch_model.eval()
print("Done")

batch_size = 128
mean = np.array([[125.3 / 255, 123.0 / 255, 113.9 / 255]]).T

std = np.array([[63.0 / 255, 62.1 / 255.0, 66.7 / 255.0]]).T
normalize = transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0))

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize

])
transform_test = transforms.Compose([
    transforms.CenterCrop(size=(32, 32)),
    transforms.ToTensor(),
    normalize
])
already_save = True
if not already_save:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=True, download=True,
                          transform=transform_train),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=False, transform=transform_test),
        batch_size=batch_size)
    torch_model.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        x = x.cuda()
        y = y.numpy()
        correct += (y == np.argmax(torch_model(x).detach().cpu().numpy(), axis=1)).sum()
        total += y.shape[0]
    print("Accuracy: ", correct / total)

    data_train = list(torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=True, download=True,
                          transform=transform_test),
        batch_size=1, shuffle=False))
    data = list(torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=False, download=True,
                          transform=transform_test),
        batch_size=1, shuffle=False))

    cifar10 = list(torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, download=True,
                         transform=transform_test),
        batch_size=1, shuffle=True))
    svhn = list(torch.utils.data.DataLoader(
        datasets.SVHN('data', split="test", download=True,
                      transform=transform_test),
        batch_size=1, shuffle=True))
    isun = list(torch.utils.data.DataLoader(
        datasets.ImageFolder("iSUN/", transform=transform_test), batch_size=1, shuffle=False))
    lsun_c = list(torch.utils.data.DataLoader(
        datasets.ImageFolder("LSUN/", transform=transform_test), batch_size=1, shuffle=True))
    lsun_r = list(torch.utils.data.DataLoader(
        datasets.ImageFolder("LSUN_resize/", transform=transform_test), batch_size=1, shuffle=True))
    tinyimagenet_c = list(torch.utils.data.DataLoader(
        datasets.ImageFolder("Imagenet/", transform=transform_test), batch_size=1, shuffle=True))
    tinyimagenet_r = list(torch.utils.data.DataLoader(
        datasets.ImageFolder("Imagenet_resize/", transform=transform_test), batch_size=1, shuffle=True))

    id_preds, id_confs, id_logits, id_ys = [], [], [], []

    for idx in range(0, len(data), 128):
        batch = torch.squeeze(torch.stack([x[0] for x in data[idx:idx + 128]]), dim=1).cuda()
        labels = [x[1] for x in data[idx:idx + 128]]

        logits = torch_model(batch)
        confs = F.softmax(logits, dim=1).cpu().detach().numpy()
        preds = np.argmax(confs, axis=1)
        logits = (logits.cpu().detach().numpy())

        id_confs.extend(confs)
        id_preds.extend(preds)
        id_logits.extend(logits)
        id_ys.extend(labels)

    print("Done")
    torch.save({"confs": id_confs, "preds": id_preds, "logits": id_logits, "ys": id_ys},
               "saved data/densenet_cifar100_id_probs_data")
    print("saved data/densenet_cifar100_id_probs_data saved")

    mappp = {"svhn": svhn, "isun": isun, "lsun_c": lsun_c, "lsun_r": lsun_r, "tinyimagenet_c": tinyimagenet_c,
             "tinyimagenet_r": tinyimagenet_r}
    for ood_dataset in ["svhn", "isun", "lsun_c", "lsun_r", "tinyimagenet_c", "tinyimagenet_r"]:
        ood_preds, ood_confs, ood_logits, ood_ys = [], [], [], []
        for idx in range(0, len(mappp[ood_dataset]), 128):
            batch = torch.squeeze(torch.stack([x[0] for x in mappp[ood_dataset][idx:idx + 128]]), dim=1).cuda()
            labels = [x[1] for x in data[idx:idx + 128]]

            logits = torch_model(batch)
            confs = F.softmax(logits, dim=1).cpu().detach().numpy()
            preds = np.argmax(confs, axis=1)
            logits = (logits.cpu().detach().numpy())

            ood_confs.extend(confs)
            ood_preds.extend(preds)
            ood_logits.extend(logits)
            ood_ys.extend(labels)

        print("Done")
        torch.save({"confs": ood_confs, "preds": ood_preds, "logits": ood_logits, "ys": ood_ys}, \
                   "saved data/densenet_cifar100_{}_probs_data".format(ood_dataset))
        print("saved data/densenet_cifar100_{}_probs_data saved".format(ood_dataset))

cifar100_fine_label_list = unpickle("saved data/meta")[b'fine_label_names']
cifar100_fine_label_list = [("#" + str(each)).replace("#b'", '').replace("\'", '') for each in cifar100_fine_label_list]
cifar100_fine_label_dict = {label: i for i, label in enumerate(cifar100_fine_label_list)}

fine_to_coarse_dict = {}
new_order = []

new_order_label = []
for k, v in coarse_to_fine_dict.items():
    for each in v:
        fine_to_coarse_dict[each] = k
        new_order.append(cifar100_fine_label_dict[each])
        new_order_label.append(each)
new_label = [0] * 100
for i, each in enumerate(new_order):
    new_label[each] = i

# logits = logits[:, new_order]
# ys = [new_label[each[0]] for each in ys]

id_probs = torch.load("saved data/densenet_cifar100_id_probs_data")
id_logits = np.array(id_probs["logits"])[:, new_order]
id_ys = [new_label[each[0]] for each in id_probs["ys"]]
acc = sum(np.argmax(id_logits, axis=1) == id_ys) / len(id_ys)
print("ACCURACY: {}".format(acc))

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('--method', type=str, default="msp", help="[msp, sl, sp]")

args = parser.parse_args()

ood_probs, ood_logits, ood_ys = {}, {}, {}
for ood_dataset in ["svhn", "isun", "lsun_c", "lsun_r", "tinyimagenet_c", "tinyimagenet_r"]:
    ood_probs[ood_dataset] = torch.load("saved data/densenet_cifar100_{}_probs_data".format(ood_dataset))
    ood_logits[ood_dataset] = np.array(ood_probs[ood_dataset]["logits"])[:, new_order]
    ood_ys[ood_dataset] = [new_label[each[0]] for each in ood_probs[ood_dataset]["ys"]]

cifar100_category = []
for i in range(20):
    cifar100_category.append([i * 5 + j for j in range(5)])

all_fpr_tpr95 = {}
all_auroc = {}
all_oodp = {}
all_idp = {}

run_iterations = 1
for ood_dataset in ["svhn", "isun", "lsun_c", "lsun_r", "tinyimagenet_c", "tinyimagenet_r"]:
    print("#" * 100)
    print("OOD dataset: ", ood_dataset)
    all_fpr_tpr95[ood_dataset] = [[] for _ in range(3)]
    all_auroc[ood_dataset] = [[] for _ in range(3)]
    all_oodp[ood_dataset] = [None for _ in range(3)]
    all_idp[ood_dataset] = [None for _ in range(3)]

    for _ in tqdm(range(run_iterations), ascii=True, leave=False, position=0):
        random_cifar100_category = copy.deepcopy(cifar100_category)
        for each in random_cifar100_category:
            random.shuffle(each)

        categories = [i for i in range(20)]
        random.shuffle(categories)
        independent = [i for i in range(100)]  # [each[0] for each in random_cifar100_category]
        correlated = []

        for i in range(20):
            correlated += random_cifar100_category[categories[i]]

        for i in range(3):
            reserve_class = [independent, correlated, correlated][i]
            id_logits_subset, id_ys_subset = filter_to_subset(id_logits, id_ys, reserve_class)
            ood_logits_subset, ood_ys_subset = filter_to_subset(ood_logits[ood_dataset], ood_ys[ood_dataset],
                                                                reserve_class)

            id_probs = softmax(id_logits_subset)
            ood_probs = softmax(ood_logits_subset)

            if args.method == "sl":
                if i >= 2:  # TODO design a learning method to reassignment confidence according label correlation
                    ts = TemperatureScaling()
                    hold_num = 0
                    logits_subset = np.concatenate([id_logits_subset[:hold_num], ood_logits_subset[:hold_num]], axis=0)
                    label_subset = np.concatenate([np.ones_like(id_logits_subset[:hold_num, 0]), \
                                                   np.zeros_like(ood_logits_subset[:hold_num, 0])])
                    # ts.fit_ood(id_logits_subset[:hold_num], ood_logits_subset[:hold_num])
                    id_logits_subset = ts.transform_logits_ood(id_logits_subset[hold_num:])
                    ood_logits_subset = ts.transform_logits_ood(ood_logits_subset[hold_num:])
                    id_probs = id_logits_subset  # softmax(id_logits_subset)
                    ood_probs = ood_logits_subset  # softmax(ood_logits_subset)
            elif args.method == "sp":
                if i >= 2:
                    def transform_probs(probs_subset):
                        for prob in probs_subset:
                            max_ids = np.argmax(prob)
                            # class_ids = [i for i in range(100)]
                            # class_ids.pop(max_ids)
                            # random.shuffle(class_ids)
                            left = max_ids % 5
                            if prob[max_ids] > 0:
                                prob[max_ids] += (sum(prob[max_ids - left: max_ids - left + 5]) - prob[max_ids])
                                # prob[max_ids] += (sum(prob[class_ids[:4]]) + prob[max_ids])
                                # prob[max_ids] = min(1., prob[max_ids])
                        return probs_subset


                    id_probs = transform_probs(id_probs)
                    ood_probs = transform_probs(ood_probs)
            else:
                pass

            # if i >= 2:  # TODO design a learning method to reassignment confidence according label correlation
            #     id_probs = transform_probs(id_probs)
            #     ood_probs = transform_probs(ood_probs)

            idp = np.max(id_probs, axis=1)
            oodp = np.max(ood_probs, axis=1)

            y = np.concatenate((np.ones_like(oodp), np.zeros_like(idp)))
            scores = np.concatenate((oodp, idp))
            fpr_tpr95 = ErrorRateAt95Recall1(y, 1 - scores)
            auroc = 100 * roc_auc_score(y, 1 - scores)

            all_fpr_tpr95[ood_dataset][i].append(fpr_tpr95)
            all_auroc[ood_dataset][i].append(auroc)

for ood_dataset in ["svhn", "isun", "lsun_c", "lsun_r", "tinyimagenet_c", "tinyimagenet_r"]:
    print("#" * 100)
    print("OOD dataset: ", ood_dataset)
    fpr_tpr95 = np.sum(np.array(all_fpr_tpr95[ood_dataset]), axis=-1) / run_iterations
    auroc = np.sum(np.array(all_auroc[ood_dataset]), axis=-1) / run_iterations
    print("MSP fpr95, auroc:  {}\t{}".format(fpr_tpr95[1] * 100, auroc[1]))
    print("Ours fpr95, auroc:  {}\t{}".format(fpr_tpr95[2] * 100, auroc[2]))
