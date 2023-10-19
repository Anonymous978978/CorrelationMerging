# Correlation-Merging
# Toy Example
""" This part is related to _Section 3_ and _Figure 2_ of our paper. """

To run toy example, you should firstly go to the "Toy Example" directory, then

    python main_toy.py
- The output is the performance (AUROC, FPR95) of MSP and our method, from m=0 to m=99 (correlation intensity increase).

The code of our method see "Toy Example/main_toy.py" line 248-269. 


# OOD Detection

## For QA: 
""" This part is related to _Table 1_, _Table 3_, and _Figure 3_ of our paper. """

In this part, we reuse code from https://github.com/xiye17/InterpCalib

In this experiment, the grouping strategy is Semantic Grouping.

To run experiments of OOD detection on QA, you should firstly go to the "Question Answering" directory, then

    python OOD_detection.py 
    --scoring_function [msp, sl, sp] 
    --id_dataset [squad, hotpot] 
    --ood_dataset [squad, trivia, hotpot, searchqa, textbookqa]
    
    # e.g.
    python OOD_detection.py --scoring_function msp --id_dataset squad --ood_dataset hotpot
- The output is OOD detection performance (AUROC, FPR95) of selected method in selected ID-OOD setting. The base model is RoBERTa.

- msp, sl, sp represents MSP, CoMe-SE, and CoMe-SP respectively. 

The code of our method see "Question Answering/utils.py" function post_process() line 487-555. 

The path is main() line 795 -> evaluate_show_order() line 790 -> nbest_post_process() line 613 -> post_process() line 391.

## For Text Classification: 
""" This part is related to _Figure 4_, _Table 4_, and _Table 5_ of our paper. """

To run experiments of OOD detection on text classification, you should firstly go to the "Text Classification" directory, then
### KNN-C

In this part, we reuse code from https://github.com/zyh190507/KnnContrastiveForOOD

Go to the "KNN-C" directory

    python run_main.py 
    --group_strategy [confusion, correlation, taxonomy]
    --scoring_function [msp, sl, sp]
    [/--use_roberta]
    
    # e.g.
    python run_main.py --group_strategy correlation --scoring_function msp --use_roberta
- The output is OOD detection performance (AUROC, FPR95) of selected method in selected base model.
    
- confusion, correlation, taxonomy means three grouping strategies: Confusion Grouping, Correlation Grouping, and Taxonomy.

- msp, sl, sp represents MSP, CoMe-SE, and CoMe-SP respectively. 

- If you want to use the RoBERTa-base model you should add --use_roberta, else the base model will be BERT-base-uncased.

The code of our method see "Text Classification/KNN-C/run_main.py", Grouping strategies in line 724-745, Scoring functions in line 848-912. 

The path is main() line 735 -> myevaluation() line 725.

### Vallian and SCL-C/L

In this part, we reuse code from https://github.com/parZival27/supervised-contrastive-learning-for-out-of-domain-detection

Go to the "SCL" directory

    python train.py 
    --model_dir ["CLINC_OOD-100-vallian-scl_lmcl", "CLINC_OOD-100-vallian-scl_ce", "CLINC_OOD-100-vallian"]   
    --group_strategy [confusion, correlation, taxonomy] 
    --scoring_function [msp, sl, sp] 
    [/--use_roberta]
    
    # e.g.
    python train.py --model_dir "CLINC_OOD-100-vallian-scl_lmcl" --group_strategy confusion --scoring_function msp --use_roberta
- The output is OOD detection performance (AUROC, FPR95) of selected method in selected base model and training strategy.

- "CLINC_OOD-100-vallian-scl_lmcl", "CLINC_OOD-100-vallian-scl_ce", "CLINC_OOD-100-vallian" represents 3 different strategies: SCL-L, SCL-C, CE.

- confusion, correlation, taxonomy means three grouping strategies: Confusion Grouping, Correlation Grouping, and Taxonomy.

- msp, sl, sp represents MSP, CoMe-SE, and CoMe-SP respectively. 

- If you want to use the RoBERTa-base model you should add --use_roberta, else the base model will be BERT-base-uncased

The code of our method see "Text Classification/SCL/train.py", Grouping strategies in line 814-837, Scoring functions in line 913-975. 


# Appendix
## For Image Classification
""" This part is related to _Appendix B_ and _Table 6_ of our paper. """

In this part, we reuse code from https://github.com/VectorInstitute/gram-ood-detection.

In image classifiaction, the ID dataset is CIFAR-100. OOD datasets includes SVHN, iSUN, LSUN(C), LSUN(R), TinyImageNet(C), and TinyImageNet(R)

The grouping strategy is Taxonomy.

To run experiments of OOD detection on image classification, you should firstly go to the "Image Classification" directory, then

### For DenseNet:

    python main_densenet.py
    --method [sp, sl, msp]
    
    # e.g.
    python main_densenet.py --method sl
    
### For ResNet:

    python main_resnet.py
    --method [sp, sl, msp]
    
    # e.g.
    python main_resnet.py --method sl
- The output is OOD detection performance (AUROC, FPR95) of selected method in selected base model. 

- msp, sl, sp represents MSP, CoMe-SE, and CoMe-SP respectively. 

The code of our method see "Image Classification/main_densenet.py" line 636-667. 

The code of our method see "Image Classification/main_resnet.py" line 638-669. 
# Calibration
""" This part is related to _Appendix C_ and _Figure 4_ of our paper. """

To run experiments of Calibration on QA, you should firstly go to the "Question Answering" directory, then

    python Calibration.py 
    --scoring_function [msp, sp] 
    --id_dataset [squad, hotpot, trivia]  
    [/--ts]
    
    # e.g.
    python Calibration.py --scoring_function msp --id_dataset squad --ts

- The output is calibration performance of those methods in three qa datasets: Squad, HotpotQA, and TriviaQA.

- msp, sp represents MSP and CoMe-SP respectively. 

- if you want to use Temperature Scaling you should add --ts.

The code of our method see "Question Answering/utils.py" function post_process() line 487-555. 

The path is main() line 781 -> evaluate_show_order() line 775 -> nbest_post_process() line 567 -> post_process() line 371.

We will supplement all citations and license after the anonymity period.
