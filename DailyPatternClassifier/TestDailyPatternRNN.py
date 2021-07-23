# Adam Patyk
# Clemson University
# MS Thesis: Daily Pattern Classifier
# Summer 2021

# TestDailyPatternRNN.py
# Purpose: Evaluate time and episode metrics of daily pattern classifier for k-fold cross validation
# Usage: python TestDailyPatternRNN.py <threshold_val_start> <threshold_val_end> <threshold_val_step> <num_epochs>

import sys
import os
import numpy as np
import pandas as pd
import sklearn.metrics
from tqdm import tqdm
from datetime import datetime

sys.path.append('../') # for .py files in ../common/
import common.testing as testing

if len(sys.argv) != 5:
    sys.exit("Usage: python TestDailyPatternRNN.py <threshold_start> <threshold_end> <threshold_step> <num_epochs>")  

thresholds = np.arange(float(sys.argv[1]), float(sys.argv[2]),float(sys.argv[3]))
k = 5
epochs = int(sys.argv[4])

results = []
start_time = datetime.now()

for T in thresholds:
    print(f'T = {T}')
    total_TPR, total_TNR, total_F1, total_Prec, total_WAcc = [], [], [], [], []
    total_ep_TPR, total_ep_F1, total_ep_FP_TP = [], [], []

    for f in range(k):
        print(f'Fold {f+1}', flush=True)
        # read saved data from DailyPatternRNN scripts
        testing_sample_lengths = np.load(f'testing/testing_lengths_{epochs}epochs_fold{f+1}.npy')
        testing_probs = np.load(f'testing/testing_probs_{epochs}epochs_fold{f+1}.npy')
        testing_labels = np.load(f'testing/testing_labels_{epochs}epochs_fold{f+1}.npy')
        
        total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0
        total_ep_TP, total_ep_FP, total_ep_FN = 0, 0, 0

        # get episode metrics on testing dataset
        for i in tqdm(range(len(testing_labels))):
            probs = testing_probs[i,:testing_sample_lengths[i]]
            gt_labels = testing_labels[i,:testing_sample_lengths[i]]
            # thresholding segmentation
            h_results = testing.single_threshold(probs, gt_labels, winmin=6, stepsec=100, threshold=T)
            # time-based metrics
            TN, FP, FN, TP = sklearn.metrics.confusion_matrix(gt_labels, h_results['predictions'][0], labels=[0,1]).ravel()
            total_TP += TP
            total_FP += FP
            total_TN += TN
            total_FN += FN
            # episode-based metrics
            ep_TP, ep_FP, ep_FN = testing.calc_episode_metrics(h_results, gt_labels)
            total_ep_TP += ep_TP
            total_ep_FP += ep_FP
            total_ep_FN += ep_FN

        # calculate and report overall metrics
        TPR = testing.true_positive_rate(total_TP, total_FN)
        TNR = testing.true_negative_rate(total_TN, total_FP)
        F1 = testing.f1_score(total_TP, total_FP, total_FN)
        Prec = testing.precision(total_TP, total_FP)
        WAcc = testing.weighted_accuracy(total_TP, total_FP, total_TN, total_FN)
        
        ep_TPR = testing.true_positive_rate(total_ep_TP, total_ep_FN)
        ep_F1 = testing.f1_score(total_ep_TP, total_ep_FP, total_ep_FN)
        ep_FP_TP = -1 if total_ep_TP == 0 else total_ep_FP / total_ep_TP

        total_TPR.append(TPR)
        total_TNR.append(TNR)
        total_F1.append(F1)
        total_Prec.append(Prec)
        total_WAcc.append(WAcc)
        total_ep_TPR.append(ep_TPR)
        total_ep_F1.append(ep_F1)
        total_ep_FP_TP.append(ep_FP_TP)
        
    T_results = {'WAcc': np.mean(total_WAcc), 'TPR': np.mean(total_TPR), 'TNR': np.mean(total_TNR), 'F1': np.mean(total_F1), 'Precision': np.mean(total_Prec), 
                 'Episode TPR': np.mean(total_ep_TPR), 'Episode F1': np.mean(total_ep_F1), 'Episode FP/TP': np.mean(total_ep_FP_TP)}
    results.append(T_results)

    print('AVERAGE:')
    print('--- Time Metrics ---')
    print(f'WAcc: {np.mean(total_WAcc):.3f}\tTPR: {np.mean(total_TPR):.3f}\tTNR: {np.mean(total_TNR):.3f}\tF1: {np.mean(total_F1):.3f}\tPrecision: {np.mean(total_Prec):.3f}')

    print('--- Episode Metrics ---')
    print(f'TPR: {np.mean(total_ep_TPR):.3f}\tF1: {np.mean(total_ep_F1):.3f}\tFP/TP: {np.mean(total_ep_FP_TP):.3f}')
    print("*****************************************************************", flush=True)
    
# prepare .csv file for export
os.makedirs('results', exist_ok=True)
results_df = pd.DataFrame(results)
results_df.insert(0, 'Threshold', thresholds)
results_df.to_csv(f'results/testing-results-{epochs}epochs.csv', index=False, header=True)

print('Results saved.')

end_time = datetime.now()
print(f'Duration: {end_time - start_time}')