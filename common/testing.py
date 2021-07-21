import numpy as np
import pandas as pd
import tensorflow as tf

def get_raw_data(samples, labels, data):
    axdata = []
    aydata = []

    for i in range(len(labels)):
        f = samples[i][0]
        t1 = samples[i][1]
        t2 = samples[i][2]
        sample = data[f][t1:t2]
        label = labels[i]
        axdata.append(sample)
        aydata.append(label)

    rawsamples = np.array(axdata, copy=True)
    rawlabels = np.array(aydata, copy=True)
    del axdata
    del aydata
    
    return rawsamples, rawlabels

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def get_raw_samples(samples, data):
    axdata = []

    for i in range(len(samples)):
        f = samples[i][0]
        t1 = samples[i][1]
        t2 = samples[i][2]
        sample = data[f][t1:t2]
        axdata.append(sample)

    rawsamples = np.array(axdata, copy=True)
    del axdata
    
    return rawsamples

def weighted_accuracy(TP, FP, TN, FN):
    P = TP + FN
    N = TN + FP
    if P > 0:
        W = N/P
    else: 
        W = 1
    WAcc = (W * TP + TN) / (W * P + N)
    return WAcc

def true_positive_rate(TP, FN):
    if TP + FN == 0:
        tpr = 0
    else:
        tpr = TP / (TP +  FN)
    return tpr

def true_negative_rate(TN, FP):
    if TN + FP == 0:
        tnr = 0
    else:
        tnr = TN / (TN + FP)
    return tnr

def f1_score(TP, FP, FN):
    if TP == 0 and FP == 0 and FN == 0:
        f1 = 0
    else:
        f1 = TP / (TP + 0.5 * (FP + FN))
    return f1

def precision(TP, FP):
    if TP + FP == 0:
        p = 0
    else:
        p = TP / (TP + FP)
    return p

def consecutive_groups(data, stepsize=1):
    segments = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    bookends = []
    for segment in segments:
        bookends.append((segment[0], segment[-1]))

    return bookends

def hysteresis_threshold(model, samples, labels, start_threshold, end_threshold, winmin, stepsec, episode_min=1.):
    """
    model: tensorflow model
    samples:   samples of raw data
    labels:    labels for raw data
    start_threshold: high threshold of the beginning of segmentation
    end_threshold: low threshold of the end of segmentation
    winmin: size of a window sample in unit of  minute
    stepsec: stride to move the window in unit of second / the number of second between two adjacent window samples
    episode_min: the minimum length of eating episode in unit of minute. If end of segmentation -start of segmentation < episode_min,
        then the episode will not be counted
    """
    import pandas as pd
    result_ls = []
    t_pause = winmin / 2 * 60

    result = {'segment_start':[], 'segment_end':[], 'prob':[], 'predictions':np.zeros([len(labels)], dtype=int), 'segment_count':0}
    
    probs = tf.squeeze(model.predict(samples, batch_size=4096))
    samples = tf.squeeze(samples)
    state, start, end = 0, 0, 0
    pause_counter = 0
    for i in range(len(labels)):
        prob = probs[i].numpy()
        result['prob'].append(prob)

        if state == 0 and prob > start_threshold:
            state = 1
            start = i
        elif state == 1 and prob < end_threshold:
            state = 2
            end = i+1 # for Python list slicing
            pause_counter = 0
        elif state == 2:
            if prob > start_threshold:
                state = 1
            else:
                pause_counter += stepsec
                if pause_counter >= t_pause:
                    # convert time to second and check threshold
                    if (end-start)*stepsec >= episode_min*60:
                        # save data
                        result['segment_start'].append(start)
                        result['segment_end'].append(end)
                        result['segment_count'] += 1
                        result['predictions'][start:end] = 1
                        pass
                    end = 0
                    state = 0
    if state == 1: # catch meal if it ends at the end of probabilities
        end = i
        if pause_counter >= t_pause and (end-start)*stepsec >= episode_min*60:
            # save data
            result['segment_start'].append(start)
            result['segment_end'].append(end)
            result['segment_count'] += 1
            result['predictions'][start:end] = 1
        
    result_ls.append(result)
                            
    return pd.DataFrame(result_ls)

def hysteresis_threshold_probs(probs, labels, start_threshold, end_threshold, winmin, stepsec, episode_min=1.):
    """
    probs:   model output probabilities from samples
    labels:    labels for raw data
    start_threshold: high threshold of the beginning of segmentation
    end_threshold: low threshold of the end of segmentation
    winmin: size of a window sample in unit of  minute
    stepsec: stride to move the window in unit of second / the number of second between two adjacent window samples
    episode_min: the minimum length of eating episode in unit of minute. If end of segmentation -start of segmentation < episode_min,
        then the episode will not be counted
    """
    import pandas as pd
    result_ls = []
    t_pause = winmin / 2 * 60

    result = {'segment_start':[], 'segment_end':[], 'prob':[], 'predictions':np.zeros([len(labels)], dtype=int), 'segment_count':0}
    
    state, start, end = 0, 0, 0
    pause_counter = 0
    for i in range(len(labels)):
        prob = probs[i]
        result['prob'].append(prob)

        if state == 0 and prob > start_threshold:
            state = 1
            start = i
        elif state == 1 and prob < end_threshold:
            state = 2
            end = i+1 # for Python list slicing
            pause_counter = 0
        elif state == 2:
            if prob > start_threshold:
                state = 1
            else:
                pause_counter += stepsec
                if pause_counter >= t_pause:
                    # convert time to second and check threshold
                    if (end-start)*stepsec >= episode_min*60:
                        # save data
                        result['segment_start'].append(start)
                        result['segment_end'].append(end)
                        result['segment_count'] += 1
                        result['predictions'][start:end] = 1
                        pass
                    end = 0
                    state = 0
    if state == 1: # catch meal if it ends at the end of probabilities
        end = i
        if pause_counter >= t_pause and (end-start)*stepsec >= episode_min*60:
            # save data
            result['segment_start'].append(start)
            result['segment_end'].append(end)
            result['segment_count'] += 1
            result['predictions'][start:end] = 1
        
    result_ls.append(result)
                            
    return pd.DataFrame(result_ls)

def single_threshold(probs, labels, winmin, stepsec, threshold=0.5, episode_min=1.):
    """
    probs:   model output probabilities from samples
    labels:    labels for raw data
    """
    import pandas as pd
    result_ls = []
    t_pause = winmin / 2 * 60

    result = {'segment_start':[], 'segment_end':[], 'prob':[], 'predictions':np.zeros([len(labels)], dtype=int), 'segment_count':0}
    
    state, start, end = 0, 0, 0
    pause_counter = 0
    for i in range(len(labels)):
        prob = probs[i]
        result['prob'].append(prob)

        if state == 0 and prob > threshold:
            state = 1
            start = i
        elif state == 1 and prob < threshold:
            state = 2
            end = i+1 # for Python list slicing
            pause_counter = 0
        elif state == 2:
            if prob > threshold:
                state = 1
            else:
                pause_counter += stepsec
                if pause_counter >= t_pause:
                    # convert time to second and check threshold
                    if (end-start)*stepsec >= episode_min*60:
                        # save data
                        result['segment_start'].append(start)
                        result['segment_end'].append(end)
                        result['segment_count'] += 1
                        result['predictions'][start:end] = 1
                        pass
                    end = 0
                    state = 0
    if state == 1: # catch meal if it ends at the end of probabilities
        end = i
        if pause_counter >= t_pause and (end-start)*stepsec >= episode_min*60:
            # save data
            result['segment_start'].append(start)
            result['segment_end'].append(end)
            result['segment_count'] += 1
            result['predictions'][start:end] = 1
        
    result_ls.append(result)
                            
    return pd.DataFrame(result_ls)

def calc_episode_metrics(results, labels):
    """
    results:    pandas dataframe output by hysteresis_threshold()
    labels:     GT labels for raw data
    """
    TP, FP, FN = 0, 0, 0

    gt_indices = np.where(labels == 1)
    if np.size(gt_indices) != 0:
        gt_segments = consecutive_groups(gt_indices[0])
    else:
        gt_segments = []
    eating_segments = list(zip(results['segment_start'][0], [x-1 for x in results['segment_end'][0]])) # to account for Python list slicing
    
    GTEval = [-1] * len(gt_segments)
    MDEval = [-1] * len(eating_segments)

    # TP - GT event, model event (any overlap) - 1
    # FN - GT event, missed by model - 2
    # FP - no event, model event - 3

    # look for matches with GT events
    for i, (gt_start, gt_end) in enumerate(gt_segments):
        for e, (md_start, md_end) in enumerate(eating_segments):
            # (1) MD within GT
            # (2) MD starts before GT and ends in GT
            # (3) MD starts in GT and ends after GT
            # (4) MD contains GT
            if  (md_start >= gt_start and md_end <= gt_end) or \
                (md_start <= gt_start and md_end > gt_start and md_end <= gt_end) or \
                (md_start >= gt_start and md_start < gt_end and md_end >= gt_end) or \
                (md_start <= gt_start and md_end >= gt_end): 
                GTEval[i] = e
                MDEval[e] = i
                
    # count up classifications
    for i in range(len(gt_segments)):
        if GTEval[i] == -1:
            FN += 1
        else:
            TP += 1
    for e in range(len(eating_segments)):
        if MDEval[e] == -1:
            FP += 1

    return TP, FP, FN

def calc_time_metrics(MD, GT):
    """
    MD: array of model detection 1s and 0s to signify eating and non-eating
    GT: array of GT 1s and 0s to signify eating and non-eating
    """
    TP, FP, TN, FN = 0, 0, 0, 0
        
    # Count TP, FP, TN, FN
    for i in range(len(GT)):
        if MD[i] == 1:
            if GT[i] == 1: 
                TP += 1
            else:
                FP += 1
        else:
            if GT[i] == 0:
                TN += 1
            else:
                FN += 1
            
    return TP, FP, TN, FN
