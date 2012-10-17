

import numpy as np
import scipy.stats as stats



def threshold_by_pvalue(data, alpha, two_tail=True):
    # may or may not work if data is multidimensional. probably not.
    
    zdata = stats.zscore(data)
    thresholded_data = np.zeros(np.shape(data))
    
    if not two_tail:
        zthreshold = stats.norm.isf(alpha)
        thresholded_data[zdata >= zthreshold] = data[zdata >= zthreshold]
        
    else:
        zupper = stats.norm.isf(alpha)
        zlower = stats.norm.isf(1.-alpha)
        thresholded_data[zdata >= zupper] = data[zdata >= zupper]
        thresholded_data[zdata <= zlower] = data[zdata <= zlower]
        
    return thresholded_data


def threshold_by_rawrange(data, percentage, two_tail=True):
    # probably not a good option if you are expecting outliers in the data...
    
    #compute the minimum and maximum value
    data_min = min(data)
    data_max = max(data)
    data_range = data_max - data_min
    upper_percent = data_max - (percentage*data_range)
    lower_percent = data_min + (percentage*data_range)
    
    thresholded_data = np.zeros(np.shape(data))

    if not two_tail:
        thresholded_data[data >= upper_percent] = data[data >= upper_percent]
        
    else:
        thresholded_data[data >= upper_percent] = data[data >= upper_percent]
        thresholded_data[data <= lower_percent] = data[data <= lower_percent]
        
    return thresholded_data
        