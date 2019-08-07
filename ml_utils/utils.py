# Author:  DINDIN Meryll
# Date:    02/03/2019
# Project: ml_utils


# Defines the weights corresponding to a given array of labels

def sample_weight(lab) :

    # Defines the sample_weight
    res = np.zeros(len(lab))
    wei = compute_class_weight('balanced', np.unique(lab), lab)
    wei = wei / sum(wei)
    
    for ele in np.unique(lab) :
        for idx in np.where(lab == ele)[0] :
            res[idx] = wei[int(ele)]

    del wei

    return res

# Defines a dictionnary composed of class weights

def class_weight(lab) :
    
    res = dict()
    
    for idx, ele in enumerate(compute_class_weight('balanced', np.unique(lab), lab)) : res[idx] = ele
        
    return res
