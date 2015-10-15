import csv
import numpy as np

def parse_horizon(infile):
    """
    Reads a horizon file output from openDetect. Contains inline, xline,
    and offsets
    """
    
    ils, xls, values = ([],[],[])
    with open(infile, 'r') as f:
        reader=csv.reader(f, delimiter='\t')
        for il, xl,z, zero, five, ten, ft, twen, tf, t, tf in reader:
            ils.append(int(il)-1)
            xls.append(int(xl)-1)
            point = [float(i) for i in[zero, five, ten, ft, twen, tf, t, tf]]
            values.append(point)
    
    value_array = np.array(values)
    inlines = np.array(ils)
    xlines = np.array(xls)
    horizons = np.zeros((np.amax(inlines)-np.amin(inlines) +1, 
                         np.amax(xlines) - np.amin(xlines)+1, value_array.shape[1]))
    horizons[inlines-np.amin(inlines), xlines - np.amin(xlines), :] += values
    
    return horizons

def horizon_norm(horizon):
    """
    Normalize a horizon to unit energy across the offset dimension. Filters out
    zero energy and offset curves with NaNs.
    """
    
    normed = np.nan_to_num(horizon / np.sqrt(np.sum(horizon**2, 2))[:,:, np.newaxis])
    normed = normed.reshape(normed.shape[0]*normed.shape[1], normed.shape[2])
    
    normed = normed[np.sum(normed,1) > 0]
    
    return normed
