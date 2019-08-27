# Author:  DINDIN Meryll
# Date:    02/03/2019
# Project: optimizers

try: from optimizers.imports import *
except: from imports import *

# Cast float to integers when needed

def handle_integers(params):

    new_params = {}

    for k, v in params.items():
        if type(v) == float and int(v) == v: new_params[k] = int(v)
        else: new_params[k] = v
    
    return new_params

# Defines the mean percentage error as metric

def mean_percentage_error(true, pred):

    msk = true != 0.0
    
    return np.nanmean(np.abs(true[msk] - pred[msk]) / true[msk])

# Transform a dataframe to an image

def dtf_to_img(dtf, row_height=0.8, font_size=10, ax=None):

    # Basic needed attributes
    header_color, row_colors, edge_color = '#40466e', ['#f1f1f2', 'w'], 'w'
    bbox, header_columns = [0, 0, 1, 1], 0

    if ax is None:
        size = (18, dtf.shape[0]*row_height)
        fig, ax = plt.subplots(figsize=(size))
    
    ax.axis('off')

    mpl_table = ax.table(cellText=dtf.values, bbox=bbox, colLabels=dtf.columns, cellLoc='center')
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])

    return ax
