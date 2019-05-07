import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.multiclass import unique_labels


def read_data(filename='trace.csv'):

    # Convert PC address to decimal and Branch to [0, 1]
    df = pd.read_csv(filename, header=None, names=['PC', 'Branch'],
                     converters={'PC': lambda x: int(x, 16),
                                 'Branch': lambda x: 1 if x == 'T' else 0
                                 })

    # Create feature for difference between conditional branches
    df['Diff'] = df['PC'].diff()
    df.fillna(0, inplace=True)

    return df.to_dict(orient='list')


def parse_trace(filename='gcc-10M.trace'):
    """
    Parse a given x86 trace file to
    PC address and branch taken/not taken
    """

    f_in = open(filename, 'r')
    f_out = open('trace.csv', 'w')

    for line in tqdm(f_in):

        # Separate by empty space
        line = line.split()

        # Only need conditional branches (flags=='R' and branch != '-')
        if line[5] == 'R' and line[6] != '-':
            f_out.write('0x{:s},{:s}\n'.format(line[1], line[6]))

    f_in.close()
    f_out.close()


def evaluate(y_true, y_pred, name='', normalize=False):
    """ Compute metrics between predicted and true labels """

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    metrics = {
            'Accuracy': np.float16(acc),
            'Precision': np.float16(prec),
            'Recall':np.float16(rec),
            'F1-Score': np.float16(f1)
            }
    plot_confusion_matrix(y_true, y_pred, classes=['Not Taken', 'Taken'],
                          normalize=normalize, title=name + ' Predictor')
    
    return metrics


# Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
# Modified for python 3.7
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title='',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    classes = [classes[i] for i in unique_labels(y_true, y_pred)]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()