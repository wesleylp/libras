import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_pickle(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def open_pickle(filename):
    with open(filename, 'rb') as f:
        p = pickle.load(f)

    return p


def df_results(opt):
    df = pd.DataFrame(opt.cv_results_['params'])
    # df.rename(columns = {0:'param_model'}, inplace = True)

    df_mean = pd.DataFrame(opt.cv_results_['mean_test_score'])
    df_std = pd.DataFrame(opt.cv_results_['std_test_score'])
    df_rank = pd.DataFrame(opt.cv_results_['rank_test_score'])

    df = df.join(df_mean)
    df.rename(columns={0: 'mean_test_score'}, inplace=True)

    df = df.join(df_std)
    df.rename(columns={0: 'std_test_score'}, inplace=True)

    df = df.join(df_rank)
    df.rename(columns={0: 'rank'}, inplace=True)

    df.sort_values(by='mean_test_score', inplace=True, ascending=False)

    return df


def normalize_cfn_mtx(mtx, by='true'):
    if by.lower() == 'true':
        norm_factor = mtx.sum(axis=1)[:, np.newaxis]

    elif by.lower() == 'pred':
        norm_factor = mtx.sum(axis=0)

    elif by.lower() == 'all':
        norm_factor = mtx.sum()

    else:
        ValueError(f'`by` is invalid must be `true`, `pred`, or `all`: {by}')

    return mtx / norm_factor


def compute_mean_mtx(cfn_mtx_dict, normalize=None):

    cfn_mean = np.zeros_like(list(cfn_mtx_dict.values())[0], dtype=float)
    all_diags = []

    for key, mtx in cfn_mtx_dict.items():
        if normalize is not None:
            mtx = normalize_cfn_mtx(mtx, by=normalize)

        cfn_mean += mtx
        all_diags.append(mtx.diagonal())

    cfn_mean = cfn_mean / len(cfn_mtx_dict)

    return cfn_mean, all_diags


def multiclass_mcc(mtx):
    """Compute multiclass Mathews correlation coefficient from confusion matix. It assumes the the predict are in cols and true labes in the rows.

    Args:
        mtx (np.array): confusion matrix

    Returns:
        float: Mathews correlation coefficient
    """

    # assuming predicted in cols and true labels in rows
    # ref: https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-corrcoef

    t_k = mtx.sum(axis=1)  # the number of times class k truly occurred
    p_k = mtx.sum(axis=0)  # the number of times class k was predicted
    c = np.diag(mtx).sum()  # the total number of samples correctly predicted
    s = mtx.sum()  # the total number of samples.

    num = c * s - np.sum(np.multiply(p_k, t_k))
    den = np.sqrt((s**2 - np.sum(p_k**2)) * (s**2 - np.sum(t_k**2)))

    mcc = num / den

    return mcc


def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857
    By HYRY
    '''

    pc.update_scalarmappable()
    ax = pc.axes
    #ax = pc.axes# FOR LATEST MATPLOTLIB
    #Use zip BELOW IN PYTHON 3
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def heatmap(AUC,
            title,
            xlabel,
            ylabel,
            xticklabels,
            yticklabels,
            figure_width=40,
            figure_height=20,
            correct_orientation=False,
            cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC,
                  edgecolors='k',
                  linestyle='dashed',
                  linewidths=0.1,
                  cmap=cmap,
                  vmin=0.0,
                  vmax=1.0)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim((0, AUC.shape[1]))

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        #         t.tick1On = False
        #         t.tick2On = False
        t.tick1line.set_visible(False)
        t.tick2line.set_visible(False)
    for t in ax.yaxis.get_major_ticks():
        #         t.tick1On = False
        #         t.tick2On = False
        t.tick1line.set_visible(False)
        t.tick2line.set_visible(False)

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

    # resize
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))


def plot_classification_report(classification_report,
                               title='Classification report',
                               cmap='RdBu',
                               figure_width=25,
                               correct_orientation=False,
                               quiet=True):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857
    '''
    # lines = classification_report.split('\n')

    # classes = []
    # plotMat = []
    # support = []
    # class_names = []
    # for line in lines[2:(len(lines) - 4)]:
    #     t = line.strip().split()
    #     if len(t) < 2: continue
    #     classes.append(t[0])
    #     v = [float(x) for x in t[1:len(t) - 1]]
    #     support.append(int(t[-1]))
    #     class_names.append(t[0])

    #     if not quiet:
    #         print(v)

    #     plotMat.append(v)

    if isinstance(classification_report, str):
        classes, plotMat, support, class_names = read_classification_report(classification_report,
                                                                            quiet=quiet)
    elif isinstance(classification_report, dict):
        classes = classification_report['classes']
        plotMat = classification_report['plotMat']
        support = classification_report['support']
        class_names = classification_report['class_names']

    else:
        ValueError(f'classification_report must be `str` or `dict`: {type(classification_report)}')

    if not quiet:
        print('plotMat: {0}'.format(plotMat))
        print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    #     yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    yticklabels = ['{0}'.format(class_names[idx]) for idx, _ in enumerate(support)]
    #     figure_width = 25
    figure_height = len(class_names) + 7
    #     correct_orientation = False
    heatmap(np.array(plotMat),
            title,
            xlabel,
            ylabel,
            xticklabels,
            yticklabels,
            figure_width,
            figure_height,
            correct_orientation,
            cmap=cmap)


def read_classification_report(classification_report, quiet=True):
    lines = classification_report.split('\n')

    classes = []
    Mat = []
    support = []
    class_names = []
    for line in lines[2:(len(lines) - 4)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1:len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])

        if not quiet:
            print(v)

        Mat.append(v)

    return classes, Mat, support, class_names


def mean_classification_report(classification_reports):
    all_cls_reports = []

    for key, cls_report in classification_reports.items():
        classes, Mat, support, class_names = read_classification_report(classification_reports[key])
        all_cls_reports.append(Mat)

    Mat_mean = np.mean(all_cls_reports, axis=0)
    Mat_std = np.std(all_cls_reports, axis=0)

    return Mat_mean, Mat_std
