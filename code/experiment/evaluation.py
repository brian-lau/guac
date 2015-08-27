import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from ..preprocessing import data_splitting as ds, labels


def calc_f1_and_acc_for_column(true, predicted):
    f1 = f1_score(true, predicted)
    acc = accuracy_score(true, predicted)
    return f1, acc

def calc_task_wise_f1s(true, predicted, name):
    tasks = true.columns
    n_tasks = len(tasks)
    index = [name + '_f1s']
    results = pd.DataFrame(np.zeros([1, n_tasks]), index=index, columns=tasks)

    for task in tasks:
        results.loc[index[0], task] = f1_score(true.loc[:, task], predicted.loc[:, task])

    return results

def calc_task_wise_acc(true, predicted, name):
    tasks = true.columns
    n_tasks = len(tasks)
    index = [name + '_acc']
    results = pd.DataFrame(np.zeros([1, n_tasks]), index=index, columns=tasks)

    for task in tasks:
        results.loc[index[0], task] = accuracy_score(true.loc[:, task], predicted.loc[:, task])

    return results


def calc_micro_mean_f1_acc(true, predicted):
    n_items, n_tasks = true.shape

    y_vector = np.reshape(true.values, [n_items*n_tasks, 1])
    p_vector = np.reshape(predicted.values, [n_items*n_tasks, 1])

    f1 = f1_score(y_vector, p_vector)
    acc = accuracy_score(y_vector, p_vector)

    return f1, acc


def calc_macro_mean_f1_pp(true, predicted):
    n_items, n_tasks = true.shape
    f1s = pd.DataFrame(np.zeros([n_items, 1]), index=true.index)

    n_perfect = 0
    for index, rid in enumerate(true.index):
        f1s.loc[rid] = f1_score(true.loc[rid], predicted.loc[rid])
        if np.sum(np.abs(true.loc[rid] - predicted.loc[rid])) == 0:
            n_perfect += 1

    percent_perfect = n_perfect / float(n_items)

    return np.mean(f1s.values), percent_perfect


def get_report_header(dataset, test_fold, dev_subfold):
    training_items = ds.get_train_documents(dataset, test_fold, dev_subfold)
    dev_items = ds.get_dev_documents(dataset, test_fold, dev_subfold)
    multi_y = labels.get_labels(dataset)

    n_items, n_tasks = multi_y.shape

    index = ['nTrain', 'nTest']
    results = pd.DataFrame(np.zeros([2, n_tasks]), index=index, columns=multi_y.columns)
    results.loc['nTrain'] = multi_y.loc[training_items].sum(axis=0)
    results.loc['nTest'] = multi_y.loc[dev_items].sum(axis=0)

    return results

def get_summary_report_header(datasets, test_fold, dev_subfold):
    n_datasets = len(datasets)
    index = ['testFold', 'devSubFold', 'nTrain', 'nTest']
    header = pd.DataFrame(np.zeros([4, n_datasets]), index=index, columns=datasets)
    header.loc['testFold', :] = test_fold
    header.loc['devSubFold', :] = dev_subfold
    for f in datasets:
        training_items = ds.get_train_documents(f, test_fold, dev_subfold)
        header.loc['nTrain', f] = len(training_items)

        dev_items = ds.get_dev_documents(f, test_fold, dev_subfold)
        header.loc['nTest', f] = len(dev_items)

    return header

