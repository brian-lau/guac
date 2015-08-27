import re
import sys
from optparse import OptionParser

from hyperopt import STATUS_OK
import numpy as np

import pandas as pd
from scipy import stats
from scipy import sparse

from ..models.sparse_model import SparseModel
from ..experiment import evaluation
from ..util import file_handling as fh, defines
from ..feature_extractors import feature_loader
from ..preprocessing import data_splitting as ds, labels


def main():
    # Handle input options and arguments
    usage = "%prog [feature1, feature2, ...] "
    parser = OptionParser(usage=usage)
    parser.add_option('-n', dest='name', default='',
                      help='Experiment name; (randomly assigned by default)')
    parser.add_option('-t', dest='test_fold', default=0,
                      help='Test fold; default=%default')
    parser.add_option('-g', dest='group_file', default='',
                      help='List of questions to group together (one group per line)')
    parser.add_option('-r', dest='regularizer', default='l1',
                      help='Regularization (l1 or l2): default=%default')
    parser.add_option('-m', dest='model_type', default='LR',
                      help='Model type (LR|SVM|MNB|SVMNB); default=%default')
    parser.add_option('-v', dest='verbose', default=1,
                      help='Level of verbosity; default=%default')
    parser.add_option('--reuse', dest='reuse', action="store_true", default=False,
                      help='Use reusable holdout; default=%default')
    parser.add_option('--reuse_T', dest='reusable_T', default=0.04,
                      help='T for reusable holdout; default=%default')
    parser.add_option('--reuse_tau', dest='reusable_tau', default=0.01,
                      help='tau for reusable holdout; default=%default')

    (options, args) = parser.parse_args()

    if len(args) == 0:
        sys.exit("Please specify at least one feature definition (e.g. ngrams,n=1,binarize=True [with no spaces])")
    feature_list = args
    test_fold = int(options.test_fold)
    group_file = options.group_file

    groups = get_groups(group_file)

    if options.name == '':
        #name = 'e' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        name = '_'.join(feature_list)
    else:
        name = options.name

    regularizer = options.regularizer
    model_type = options.model_type
    verbose = options.verbose
    reuse_holdout = options.reuse
    reuseable_T = options.reusable_T
    reuseable_tau = options.reusable_tau

    input_dir = defines.data_raw_csv_dir
    files = fh.ls(input_dir, '*.csv')
    files.sort()

    # process all the (given) data files
    print "Running experiments"
    for group in groups:
        #valid_macro_f1, test_macro_f1 = run_group_experiment(name, group, test_fold, feature_list, model_type,
        #    regularizer, reuse=reuse_holdout, orig_T=reuseable_T, tau=reuseable_tau, verbose=verbose)
        #print group, valid_macro_f1, test_macro_f1
        run_group_experiment(name, group, test_fold, feature_list, model_type,
                             regularizer, min_alpha_exp=0, max_alpha_exp=0,
                             reuse=reuse_holdout, orig_T=reuseable_T, tau=reuseable_tau, verbose=1)



def write_log(exp_dir, names_list, values_list):
    output_filename = fh.make_filename(exp_dir, 'log', 'json')
    names = names_list[1:-1].split(',')
    summary = dict(zip(names, [str(v) for v in values_list]))
    fh.write_to_json(summary, output_filename)

# Run an experiment on a group of questions
@profile
def run_group_experiment(name, datasets, test_fold, feature_list, model_type='LR', regularization='l1',
                         min_alpha_exp=-1, max_alpha_exp=8, alpha_exp_base=np.sqrt(10), beta=1,
                         reuse=False, orig_T=0.04, tau=0.01, verbose=1,
                         best_alphas=None, svm_beta=0):

    # create experiments directory
    exp_dir = make_exp_dir(datasets, test_fold, name)
    params_list = [exp_dir, name, datasets, test_fold, feature_list, model_type, regularization,
                   min_alpha_exp, max_alpha_exp, alpha_exp_base, reuse, orig_T, tau, best_alphas]
    params_names =  '[exp_dir, name, datasets, test_fold, feature_list, model_type, regularization,\
        min_alpha_exp, max_alpha_exp, alpha_exp_base, reuse, orig_T, tau, best_alphas]'
    write_log(exp_dir, params_names, params_list)

    T_hat = orig_T
    if reuse:
        gamma = stats.laplace.rvs(scale=4*tau/np.sqrt(2))
        T_hat = orig_T + gamma

    ys = []
    codes = None
    for f in datasets:
        # load all labels for this dataset
        multi_y = labels.get_labels(f)
        ys.append(multi_y)
        if codes is None:
            codes = multi_y.columns
        else:
            assert (codes == multi_y.columns).all()
    y = pd.concat(ys, axis=0)

    #summary_per_code = create_summary_dfs_per_code(datasets, codes, y, test_fold, np.arange(len(codes)))

    # build an index of the items in the datasets for this group
    test_items = []
    index = []
    for dataset in datasets:
        test_items.extend(ds.get_test_documents(dataset, test_fold))
        index.extend(ds.get_all_documents(dataset))

    n_dev_folds = ds.get_n_dev_folds(datasets[0])

    if best_alphas is None:

        alpha_exponents = np.arange(min_alpha_exp, max_alpha_exp+1)
        alphas = alpha_exp_base**alpha_exponents

        train_f1s = {}
        valid_f1s = {}
        for code in codes:
            train_f1s[code] = pd.DataFrame(np.zeros([n_dev_folds, len(alphas)]),
                                           index=range(n_dev_folds), columns=[str(l) for l in alphas])
            valid_f1s[code] = pd.DataFrame(np.zeros([n_dev_folds, len(alphas)]),
                                           index=range(n_dev_folds), columns=[str(l) for l in alphas])

        # do cross validation on the dev_subfolds
        print "Doing cross validation"
        for dev_subfold in range(n_dev_folds):

            # get a list of training and validation items for this dev fold
            train_dict, valid_dict, test_dict = get_item_dicts(datasets, test_fold, dev_subfold)

            X, column_names = load_features(feature_list, test_fold, dev_subfold, index)
            if verbose > 0:
                n_train = np.sum([len(train_dict[f]) for f in datasets])
                print 'dev_fold =', str(dev_subfold), '; n_train =', n_train, '; n_features =', len(column_names)

            # train a best model for each data set based on the training and dev data
            cv_train_f1s, cv_valid_f1s = compare_r_strengths(datasets, X, index, column_names, y,
                                                             train_dict, valid_dict,
                                                             model_type, regularization=regularization,
                                                             alphas=alphas, beta=beta,
                                                             reuse=reuse, orig_T=orig_T, T_hat=T_hat, tau=tau)

            #print cv_train_f1s
            if verbose > 1:
                print cv_valid_f1s

            for code_index, code in enumerate(codes):
                train_f1s[code].loc[dev_subfold] = cv_train_f1s.loc[code]
                valid_f1s[code].loc[dev_subfold] = cv_valid_f1s.loc[code]

        # determine best value of the model parameter
        best_alphas = []
        for code in codes:
            best_alpha = valid_f1s[code].mean(axis=0).idxmax()
            best_alphas.append(float(best_alpha))
            if verbose > 1:
                print "Code:", code, " y sum:", y[code].sum(),  " best lambda:", best_alpha, \
                    " mean_f1_dev:", valid_f1s[code].loc[:, str(best_alpha)].mean(), \
                    " mean_f1_train:", train_f1s[code].loc[:, str(best_alpha)].mean()

    # re-run above, with best lambda, to get predictions (to estimate expected dev performance)
    print "Estimating hold-out performance"
    train_cv_summary = create_summary_dfs(datasets)
    valid_cv_summary = create_summary_dfs(datasets)
    masked_valid_cv_summary = create_summary_dfs(datasets)
    summary_per_code = create_summary_dfs_per_code(datasets, codes, y, test_fold, best_alphas)
    for dev_subfold in range(n_dev_folds):
        # get a list of training and validation items for this dev fold
        train_dict, valid_dict, test_dict = get_item_dicts(datasets, test_fold, dev_subfold)

        X, column_names = load_features(feature_list, test_fold, dev_subfold, index)
        if verbose > 0:
            n_train = np.sum([len(train_dict[f]) for f in datasets])
            print 'dev_fold =', str(dev_subfold), '; n_train =', n_train, '; n_features =', len(column_names)

        pred_train, pred_valid,\
        pred_train_prob, pred_valid_prob, models = train_and_predict(datasets, X, index, column_names, y, train_dict,
                                                                     valid_dict, model_type,
                                                                     regularization=regularization,
                                                                     alphas=best_alphas, beta=beta, verbose=verbose)

        evaluate_predictions(train_cv_summary, datasets, pred_train, y, train_dict, 'dev_fold_' + str(dev_subfold))
        evaluate_predictions(valid_cv_summary, datasets, pred_valid, y, valid_dict, 'dev_fold_' + str(dev_subfold))

        T_hat = evaluate_predictions(masked_valid_cv_summary, datasets, pred_valid, y, valid_dict,
                                     'dev_fold_' + str(dev_subfold),
                                     reuse=reuse, orig_T=orig_T, T_hat=T_hat, tau=tau, train_summary=train_cv_summary)

        evaluate_predictions_per_code(summary_per_code, datasets, pred_valid, y, valid_dict, 'f1_dev_fold_'
                                      + str(dev_subfold))

    print "Train CV summary:"
    print train_cv_summary['macrof1']
    print "Validation CV summary:"
    print valid_cv_summary['macrof1']
    print "Noisy validation CV summary:"
    print masked_valid_cv_summary['macrof1']
    write_summary_dfs_to_file(exp_dir, 'train_cv', train_cv_summary)
    write_summary_dfs_to_file(exp_dir, 'valid_cv', valid_cv_summary)
    write_summary_dfs_to_file(exp_dir, 'masked_valid_cv', masked_valid_cv_summary)

    # finally, train one full model and evaluate on the test data
    dev_subfold = None
    print "Training final model"
    train_summary = create_summary_dfs(datasets)
    test_summary = create_summary_dfs(datasets)
    train_dict, valid_dict, test_dict = get_item_dicts(datasets, test_fold, dev_subfold)
    X, column_names = load_features(feature_list, test_fold, dev_subfold, index)
    if verbose > 0:
        n_train = np.sum([len(train_dict[f]) for f in datasets])
        print 'dev_fold =', str(dev_subfold), '; n_train =', n_train, '; n_features =', len(column_names)

    pred_train, pred_test, pred_train_prob,\
    pred_test_prob, models = train_and_predict(datasets, X, index, column_names, y, train_dict, test_dict,
                                               model_type, regularization=regularization,
                                               alphas=best_alphas, beta=beta, verbose=verbose)

    for f in datasets:
        pred_train[f].to_csv(fh.make_filename(make_prediction_dir(exp_dir), f + '_' + 'train', 'csv'))
        pred_test[f].to_csv(fh.make_filename(make_prediction_dir(exp_dir), f + '_' + 'test', 'csv'))
        pred_train_prob[f].to_csv(fh.make_filename(make_prediction_dir(exp_dir), f + '_' + 'train_prob', 'csv'))
        pred_test_prob[f].to_csv(fh.make_filename(make_prediction_dir(exp_dir), f + '_' + 'test_prob', 'csv'))

    evaluate_predictions(train_summary, datasets, pred_train, y, train_dict, 'train')
    evaluate_predictions(test_summary,  datasets, pred_test,  y, test_dict,  'test')
    evaluate_predictions_per_code(summary_per_code, datasets, pred_train, y, train_dict, 'f1_train')
    evaluate_predictions_per_code(summary_per_code, datasets, pred_test, y, test_dict, 'f1_test')

    write_summary_dfs_to_file(exp_dir, 'train', train_summary)
    write_summary_dfs_to_file(exp_dir, 'test',  test_summary)
    for f in datasets:
        output_filename = fh.make_filename(make_results_dir(exp_dir), f + '_summary', 'csv')
        summary_per_code[f].to_csv(output_filename)

    for code in codes:
        output_filename = fh.make_filename(make_models_dir(exp_dir), re.sub(' ', '_', code), 'json')
        model = models[code]
        model.write_to_file(output_filename)

    print {'valid_f1': valid_cv_summary['macrof1']['overall'].mean(),
           'test_f1': test_summary['macrof1'].loc['test', 'overall']}
    return {'loss': -valid_cv_summary['macrof1']['overall'].mean(),
            'test_f1': test_summary['macrof1'].loc['test', 'overall'],
            'status': STATUS_OK
            }

def load_features(feature_list, test_fold, dev_subfold, items, verbose=1):
    # for each feature in feature_list:
    feature_matrices = []
    column_names = []
    for feature in feature_list:
        feature_description = feature + ',test_fold=' + str(test_fold)
        # TRYING THIS: JUST USE ALL THE ITEMS TO BUILD THE VOCAB...
        #if dev_subfold is not None:
        #    feature_description += ',dev_subfold=' + str(dev_subfold)
        counts, columns = feature_loader.load_feature(feature_description, items)
        if verbose > 1:
            print "Loaded", feature, "with shape", counts.shape
        feature_matrices.append(counts)
        column_names.extend(columns)

    # concatenate all features together
    X = sparse.csr_matrix(sparse.hstack(feature_matrices))
    #print "Feature martix size:", X.shape

    return X, column_names


def get_item_dicts(datasets, test_fold, dev_subfold):
    train_dict = {}
    valid_dict = {}
    test_dict = {}
    for f in datasets:
        # add the training items for this dataset to the list
        train_dict[f] = ds.get_train_documents(f, test_fold, dev_subfold)
        valid_dict[f] = ds.get_dev_documents(f, test_fold, dev_subfold)
        test_dict[f] = ds.get_test_documents(f, test_fold)
    return train_dict, valid_dict, test_dict


def get_items_and_indices(datasets, item_dict, index):
    items = []
    for f in datasets:
        items.extend(item_dict[f])
    indices = [index.index(i) for i in items]
    return items, indices


def compare_r_strengths(datasets, X, index, column_names, all_y, train_dict, valid_dict,
                        model_type, regularization=None, alphas=None, beta=None,
                        reuse=False, orig_T=None, T_hat=None, tau=None):

    codes = all_y.columns
    cv_train_f1s = pd.DataFrame(np.zeros([len(codes), len(alphas)]), index=codes, columns=[str(l) for l in alphas])
    cv_valid_f1s = pd.DataFrame(np.zeros([len(codes), len(alphas)]), index=codes, columns=[str(l) for l in alphas])

    train_items, train_indices = get_items_and_indices(datasets, train_dict, index)
    valid_items, valid_indices = get_items_and_indices(datasets, valid_dict, index)

    # go through labels one by one
    for code in codes:
        y = all_y[code].as_matrix()

        #if y.sum() == 0:
        if False:
            ## I THINK I CAN DELETE ALL OF THIS...
            """
            model = models_sparse.train_model(X[train_indices, :], y.loc[train_items].as_matrix(), column_names,
                                              model_type=model_type, penalty=regularizer, alpha=0)
            f1_train, acc_train = model.eval_f1_acc(X[train_indices, :], y.loc[train_items])
            f1_dev, acc_dev = model.eval_f1_acc(X[valid_indices, :], y.loc[valid_items])

            for r_strength in lambdas:
                if reuse:
                    f1_dev, T_hat = mask_valid_f1(f1_dev, f1_train, orig_T, T_hat, tau)

                cv_train_f1s.loc[code, str(r_strength)] = f1_train
                cv_valid_f1s.loc[code, str(r_strength)] = f1_dev
            """
            pass
        else:
            for alpha in alphas:
                model = SparseModel(model_type=model_type, column_names=column_names,
                                    regularization=regularization, alpha=alpha, beta=beta)

                model.fit(X[train_indices, :], y[train_indices])

                f1_train, acc_train = model.eval_f1_acc(X[train_indices, :], y[train_indices])
                f1_dev, acc_dev = model.eval_f1_acc(X[valid_indices, :], y[valid_indices])

                if reuse:
                    f1_dev, T_hat = mask_valid_f1(f1_dev, f1_train, orig_T, T_hat, tau)

                cv_train_f1s.loc[code, str(alpha)] = f1_train
                cv_valid_f1s.loc[code, str(alpha)] = f1_dev

    return cv_train_f1s, cv_valid_f1s


def train_and_predict(datasets, X, index, column_names, all_y, train_dict, valid_dict,
                      model_type, regularization=None, alphas=None, beta=None, verbose=1):

    codes = all_y.columns

    train_items, train_indices = get_items_and_indices(datasets, train_dict, index)
    valid_items, valid_indices = get_items_and_indices(datasets, valid_dict, index)

    # create empty prediction matrices
    prediction_matrices_train = create_prediction_matrices(datasets, train_dict)
    prediction_matrices_valid = create_prediction_matrices(datasets, valid_dict)
    prediction_matrices_train_prob = create_prediction_matrices(datasets, train_dict, dtype=float)
    prediction_matrices_valid_prob = create_prediction_matrices(datasets, valid_dict, dtype=float)

    models = {}

    for code_index, code in enumerate(codes):
        y = all_y[code].as_matrix()

        model = SparseModel(model_type=model_type, column_names=column_names,
                            regularization=regularization, alpha=float(alphas[code_index]), beta=beta)

        model.fit(X[train_indices, :], y[train_indices])

        models[code] = model

        # make predictions
        predictions_train = model.predict(X[train_indices, :])
        predictions_valid = model.predict(X[valid_indices, :])
        predictions_train_prob = model.predict_log_probs(X[train_indices, :])
        predictions_valid_prob = model.predict_log_probs(X[valid_indices, :])

        # write predictions back to the corresponding code for each question
        for f in datasets:
            f_train_indices = [train_items.index(i) for i in train_dict[f]]
            f_valid_indices = [valid_items.index(i) for i in valid_dict[f]]
            prediction_matrices_train[f][code] = predictions_train[f_train_indices]
            prediction_matrices_train_prob[f][code] = predictions_train_prob[f_train_indices]
            if len(f_valid_indices) > 0:
                prediction_matrices_valid[f][code] = predictions_valid[f_valid_indices]
                prediction_matrices_valid_prob[f][code] = predictions_valid_prob[f_valid_indices]

    return prediction_matrices_train, prediction_matrices_valid, \
        prediction_matrices_train_prob, prediction_matrices_valid_prob, models


def create_prediction_matrices(datasets, items_dict, dtype=int):
    prediction_matrices = {}
    for dataset in datasets:
        y = labels.get_labels(dataset)
        prediction_matrices[dataset] = pd.DataFrame(np.zeros([len(items_dict[dataset]), len(y.columns)],
                                                    dtype=dtype), index=items_dict[dataset],
                                                    columns=y.columns)
    return prediction_matrices


def create_summary_dfs(datasets):
    summary_columns = datasets + ['overall']
    microf1 = pd.DataFrame(columns=summary_columns)
    macrof1 = pd.DataFrame(columns=summary_columns)
    acc = pd.DataFrame(columns=summary_columns)
    pp = pd.DataFrame(columns=summary_columns)
    return {'microf1': microf1, 'macrof1': macrof1, 'acc': acc, 'pp': pp}


def evaluate_predictions(output_dfs, datasets, predictions, true, item_dict, rowname,
                         reuse=False, orig_T=None, T_hat=None, tau=None, train_summary=None):
    true_concat = []
    pred_concat = []
    for f in datasets:
        if len(item_dict[f]) > 0:
            micro_f1, acc = evaluation.calc_micro_mean_f1_acc(true.loc[item_dict[f]], predictions[f])
            macro_f1, pp = evaluation.calc_macro_mean_f1_pp(true.loc[item_dict[f]], predictions[f])
            if reuse:
                micro_f1, T_hat = mask_valid_f1(micro_f1, train_summary['microf1'].loc[rowname, f], orig_T, T_hat, tau)
                macro_f1, T_hat = mask_valid_f1(macro_f1, train_summary['macrof1'].loc[rowname, f], orig_T, T_hat, tau)
            output_dfs['microf1'].loc[rowname, f] = micro_f1
            output_dfs['macrof1'].loc[rowname, f] = macro_f1
            output_dfs['acc'].loc[rowname, f] = acc
            output_dfs['pp'].loc[rowname, f] = pp
            pred_concat.append(predictions[f])
            true_concat.append(true.loc[item_dict[f]])
    pred_concat = pd.concat(pred_concat, axis=0)
    true_concat = pd.concat(true_concat, axis=0)
    micro_f1, acc = evaluation.calc_micro_mean_f1_acc(true_concat, pred_concat)
    macro_f1, pp = evaluation.calc_macro_mean_f1_pp(true_concat, pred_concat)
    if reuse:
        micro_f1, T_hat = mask_valid_f1(micro_f1, train_summary['microf1'].loc[rowname, 'overall'], orig_T, T_hat, tau)
        macro_f1, T_hat = mask_valid_f1(macro_f1, train_summary['macrof1'].loc[rowname, 'overall'], orig_T, T_hat, tau)
    output_dfs['microf1'].loc[rowname, 'overall'] = micro_f1
    output_dfs['macrof1'].loc[rowname, 'overall'] = macro_f1
    output_dfs['acc'].loc[rowname, 'overall'] = acc
    output_dfs['pp'].loc[rowname, 'overall'] = pp
    return T_hat


def create_summary_dfs_per_code(datasets, codes, true, test_fold, best_lambdas):
    dfs = {}
    for f in datasets:
        all_items = ds.get_all_documents(f)
        df = pd.DataFrame(columns=codes)
        df.loc['true_y_sum'] = true.loc[all_items].sum(axis=0)
        df.loc['lambdas'] = best_lambdas
        dfs[f] = df
    return dfs

def evaluate_predictions_per_code(summary_df, datasets, predictions, true, item_dict, rowname):
    codes = true.columns
    for f in datasets:
        if len(item_dict[f]) > 0:
            for code_index, code in enumerate(codes):
                true_col = true.loc[item_dict[f], code]
                pred_col = predictions[f][code]
                f1, acc = evaluation.calc_f1_and_acc_for_column(true_col, pred_col)
                summary_df[f].loc[rowname, code] = f1


def write_summary_dfs_to_file(exp_dir, prefix, summary_dfs):
    output_filename = fh.make_filename(make_results_dir(exp_dir), prefix + '_micro_f1', 'csv')
    summary_dfs['microf1'].to_csv(output_filename)
    output_filename = fh.make_filename(make_results_dir(exp_dir), prefix + '_macro_f1', 'csv')
    summary_dfs['macrof1'].to_csv(output_filename)
    output_filename = fh.make_filename(make_results_dir(exp_dir), prefix + '_acc', 'csv')
    summary_dfs['acc'].to_csv(output_filename)
    output_filename = fh.make_filename(make_results_dir(exp_dir), prefix + '_pp', 'csv')
    summary_dfs['pp'].to_csv(output_filename)


def mask_valid_f1(dev_f1, train_f1, orig_T, T_hat, tau):
    xi = stats.laplace.rvs(scale=2*tau/np.sqrt(2))
    gamma = stats.laplace.rvs(scale=4*tau/np.sqrt(2))
    eta = stats.laplace.rvs(scale=8*tau/np.sqrt(2))
    if np.abs(dev_f1 - train_f1) > (T_hat + eta):
        T_hat = orig_T + gamma
        return dev_f1 + xi, T_hat
    else:
        return train_f1, T_hat

""" CHECK
# do baseline predictions
dumb_baseline = get_dumb_baseline_predictions(multi_y, training_items)
db_acc = evaluation.calc_task_wise_acc(multi_y.loc[dev_items], dumb_baseline.loc[dev_items], 'dumb_baseline')

smart_baseline = get_smart_baseline_predictions(multi_y, training_items)
sb_acc = evaluation.calc_task_wise_acc(multi_y.loc[dev_items], smart_baseline.loc[dev_items], 'smart_baseline')
sb_f1s = evaluation.calc_task_wise_f1s(multi_y.loc[dev_items], smart_baseline.loc[dev_items], 'smart_baseline')

results = pd.concat([report_header, db_acc, sb_acc, model_acc, sb_f1s, model_f1s], axis=0)
results_filename = get_results_filename(name, test_fold, dev_subfold, f)
results.to_csv(results_filename)

micro_f1, acc = evaluation.calc_micro_mean_f1_acc(multi_y.loc[dev_items], dumb_baseline.loc[dev_items])
macro_f1, pp = evaluation.calc_macro_mean_f1_pp(multi_y.loc[dev_items], dumb_baseline.loc[dev_items])
summary_report.loc['db_acc', f] = acc
summary_report.loc['db_micro_f1', f] = micro_f1
summary_report.loc['db_macro_f1', f] = macro_f1
summary_report.loc['db_percent_perfect', f] = pp

micro_f1, acc = evaluation.calc_micro_mean_f1_acc(multi_y.loc[dev_items], smart_baseline.loc[dev_items])
macro_f1, pp = evaluation.calc_macro_mean_f1_pp(multi_y.loc[dev_items], smart_baseline.loc[dev_items])
summary_report.loc['sb_acc', f] = acc
summary_report.loc['sb_micro_f1', f] = micro_f1
summary_report.loc['sb_macro_f1', f] = macro_f1
summary_report.loc['sb_percent_perfect', f] = pp
"""

"""

micro_f1, acc = evaluation.calc_micro_mean_f1_acc(multi_y.loc[dev_items], predictions.loc[dev_items])
macro_f1, pp = evaluation.calc_macro_mean_f1_pp(multi_y.loc[dev_items], predictions.loc[dev_items])

summary_report.loc['model_acc', f] = acc
summary_report.loc['model_micro_f1', f] = micro_f1
summary_report.loc['model_macro_f1', f] = macro_f1
summary_report.loc['model_percent_perfect', f] = pp

non_zero_items = [i for i in dev_items if np.sum(predictions.loc[i]) != 0]
micro_f1, acc = evaluation.calc_micro_mean_f1_acc(multi_y.loc[non_zero_items], predictions.loc[non_zero_items])
macro_f1, pp = evaluation.calc_macro_mean_f1_pp(multi_y.loc[non_zero_items], predictions.loc[non_zero_items])

summary_report.loc['non_zero_model_excluded', f] = len(dev_items) - len(non_zero_items)
summary_report.loc['non_zero_model_acc', f] = acc
summary_report.loc['non_zero_model_micro_f1', f] = micro_f1
summary_report.loc['non_zero_model_macro_f1', f] = macro_f1
summary_report.loc['non_zero_model_percent_perfect', f] = pp

"""

## OUTPUT A SIMPLE ERROR ANALYSIS
""" CHECK
output_columns = ['n_errors', 'percent_oov', 'text']
error_summary = pd.DataFrame(index=dev_items, columns=output_columns)
errors = multi_y.loc[dev_items] - predictions.loc[dev_items] + 0.1*multi_y.loc[dev_items]
error_summary['n_errors'] = np.sum(np.abs(multi_y.loc[dev_items] - predictions.loc[dev_items]), axis=1)
error_summary['percent_oov'] = n_oov.loc[dev_items]
responses = fh.read_json(defines.data_raw_text_file)
for i in dev_items:
    error_summary.loc[i, 'text'] = responses[i]
error_output = pd.concat([errors, error_summary], axis=1)
error_filename = get_error_filename(name, test_fold, dev_subfold, f)
error_output.to_csv(error_filename)
"""


#print summary_report
#return summary_report


def get_dumb_baseline_predictions(multi_y, training_items):
    n_items, n_tasks = multi_y.shape
    dumb_baseline = pd.DataFrame(np.zeros([n_items, n_tasks]),
                                 index=multi_y.index, columns=multi_y.columns)
    tasks = multi_y.columns
    for task in tasks:
        mode = stats.mode(multi_y.loc[training_items, task])[0]
        dumb_baseline.loc[:, task] = np.repeat(mode, n_items)

    return dumb_baseline


def get_smart_baseline_predictions(multi_y, training_items):
    n_items, n_tasks = multi_y.shape
    row_counts = {}
    training_ys = multi_y.loc[training_items].index
    for i in training_ys:
        row = ''.join([str(c) for c in multi_y.loc[i].tolist()])
        if row in row_counts:
            row_counts[row] += 1
        else:
            row_counts[row] = 1
    index = np.argmax(row_counts.values())
    most_common_row = row_counts.keys()[index]
    print most_common_row
    row = np.zeros([1, len(list(most_common_row))], dtype=int)
    row[0, :] = np.array([int(i) for i in list(most_common_row)])
    smart_baseline = pd.DataFrame(np.repeat(row, n_items, axis=0),
                                  index=multi_y.index, columns=multi_y.columns)
    return smart_baseline



def get_groups(group_file):
    groups = []
    lines = fh.read_text(group_file)
    for line in lines:
        if len(line) > 0:
            groups.append(line.split())
    return groups


def make_exp_dir(group, test_fold, name):
    return fh.makedirs(defines.exp_dir, '_'.join(group), "test_fold_" + str(test_fold), name)

def make_training_dir(exp_dir):
    return fh.makedirs(exp_dir, 'training')

def make_results_dir(exp_dir):
    return fh.makedirs(exp_dir, 'results')

def make_models_dir(exp_dir):
    return fh.makedirs(exp_dir, 'models')

def make_prediction_dir(exp_dir):
    return fh.makedirs(exp_dir, 'predictions')

if __name__ == '__main__':
    main()
