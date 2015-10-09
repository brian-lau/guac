import re
import codecs

import pandas as pd

from ..util import defines
from ..util import file_handling as fh
from ..preprocessing import labels
from ..preprocessing import data_splitting as ds

from . import common

code_names = ['General',
              'Another party',
              'Specific person',
              'Ideology',
              'Electability',
              'Scandal',
              'Campaign',
              'Other past activity',
              'Ability',
              'Honesty',
              'Intelligence',
              'Leadership',
              'Management',
              'Religion',
              'Education',
              'Physical appearance',
              'Demographics',
              'Policy-Economic',
              'Policy-Poor people',
              'Policy-Liberty',
              'Policy-Enemy countries',
              'Policy-Friendly countries',
              'Policy-General foreign',
              'Policy-Other',
              'Groups',
              'Emotions/Feelings',
              'Party-Other',
              'Non-party',
              'Party membership',
              "Don't know",
              'Refuse to answer',
              'Respondent-Other',
              'Other']

def output_responses(dataset):
    print dataset
    output_dir = fh.makedirs(defines.web_dir, 'DRLD')
    rnn_dir = fh.makedirs(defines.exp_dir, 'rnn', 'bayes_opt_rnn_LSTM_reuse_mod_34_rerun', 'fold0', 'responses')
    blm_dir = fh.makedirs(defines.exp_dir, 'Democrat-Dislikes_Democrat-Likes_Republican-Dislikes_Republican-Likes', 'test_fold_0', 'L1LR_all_groups_a0', 'models')
    predictions_dir = fh.makedirs(defines.exp_dir, 'Democrat-Dislikes_Democrat-Likes_Republican-Dislikes_Republican-Likes', 'test_fold_0', 'L1LR_all_groups_a0', 'predictions')
    train_pred = pd.read_csv(fh.make_filename(predictions_dir, dataset + '_train', 'csv'), header=0, index_col=0)
    test_pred = pd.read_csv(fh.make_filename(predictions_dir, dataset + '_test', 'csv'), header=0, index_col=0)

    text_file_dir = fh.makedirs(defines.data_dir, 'rnn')
    text = fh.read_json(fh.make_filename(text_file_dir, 'ngrams_n1_m1_rnn', 'json'))

    true = labels.get_labels([dataset])
    all_items = ds.get_all_documents(dataset)

    for i in all_items:
        print i
        true_i = true.loc[i]
        rnn_file = fh.make_filename(rnn_dir, i, 'csv')
        rnn_vals = pd.read_csv(rnn_file, header=-1)
        rnn_vals.columns = true.columns

        if i in train_pred.index:
            pred_i = train_pred.loc[i]
        else:
            pred_i = test_pred.loc[i]

        output_filename = fh.make_filename(output_dir, i, 'html')
        with codecs.open(output_filename, 'w') as output_file:

            output_file.write(common.make_header(i))
            output_file.write(common.make_body_start())
            table_header = ['Label'] + text[i] + ['True', 'Pred.']
            output_file.write(common.make_table_start(style='t1'))
            output_file.write(common.make_table_header(table_header))
            for code_index, code in enumerate(true.columns):
                # load coefficients from unigram model
                words = text[i]
                model_filename = fh.make_filename(blm_dir, re.sub(' ', '_', code), 'json')
                model = fh.read_json(model_filename)
                intercept = float(model.get('intercept', 1.0))
                if 'coefs' in model:
                    coefs = dict(model['coefs'])
                    colours = [str((0, 0, 0))]
                    for word in words:
                        coef = coefs.get('_n1_' + word, 0.0)/abs(intercept)
                        val = int(255 - (min(1, abs(coef))*255))
                        if coef > 0:
                            colours += [(val, val, 255)]
                        else:
                            colours += [(255, val, val)]
                else:
                    colours = [str((0, 0, 0))]
                    colours += [(255, 255, 255) for w in words]

                colours += [str((0, 0, 0))]*2
                row = [code_names[code_index]] + words + [str(true_i[code]), str(int(pred_i[code]))]
                output_file.write(common.make_table_row(row, colours=colours))

                colours = [str((0, 0, 0))]
                vals = [int(235 - (v*235)) for v in rnn_vals[code]]
                colours += [(v, 235, v) for v in vals]
                colours += [str((0, 0, 0))]*2
                row = [' '] + text[i] + [' ', str(int(rnn_vals[code].max()))]
                output_file.write(common.make_table_row(row, colours=colours))
            output_file.write(common.make_table_end())

            output_file.write(common.make_body_end())
            output_file.write(common.make_footer())

def main():
    output_responses(dataset='Democrat-Likes')
    output_responses(dataset='Democrat-Dislikes')
    output_responses(dataset='Republican-Dislikes')
    output_responses(dataset='Republican-Likes')

if __name__ == '__main__':
    main()
