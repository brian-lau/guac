import os

base_dir = ''
cwd = os.getcwd()
parts = os.path.split(cwd)
if parts[-1] == 'code':
    base_dir = os.path.split(parts[0])[0]
elif parts[-1] == 'guac':
    base_dir = parts[0]
parts = os.path.split(base_dir)
assert parts[1] == 'guac'

print "Base directory:", base_dir

data_dir = os.path.join(base_dir, 'data')
features_dir = os.path.join(base_dir, 'features')
resources_dir = os.path.join(base_dir, 'resources')

data_raw_dir = os.path.join(data_dir, 'raw')

data_raw_csv_dir = os.path.join(data_raw_dir, 'csv')
data_raw_combined_dir = os.path.join(data_raw_dir, 'combined')
data_raw_text_dir = os.path.join(data_raw_dir, 'text')
data_raw_concat_dir = os.path.join(data_raw_dir, 'concat')
data_raw_labels_dir = os.path.join(data_raw_dir, 'labels')
data_raw_sents_dir = os.path.join(data_raw_dir, 'sents')

data_raw_text_file = os.path.join(data_raw_text_dir, 'text.json')

data_token_dir = os.path.join(data_dir, 'tokens')
data_values_dir = os.path.join(data_dir, 'values')
data_feature_dir = os.path.join(data_dir, 'features')
data_featuredefns_dir = os.path.join(data_dir, 'feature_definitions')
data_processed_dir = os.path.join(data_dir, 'processed')
data_processed_text_dir = os.path.join(data_processed_dir, 'text')
data_semafor_dir = os.path.join(data_processed_dir, 'semafor')
data_stanford_dir = os.path.join(data_processed_dir, 'stanford')
data_rnn_dir = os.path.join(data_dir, 'rnn')

data_normalized_text_file = os.path.join(data_processed_text_dir, 'normalized.json')

resources_group_dir = os.path.join(resources_dir, 'groups')
resources_clusters_dir = os.path.join(resources_dir, 'clusters')

vectors_dir = os.path.join(resources_dir, 'vectors')
non_distributional_vectors_dir = os.path.join(vectors_dir, 'non-distributional')
word2vec_vectors_filename = os.path.join(vectors_dir, 'GoogleNews-vectors-negative300.bin')
brown_augmented_word2vec_filename = os.path.join(vectors_dir, 'brown_augmented_word2vec_300.csv')
my_word2vec_filename = os.path.join(vectors_dir, 'anes_word2vec_300.bin')

data_subsets_dir = os.path.join(data_dir, 'subsets')

exp_dir = os.path.join(data_dir, '..', 'experiments')

data_dir_20ng = os.path.join(base_dir, '20ng')

