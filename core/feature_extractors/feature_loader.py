import os
from optparse import OptionParser

from feature_extractor_counts_ngrams import FeatureExtractorCountsNgrams
from feature_extractor_counts_anugrams import FeatureExtractorCountsAnUgrams
from feature_extractor_counts_brownclusters import FeatureExtractorCountsBrownClusters

from ..util import file_handling as fh

# Take a feature definition given on the command line
# (with comma separated parameters, without spaces)
# determine the feature it describes
# check if it exists
# if not, create it
# finally, load it
def load_feature(feature_description, index_to_load, verbose=1):
    parts = feature_description.split(',')
    feature_name = parts[0]
    extractor = None
    if feature_name == 'ngrams':
        extractor = extractor_factory(FeatureExtractorCountsNgrams, kwargs_list_to_dict(parts[1:]))
        #extractor = extractor_factory(FeatureExtractorCountsNgrams, kwargs_list_to_dict(kwargs))
    elif feature_name == 'brown':
        extractor = extractor_factory(FeatureExtractorCountsBrownClusters, kwargs_list_to_dict(parts[1:]))
        #extractor = extractor_factory(FeatureExtractorCountsBrownClusters, kwargs_list_to_dict(kwargs))
    elif feature_name == 'anugrams':
        extractor = extractor_factory(FeatureExtractorCountsAnUgrams, kwargs_list_to_dict(parts[1:]))
        #extractor = extractor_factory(FeatureExtractorCountsAnUgrams, kwargs_list_to_dict(kwargs))

    if not os.path.exists(extractor.dirname):
        if verbose > 0:
            print "Extracting", feature_description
        extractor.extract_features(write_to_file=True)
    else:
        if verbose > 1:
            print "Loading", extractor.get_full_name()
        extractor.load_from_files()

    counts, index, column_names = extractor.get_counts()
    #index = [items.index(i) for i in index_to_load]
    indices_to_load = index.loc[index_to_load, 'index']
    return counts[indices_to_load, :], column_names


def kwargs_list_to_dict(list_of_kwargs):
    kwargs = {}
    for kwarg in list_of_kwargs:
        name, value = kwarg.split('=')
        if value[0] == '[' and value[-1] == ']':
            parts = value[1:-1].split(';')
            value = parts
        kwargs[name] = value
    return kwargs

def extractor_factory(extractor_class, kwargs):
    return extractor_class(**kwargs)

def main():
    # Handle input options and arguments
    usage = "%prog dirname"
    parser = OptionParser(usage=usage)
    parser.add_option('-n', dest='ngrams', default=1,
                      help='n for ngrams; default=%default')
    parser.add_option('-m', dest='min_doc_thresh', default=1,
                      help='Minimum document threshold; default=%default')
    parser.add_option('-b', dest='binarize', action="store_true", default=False,
                      help='Binarize counts; default=%default')
    parser.add_option('-d', dest='dev_subfold', default=0,
                      help='dev subfold to not learn from; default=%default')
    parser.add_option('-t', dest='test_fold', default=0,
                      help='test fold to not learn from; default=%default')

    (options, args) = parser.parse_args()
    n = int(options.ngrams)
    min_doc_thresh = int(options.min_doc_thresh)
    binarize = options.binarize
    dev_subfold = int(options.dev_subfold)
    test_fold = int(options.test_fold)

    dirname = args[0]

    # THREE OPTIONS

    # option A: specify parameters explicitly
    extractor = FeatureExtractorCountsNgrams(test_fold=test_fold, dev_subfold=dev_subfold, n=n,
                                min_doc_threshold=min_doc_thresh, binarize=binarize)
    extractor.load_from_files(debug=True, debug_index=0)

    # option B: give it the dirname and create an object
    basename = fh.get_basename(dirname)
    extractor2 = FeatureExtractorCountsNgrams.from_files(basename)
    extractor2.load_from_files(debug=True, debug_index=0)

    # option C: same as option B, but automatic
    load_feature(dirname)

if __name__ == '__main__':
    main()
