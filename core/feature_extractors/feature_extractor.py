import os.path

from ..util import file_handling as fh
from ..util import defines

class FeatureExtractor:

    params = None

    def __init__(self, name, prefix, feature_type, test_fold=0, dev_subfold=None):
        self.params = {'name': name,
                       'prefix': prefix,
                       'type': feature_type,
                       'test_fold': int(test_fold)}
        if dev_subfold is not None:
            self.params['dev_subfold'] = int(dev_subfold)
        else:
            self.params['dev_subfold'] = None

        self.dirname = FeatureExtractor.make_dirname(self)

    def get_name(self):
        return self.params['name']

    def get_prefix(self):
        return self.params['prefix']

    def get_type(self):
        return self.params['type']

    def get_test_fold(self):
        return self.params['test_fold']

    def get_dev_subfold(self):
        return self.params['dev_subfold']

    def make_dirname(self):
        dirname = self.get_name() + ',' + self.get_type() + ',' \
            + str(self.get_test_fold()) + ',' + str(self.get_dev_subfold())
        return os.path.join(defines.features_dir, dirname)

    @classmethod
    def parse_dirname(cls, dirname):
        parts = fh.get_basename(dirname).split('_')
        assert len(parts) > 3
        name = parts[0]
        feature_type = parts[1]
        test_fold = int(parts[2])
        dev_subfold = int(parts[3])
        if len(parts) > 4:
            extra = parts[4:]
        else:
            extra = None
        return name, feature_type, test_fold, dev_subfold, extra

    def get_dirname(self):
        return self.dirname

    def get_vocab_filename(self):
        return fh.make_filename(fh.makedirs(self.get_dirname()), 'vocab', 'json')

    def get_index_filename(self):
        return fh.make_filename(fh.makedirs(self.get_dirname()), 'index', 'json')

    def write_feature_definition(self, filename):
        fh.write_to_json(self.params, filename)

    def extract_features(self):
        pass

