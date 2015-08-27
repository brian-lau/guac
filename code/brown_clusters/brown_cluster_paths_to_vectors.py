import numpy as np

from ..util import defines
from ..util import file_handling as fh

#input_filename = base_dir + 'Tools/NLP/Stanford/brown-cluster-master/drld_brown_input-c200-p1.out/paths'
bc_dim = 200

lines = fh.read_text(input_filename)

index = {}
vectors = {}
counts = {}

for line in lines:
    parts = line.split()
    if len(parts) > 2:
        code = parts[0]
        word = parts[1]
        count = int(parts[2])

        if code not in vectors:
            new_vector = np.zeros(bc_dim, dtype=int)
            new_vector[len(vectors)] = 1
            vectors[code] = list(new_vector)

        index[word] = code
        counts[word] = count

print len(vectors)

output = {'index': index, 'vectors': vectors, 'counts': counts}
output_filename = fh.make_filename(defines.vectors_dir, 'brown_vectors', 'json')
fh.write_to_json(output, output_filename)
