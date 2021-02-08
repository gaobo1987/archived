import numpy as np
from tqdm import tqdm


def normalize_vecs(input_filename, output_filename):
    # input_filename = '../data/msmarco-docs-10000-vecs-512.npy'
    # output_filename = '../data/msmarco-docs-10000-512-norm.npy'
    print('loading ', input_filename)
    npy = np.load(input_filename, allow_pickle=True)
    data = npy.item()
    d = {}
    print('normalizing...')
    for k in tqdm(data, total=len(data)):
        vec = data[k]
        nvec = vec / np.linalg.norm(vec)
        d[k] = nvec
    print('saving...')
    np.save(output_filename, d)


if __name__ == '__main__':
    MAX_SEQ = 128
    VERSION = 'v2'
    CUT = 1000 if MAX_SEQ == 128 else 10000
    folder_name = f'{MAX_SEQ}-{VERSION}'
    infile = f'../data/{folder_name}/msmarco-docs-{CUT}-vecs-{MAX_SEQ}-{VERSION}.npy'
    outfile = f'../data/{folder_name}/msmarco-docs-{CUT}-vecs-{MAX_SEQ}-{VERSION}-norm.npy'
    normalize_vecs(infile, outfile)
