import h5py
from tqdm import tqdm

# ref: http://millionsongdataset.com/pages/example-track-description/
# 02: artist_familiarity
# 03: artist_hotttness
# 04: artist_id
# 06: location
# 09: artist_name
# 14: album_name
# 17: song_id
# 18: song_name
# 19: track_7digitalid

MILLION_SONGS_FILE = 'msd_summary_file.h5'
RESOURCES = {
    'song': {
        'index': 18,
        'description': 'song_name',
        'output': 'songs.csv'
    },
    'album': {
        'index': 14,
        'description': 'album_name',
        'output': 'albums.csv'
    }
}


def extract():
    f = h5py.File(MILLION_SONGS_FILE, "r")
    data = f['metadata']['songs']
    indices = []
    outs = []
    for k in RESOURCES:
        resource = RESOURCES[k]
        indices.append(resource['index'])
        outs.append(open(resource['output'], 'w'))

    for s in tqdm(data):
        arr = list(s)
        for i, indx in enumerate(indices):
            entity = arr[indx].decode('utf-8').strip()
            if len(entity) > 0:
                outs[i].write(entity+'\n')

    for out in outs:
        print(f'{out} finished')
        out.close()

    f.close()


def is_ascii(data):
    try:
        data.decode('ASCII')
    except UnicodeDecodeError:
        return False
    else:
        return True


def is_utf8(data):
    try:
        data.decode('UTF-8')
    except UnicodeDecodeError:
        return False
    else:
        return True


def is_utf16(data):
    try:
        data.decode('UTF-16')
    except UnicodeDecodeError:
        return False
    else:
        return True
