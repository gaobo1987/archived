import errno, os, sys, json
import random
from tqdm import tqdm
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# Sadly, Python fails to provide the following magic number for us.
ERROR_INVALID_NAME = 123


# For more details, see the documentation:
# * Training: https://spacy.io/usage/training
# * NER: https://spacy.io/usage/linguistic-features#named-entities
# Example train data:
# TRAIN_DATA = [
#     ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
#     ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
# ]
def train(train_data, model=None, output_dir=None, n_iter=2, lang='nl'):
    """Load the model, set up the pipeline and train the entity recognizer."""
    spacy.prefer_gpu() # or spacy.require_gpu() or spacy.prefer_gpu()
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank(lang)  # create blank Language class
        print(f"Created blank {lang} model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        # prev_loss = float('inf')
        # prev_loss = 90610 # (custom_blank)
        prev_loss = 92273 # (custom_default)
        for itn in range(n_iter):
            print('Iteration ', itn)
            random.shuffle(train_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in tqdm(batches):
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses
                )
            print("Losses", losses)
            if losses['ner'] < prev_loss:
                prev_loss = losses['ner']
                save_model_to_disk(nlp, output_dir)


def save_model_to_disk(nlp, output_dir):
    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


# convert tokenized text to the json format that is suitable for spacy NER training
def convert_iob_to_json(data_path, output_json_path=''):
    f = open(data_path, 'r', encoding='utf8')
    json_result = []
    sent_str = ''
    entities = []
    for l in tqdm(f, unit_scale=True, unit=' lines'):
        if l != '\n':
            parts = l[:-1].split(' ')
            token = parts[0]
            tag = parts[1]
            sent_str += token + ' '
            if tag != 'O':
                entities.append({
                    'token': token,
                    'tag': tag
                })
        else:
            sent = sent_str[:-1].strip()
            sent = sent.replace(' , ', ', ')
            sent = sent.replace(' .', '.')
            sent = sent.replace(' !', '!')
            sent = sent.replace(' ?', '?')
            sent = sent.replace(' : ', ': ')
            sent = sent.replace(' ( ', ' (')
            sent = sent.replace(' ) ', ') ')
            start_search_index = 0
            for ent in entities:
                start_index = sent.index(ent['token'], start_search_index)
                end_index = start_index + len(ent['token'])
                ent['start_index'] = start_index
                ent['end_index'] = end_index
                start_search_index = end_index

            if len(sent) > 0:
                json_result.append({
                    'sentence': sent,
                    'entities': entities.copy()
                })
            else:
                print('found empty sentence')
            sent_str = ''
            entities = []

    f.close()

    if is_pathname_valid(output_json_path):
        with open(output_json_path, 'w') as out:
            json.dump(json_result, out)

    return json_result


def convert_iob_json_objects_to_spacy_tuples(json):
    tuples = []
    for obj in tqdm(json, unit_scale=True, unit=' items'):
        sent = obj['sentence']
        entities = []
        for ent in obj['entities']:
            tag = ent['tag']
            ent_type = tag if tag == 'O' else tag[2:]
            entities.append((ent['start_index'], ent['end_index'], ent_type))
        tpl = (sent, {'entities': entities})
        tuples.append(tpl)

    return tuples


def is_pathname_valid(pathname: str) -> bool:
    '''
    `True` if the passed pathname is a valid pathname for the current OS;
    `False` otherwise.
    '''
    # If this pathname is either not a string or is but is empty, this pathname
    # is invalid.
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        # Strip this pathname's Windows-specific drive specifier (e.g., `C:\`)
        # if any. Since Windows prohibits path components from containing `:`
        # characters, failing to strip this `:`-suffixed prefix would
        # erroneously invalidate all valid absolute Windows pathnames.
        _, pathname = os.path.splitdrive(pathname)

        # Directory guaranteed to exist. If the current OS is Windows, this is
        # the drive to which Windows was installed (e.g., the "%HOMEDRIVE%"
        # environment variable); else, the typical root directory.
        root_dirname = os.environ.get('HOMEDRIVE', 'C:') \
            if sys.platform == 'win32' else os.path.sep
        assert os.path.isdir(root_dirname)   # ...Murphy and her ironclad Law

        # Append a path separator to this directory if needed.
        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        # Test whether each path component split from this pathname is valid or
        # not, ignoring non-existent and non-readable path components.
        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            # If an OS-specific exception is raised, its error code
            # indicates whether this pathname is valid or not. Unless this
            # is the case, this exception implies an ignorable kernel or
            # filesystem complaint (e.g., path not found or inaccessible).
            #
            # Only the following exceptions indicate invalid pathnames:
            #
            # * Instances of the Windows-specific "WindowsError" class
            #   defining the "winerror" attribute whose value is
            #   "ERROR_INVALID_NAME". Under Windows, "winerror" is more
            #   fine-grained and hence useful than the generic "errno"
            #   attribute. When a too-long pathname is passed, for example,
            #   "errno" is "ENOENT" (i.e., no such file or directory) rather
            #   than "ENAMETOOLONG" (i.e., file name too long).
            # * Instances of the cross-platform "OSError" class defining the
            #   generic "errno" attribute whose value is either:
            #   * Under most POSIX-compatible OSes, "ENAMETOOLONG".
            #   * Under some edge-case OSes (e.g., SunOS, *BSD), "ERANGE".
            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
    # If a "TypeError" exception was raised, it almost certainly has the
    # error message "embedded NUL character" indicating an invalid pathname.
    except TypeError as exc:
        return False
    # If no exception was raised, all path components and hence this
    # pathname itself are valid. (Praise be to the curmudgeonly python.)
    else:
        return True
    # If any other exception was raised, this is an unrelated fatal issue
    # (e.g., a bug). Permit this exception to unwind the call stack.
    #
    # Did we mention this should be shipped with Python already?