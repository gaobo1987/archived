import os
import unicodedata


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def strip_accents_and_lower_case(s: str) -> str:
    return strip_accents(s).lower()


def read_ner_examples_from_file(data_dir, do_lower_case=False, do_strip_accents=False):
    file_path = os.path.join(data_dir)
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid=f'{guid_index}', words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(' ')
                word = splits[0]
                if do_strip_accents:
                    word = strip_accents(word)
                if do_lower_case:
                    word = word.lower()
                label = splits[-1].replace('\n', '') if len(splits) > 1 else 'O'
                words.append(word)
                labels.append(label)

        if len(words) > 0:
            examples.append(InputExample(guid=f'{guid_index}', words=words, labels=labels))
    return examples
