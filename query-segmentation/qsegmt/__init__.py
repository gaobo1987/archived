import logging

logger = logging.getLogger(__name__)


class Segmenter:
    def get_instance(self):
        raise NotImplementedError

    # offline segmentation
    def index_segment(self, input_string):
        raise NotImplementedError

    # online segmentation
    def search_segment(self, input_string):
        raise NotImplementedError
