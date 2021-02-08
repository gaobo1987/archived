class DictionaryEntry:
    def __init__(self, inflected_form=None, lemma=None, part_of_speech=None, subcategory=None, morph_attributes=None):
        self.inflected_form = inflected_form
        self.lemma = lemma
        self.part_of_speech = part_of_speech
        self.subcategory = subcategory
        self.morph_attributes = morph_attributes

    def print_entry(self):
        print("Inflected form : " + self.inflected_form)
        print("Lemma form : " + self.lemma)
        print("Part of speech form : " + self.part_of_speech)
        print("Subcategory form : " + self.subcategory)
        print("Morph attributes form : " + self.morph_attributes)
