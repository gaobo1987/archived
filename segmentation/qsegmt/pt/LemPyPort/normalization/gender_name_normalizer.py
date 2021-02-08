import xmltodict


class GenderNameNormalizer:
    def __init__(self):

        self.declesion_exceptions = []
        self.declesion_targets = []
        self.declesion_tags = []
        self.declesion_replacement = []

    def gender_name_normalizer_load(self, file_name):
        with open(file_name) as file:
            doc = xmltodict.parse(file.read())
            result = (doc["gendernouns"]["replacement"])
            '''
            Get the number of rules and set the size of the class variables
            '''
            gender_name_number_of_rules = len(result)
            self.declesion_exceptions = [None] * gender_name_number_of_rules
            self.declesion_targets = [None] * gender_name_number_of_rules
            self.declesion_tags = [None] * gender_name_number_of_rules
            self.declesion_replacement = [None] * gender_name_number_of_rules
            '''
            Check if the rule has any of the elements 
            '''
            for index, elem in enumerate(result):
                if '@target' in elem.keys():
                    self.declesion_targets[index] = elem["@target"]
                else:
                    self.declesion_targets[index] = ""
                if '@exceptions' in elem.keys():
                    self.declesion_exceptions[index] = elem["@exceptions"]
                else:
                    self.declesion_exceptions[index] = ""
                if '@tag' in elem.keys():
                    self.declesion_tags[index] = elem["@tag"]
                else:
                    self.declesion_tags[index] = ""
                if '#text' in elem.keys():
                    self.declesion_replacement[index] = elem["#text"]
                else:
                    self.declesion_replacement[index] = ""

    def print_name_normalizer(self):
        for i in range(len(self.declesion_targets)):
            print(i)
            print("---------\n")
            print("Target: " + self.declesion_targets[i])
            print("Tags: " + self.declesion_tags[i])
            print("Exceptions: " + self.declesion_exceptions[i])
            print("Replacement: " + self.declesion_replacement[i])

    def normalize_gender_name(self, token, tag):
        lemmatized_word = ""
        '''
        Token to lowercase to match the rules
        '''
        normalization = token.lower()
        '''
        Check for every rule it the element matches the target,
        if the tags match and if the element is not an exception
        '''
        for index, elem in enumerate(self.declesion_targets):
            if (normalization == self.declesion_targets[index]
                    and tag.lower() in self.declesion_tags[index].split("|")):
                lemmatized_word = self.declesion_replacement[index]
                return lemmatized_word
        return lemmatized_word
