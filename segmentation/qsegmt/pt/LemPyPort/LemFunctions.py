# -*- coding: utf-8 -*-
"""
@author: João Ferreira
"""
import os
from qsegmt.pt.LemPyPort.normalization.adverb_normalizer import *
from qsegmt.pt.LemPyPort.normalization.gender_normalizer import *
from qsegmt.pt.LemPyPort.normalization.number_normalizer import *
from qsegmt.pt.LemPyPort.normalization.gender_name_normalizer import *
from qsegmt.pt.LemPyPort.normalization.augmentative_normalizer import *
from qsegmt.pt.LemPyPort.normalization.diminutive_normalizer import *
from qsegmt.pt.LemPyPort.normalization.superlative_normalizer import *
from qsegmt.pt.LemPyPort.normalization.verb_normalizer import *
from qsegmt.pt.LemPyPort.dictionary.dictionary import *
from qsegmt.pt.LemPyPort.rank.word_ranking import *
from pathlib import Path

adverb_path = ""
number_path = ""
superlative_path = ""
augmentative_path = ""
diminutive_path = ""
gender_path = ""
gender_name_path = ""
irregular_verb_path = ""
lexeme_verb_path = ""
regular_verb_path = ""
ranking_path = ""
dict_path = ""
lexical_conversions = ""
dict_exclusions = ""
run_modules = ""
break_on_hyphen = ""
break_on_underscore = ""
dictionary_main_path = ""
custom_dictionary_path = ""


def all_normalizations(adverb_norm, number_norm, superlative_norm, augmentative_norm, diminutive_norm, gender_norm,
                       gender_name_norm, verb_norm, token, tag, ranking, novo_dict):
    adverb_tags, augmentative_tags, diminutive_tags, gender_name_tags, gender_tags,\
        number_tags, superlative_tags, verb_tags = tags_classes()
    global run_modules
    global break_on_hyphen
    global break_on_underscore
    flags = run_modules
    if len(token) == 0 or len(tag) == 0:
        return token
    lemma = token.lower()
    # simplify pos
    result = token
    lex_tag = tag.upper()
    if "-" in lex_tag:
        lex_tag = lex_tag[:lex_tag.find("-")]

    # make translation
    conversion_key = lexical_conversions.split(";")
    for converted_tag in conversion_key:
        if lex_tag == converted_tag.split(":")[0]:
            lex_tag = converted_tag.split(":")[1]

    if break_on_hyphen == "True" and "-" in lemma:
        lema1 = lemma[0:lemma.find("-")]
        lema2 = lemma[lemma.find("-") + 1:]
        res_lema_one = ""
        res_lema_two = ""
        if lema1 != "-":
            res_lema_one = all_normalizations(adverb_norm, number_norm, superlative_norm, augmentative_norm,
                                              diminutive_norm, gender_norm, gender_name_norm, verb_norm, lema1, tag,
                                              ranking, novo_dict)
        if lema2 != "-":
            res_lema_two = all_normalizations(adverb_norm, number_norm, superlative_norm, augmentative_norm,
                                              diminutive_norm, gender_norm, gender_name_norm, verb_norm, lema2, tag,
                                              ranking, novo_dict)
        return res_lema_one + "-" + res_lema_two

    if break_on_underscore == "True" and "_" in lemma:
        lema1 = lemma[0:lemma.find("_")]
        lema2 = lemma[lemma.find("_") + 1:]
        res_lema_one = ""
        res_lema_two = ""
        if lema1 != "-":
            res_lema_one = all_normalizations(adverb_norm, number_norm, superlative_norm, augmentative_norm,
                                              diminutive_norm, gender_norm, gender_name_norm, verb_norm, lema1, tag,
                                              ranking, novo_dict)
        if lema2 != "-":
            res_lema_two = all_normalizations(adverb_norm, number_norm, superlative_norm, augmentative_norm,
                                              diminutive_norm, gender_norm, gender_name_norm, verb_norm, lema2, tag,
                                              ranking, novo_dict)
        return res_lema_one + "_" + res_lema_two

    if flags[0] == "1" and tag in adverb_tags.split("|"):
        # check dictionary
        if len(result) > 0:
            if result[-1] == "-":
                result = result[:-1]
        lema_dict = novo_dict.retrieve_lemas(result, lex_tag)
        if lema_dict != [] and lex_tag not in dict_exclusions.split("|"):
            res = ranking.retrieve_top_word(lema_dict)
            if res is not None:
                return ranking.retrieve_top_word(lema_dict)

        result_adverb = adverb_norm.normalize_adverb(result, tag)
        if result_adverb != "":
            result = result_adverb
    if flags[1] == "1" and tag in number_tags.split("|"):
        # check dictionary
        if len(result) > 0:
            if result[-1] == "-":
                result = result[:-1]
        lema_dict = novo_dict.retrieve_lemas(result, lex_tag)

        if lema_dict != [] and lex_tag not in dict_exclusions.split("|"):
            res = ranking.retrieve_top_word(lema_dict)
            if res is not None:
                return ranking.retrieve_top_word(lema_dict)

        result_number = number_norm.normalize_number(result, tag)
        if result_number != "":
            result = result_number

    if flags[2] == "1" and tag in superlative_tags.split("|"):
        # check dictionary
        if len(result) > 0:
            if result[-1] == "-":
                result = result[:-1]
        lema_dict = novo_dict.retrieve_lemas(result, lex_tag)
        if lema_dict != [] and lex_tag not in dict_exclusions.split("|"):
            res = ranking.retrieve_top_word(lema_dict)
            if res is not None:
                return ranking.retrieve_top_word(lema_dict)

        result_superlative = superlative_norm.normalize_superlative(result, tag)
        if result_superlative != "":
            result = result_superlative
    if flags[3] == "1" and tag in augmentative_tags.split("|"):
        # check dictionary
        if len(result) > 0:
            if result[-1] == "-":
                result = result[:-1]
        lema_dict = novo_dict.retrieve_lemas(result, lex_tag)
        if lema_dict != [] and lex_tag not in dict_exclusions.split("|"):
            res = ranking.retrieve_top_word(lema_dict)
            if res is not None:
                return ranking.retrieve_top_word(lema_dict)

        result_augmentative = augmentative_norm.normalize_augmentative(result, tag)
        if result_augmentative != "":
            result = result_augmentative
    if flags[4] == "1" and tag in diminutive_tags.split("|"):
        # check dictionary
        if len(result) > 0:
            if result[-1] == "-":
                result = result[:-1]
        lema_dict = novo_dict.retrieve_lemas(result, lex_tag)
        if lema_dict != [] and lex_tag not in dict_exclusions.split("|"):
            res = ranking.retrieve_top_word(lema_dict)
            if res is not None:
                return ranking.retrieve_top_word(lema_dict)

        result_diminutive = diminutive_norm.normalize_diminutive(result, tag)
        if result_diminutive != "":
            result = result_diminutive
    if flags[5] == "1" and tag in gender_tags.split("|"):
        # check dictionary
        if len(result) > 0:
            if result[-1] == "-":
                result = result[:-1]
        # check dictionary
        lema_dict = novo_dict.retrieve_lemas(result, lex_tag)
        if lema_dict != [] and lex_tag not in dict_exclusions.split("|"):
            res = ranking.retrieve_top_word(lema_dict)
            if res is not None:
                return ranking.retrieve_top_word(lema_dict)

        result_gender = gender_norm.normalize_gender(result, tag)
        if result_gender != "":
            result = result_gender
    if flags[6] == "1" and tag in gender_name_tags.split("|"):
        # check dictionary
        if len(result) > 0:
            if result[-1] == "-":
                result = result[:-1]
        lema_dict = novo_dict.retrieve_lemas(result, lex_tag)
        if lema_dict != [] and lex_tag not in dict_exclusions.split("|"):
            res = ranking.retrieve_top_word(lema_dict)
            if res is not None:
                return ranking.retrieve_top_word(lema_dict)

        result_gender_name = gender_name_norm.normalize_gender_name(result, tag)
        if result_gender_name != "":
            result = result_gender_name
    if flags[7] == "1" and tag in verb_tags.split("|"):
        # check dictionary
        if len(result) > 0:
            if result[-1] == "-":
                result = result[:-1]
        lema_dict = novo_dict.retrieve_lemas(result, lex_tag)
        if lema_dict != [] and lex_tag not in dict_exclusions.split("|"):
            res = ranking.retrieve_top_word(lema_dict)
            if res is not None:
                return ranking.retrieve_top_word(lema_dict)

        result_verb = verb_norm.normalize_verb(result, tag)
        if result_verb != "":
            result = result_verb
    return result


def target_sorter(valores, lista):
    valores_dic = {}
    desconta_valores = [0] * len(valores)
    for i in range(len(valores)):
        valores_dic[i] = valores[i]
        res = re.findall(r"\[(?:\w|àáãâéêíóõôúç\-)*\]", valores[i])
        if len(res) > 0:
            for elem in res:
                desconta_valores[i] += len(elem) - 2
    # indices = []
    new_list = [0] * len(valores)
    place = 0
    for k in sorted(valores_dic, key=lambda k: (len(valores_dic[k]) - desconta_valores[k]), reverse=True):
        new_list[place] = lista[k]
        place += 1
    return new_list


def get_paths_lematizador(config_file):
    global adverb_path
    global number_path
    global superlative_path
    global augmentative_path
    global diminutive_path
    global gender_path
    global gender_name_path
    global irregular_verb_path
    global lexeme_verb_path
    global regular_verb_path
    global ranking_path
    global dict_path
    global lexical_conversions
    global dict_exclusions
    global run_modules
    global break_on_hyphen
    global break_on_underscore
    global dictionary_main_path
    global custom_dictionary_path
    resources_dir = Path(config_file).parent.parent
    with open(config_file, 'r') as f:
        for line in f:
            if line[0] != "#":
                divided_line = line.split("=")
                if divided_line[0] == "adverb_path":
                    adverb_path = Path(divided_line[1].strip('\n'))
                    adverb_path = os.path.join(resources_dir, adverb_path)
                elif divided_line[0] == "number_path":
                    number_path = Path(divided_line[1].strip('\n'))
                    number_path = os.path.join(resources_dir, number_path)
                elif divided_line[0] == "superlative_path":
                    superlative_path = Path(divided_line[1].strip('\n'))
                    superlative_path = os.path.join(resources_dir, superlative_path)
                elif divided_line[0] == "augmentative_path":
                    augmentative_path = Path(divided_line[1].strip('\n'))
                    augmentative_path = os.path.join(resources_dir, augmentative_path)
                elif divided_line[0] == "diminutive_path":
                    diminutive_path = Path(divided_line[1].strip('\n'))
                    diminutive_path = os.path.join(resources_dir, diminutive_path)
                elif divided_line[0] == "gender_path":
                    gender_path = Path(divided_line[1].strip('\n'))
                    gender_path = os.path.join(resources_dir, gender_path)
                elif divided_line[0] == "gender_name_path":
                    gender_name_path = Path(divided_line[1].strip('\n'))
                    gender_name_path = os.path.join(resources_dir, gender_name_path)
                elif divided_line[0] == "irregular_verb_path":
                    irregular_verb_path = Path(divided_line[1].strip('\n'))
                    irregular_verb_path = os.path.join(resources_dir, irregular_verb_path)
                elif divided_line[0] == "lexeme_verb_path":
                    lexeme_verb_path = Path(divided_line[1].strip('\n'))
                    lexeme_verb_path = os.path.join(resources_dir, lexeme_verb_path)
                elif divided_line[0] == "regular_verb_path":
                    regular_verb_path = Path(divided_line[1].strip('\n'))
                    regular_verb_path = os.path.join(resources_dir, regular_verb_path)
                elif divided_line[0] == "ranking_path":
                    ranking_path = Path(divided_line[1].strip('\n'))
                    ranking_path = os.path.join(resources_dir, ranking_path)
                elif divided_line[0] == "dict_path":
                    dict_path = Path(divided_line[1].strip('\n'))
                    dict_path = os.path.join(resources_dir, dict_path)
                elif divided_line[0] == "lexical_conversions":
                    lexical_conversions = divided_line[1].strip('\n')

                elif divided_line[0] == "dict_exclusions":
                    dict_exclusions = divided_line[1].strip('\n')

                elif divided_line[0] == "run_modules":
                    run_modules = divided_line[1].strip('\n')

                elif divided_line[0] == "break_on_hyphen":
                    break_on_hyphen = divided_line[1].strip('\n')

                elif divided_line[0] == "break_on_underscore":
                    break_on_underscore = divided_line[1].strip('\n')

                elif divided_line[0] == "dictionary_main_path":
                    dictionary_main_path = Path(divided_line[1].strip('\n'))
                    dictionary_main_path = os.path.join(resources_dir, dictionary_main_path)

                elif divided_line[0] == "custom_dictionary_path":
                    custom_dictionary_path = Path(divided_line[1].strip('\n'))
                    custom_dictionary_path = os.path.join(resources_dir, custom_dictionary_path)


def load_lematizador():
    adverb_norm = AdverbNormalizer()
    adverb_norm.adverb_normalizer_load(adverb_path)
    adverb_norm.declesion_exceptions = target_sorter(adverb_norm.declesion_targets, adverb_norm.declesion_exceptions)
    adverb_norm.declesion_tags = target_sorter(adverb_norm.declesion_targets, adverb_norm.declesion_tags)
    adverb_norm.declesion_replacement = target_sorter(adverb_norm.declesion_targets, adverb_norm.declesion_replacement)
    adverb_norm.declesion_targets = target_sorter(adverb_norm.declesion_targets, adverb_norm.declesion_targets)
    adverb_norm.compile_rules()

    number_norm = NumberNormalizer()
    number_norm.number_normalizer_load(number_path)
    number_norm.declesion_exceptions = target_sorter(number_norm.declesion_targets, number_norm.declesion_exceptions)
    number_norm.declesion_tags = target_sorter(number_norm.declesion_targets, number_norm.declesion_tags)
    number_norm.declesion_replacement = target_sorter(number_norm.declesion_targets, number_norm.declesion_replacement)
    number_norm.declesion_targets = target_sorter(number_norm.declesion_targets, number_norm.declesion_targets)
    number_norm.compile_rules()

    superlative_norm = SuperlativeNormalizer()
    superlative_norm.superlative_normalizer_load(superlative_path)
    superlative_norm.declesion_exceptions = target_sorter(superlative_norm.declesion_targets,
                                                          superlative_norm.declesion_exceptions)
    superlative_norm.declesion_tags = target_sorter(superlative_norm.declesion_targets, superlative_norm.declesion_tags)
    superlative_norm.declesion_replacement = target_sorter(superlative_norm.declesion_targets,
                                                           superlative_norm.declesion_replacement)
    superlative_norm.declesion_targets = target_sorter(superlative_norm.declesion_targets,
                                                       superlative_norm.declesion_targets)
    superlative_norm.compile_rules()

    augmentative_norm = AugmentativeNormalizer()
    augmentative_norm.augmentative_normalizer_load(augmentative_path)
    augmentative_norm.declesion_exceptions = target_sorter(augmentative_norm.declesion_targets,
                                                           augmentative_norm.declesion_exceptions)
    augmentative_norm.declesion_tags = target_sorter(augmentative_norm.declesion_targets,
                                                     augmentative_norm.declesion_tags)
    augmentative_norm.declesion_replacement = target_sorter(augmentative_norm.declesion_targets,
                                                            augmentative_norm.declesion_replacement)
    augmentative_norm.declesion_targets = target_sorter(augmentative_norm.declesion_targets,
                                                        augmentative_norm.declesion_targets)
    augmentative_norm.compile_rules()

    diminutive_norm = DiminutiveNormalizer()
    diminutive_norm.diminutive_normalizer_load(diminutive_path)
    diminutive_norm.declesion_exceptions = target_sorter(diminutive_norm.declesion_targets,
                                                         diminutive_norm.declesion_exceptions)
    diminutive_norm.declesion_tags = target_sorter(diminutive_norm.declesion_targets, diminutive_norm.declesion_tags)
    diminutive_norm.declesion_replacement = target_sorter(diminutive_norm.declesion_targets,
                                                          diminutive_norm.declesion_replacement)
    diminutive_norm.declesion_targets = target_sorter(diminutive_norm.declesion_targets,
                                                      diminutive_norm.declesion_targets)
    diminutive_norm.compile_rules()

    gender_norm = GenderNormalizer()
    gender_norm.gender_normalizer_load(gender_path)
    gender_norm.declesion_exceptions = target_sorter(gender_norm.declesion_targets, gender_norm.declesion_exceptions)
    gender_norm.declesion_tags = target_sorter(gender_norm.declesion_targets, gender_norm.declesion_tags)
    gender_norm.declesion_replacement = target_sorter(gender_norm.declesion_targets, gender_norm.declesion_replacement)
    gender_norm.declesion_targets = target_sorter(gender_norm.declesion_targets, gender_norm.declesion_targets)
    gender_norm.compile_rules()

    gender_name_norm = GenderNameNormalizer()
    gender_name_norm.gender_name_normalizer_load(gender_name_path)
    gender_name_norm.declesion_exceptions = target_sorter(gender_name_norm.declesion_targets,
                                                          gender_name_norm.declesion_exceptions)
    gender_name_norm.declesion_tags = target_sorter(gender_name_norm.declesion_targets, gender_name_norm.declesion_tags)
    gender_name_norm.declesion_replacement = target_sorter(gender_name_norm.declesion_targets,
                                                           gender_name_norm.declesion_replacement)
    gender_name_norm.declesion_targets = target_sorter(gender_name_norm.declesion_targets,
                                                       gender_name_norm.declesion_targets)

    verb_norm = VerbNormalizer()
    verb_norm.verb_normalizer_load(irregular_verb_path, lexeme_verb_path, regular_verb_path)

    verb_norm.declesion_exceptions = target_sorter(verb_norm.declesion_targets, verb_norm.declesion_exceptions)
    verb_norm.declesion_tags = target_sorter(verb_norm.declesion_targets, verb_norm.declesion_tags)
    verb_norm.declesion_replacement = target_sorter(verb_norm.declesion_targets, verb_norm.declesion_replacement)
    verb_norm.declesion_targets = target_sorter(verb_norm.declesion_targets, verb_norm.declesion_targets)

    verb_norm.lexeme_exceptions = target_sorter(verb_norm.lexeme_targets, verb_norm.lexeme_exceptions)
    verb_norm.lexeme_tags = target_sorter(verb_norm.lexeme_targets, verb_norm.lexeme_tags)
    verb_norm.lexeme_replacement = target_sorter(verb_norm.lexeme_targets, verb_norm.lexeme_replacement)
    verb_norm.lexeme_targets = target_sorter(verb_norm.lexeme_targets, verb_norm.lexeme_targets)

    verb_norm.conjugation_exceptions = target_sorter(verb_norm.conjugation_targets, verb_norm.conjugation_exceptions)
    verb_norm.conjugation_tags = target_sorter(verb_norm.conjugation_targets, verb_norm.conjugation_tags)
    verb_norm.conjugation_replacement = target_sorter(verb_norm.conjugation_targets, verb_norm.conjugation_replacement)
    verb_norm.conjugation_targets = target_sorter(verb_norm.conjugation_targets, verb_norm.conjugation_targets)

    verb_norm.compile_rules()

    # print("Lemmatizer load completed")

    ranking = WordRanking()
    ranking.load(ranking_path)

    novo_dict = Dictionary()
    novo_dict.load(dictionary_main_path)
    novo_dict.load(custom_dictionary_path)
    # ranking.load("Resources/acdc/formas.total.txt")

    # print("Dictionary load completed")

    return adverb_norm, number_norm, superlative_norm, augmentative_norm, diminutive_norm, gender_norm, gender_name_norm, verb_norm, ranking, novo_dict


def tags_classes():
    adverb_tags = "adv"
    augmentative_tags = "n|n-adj|adj"
    diminutive_tags = "n|n-adj|adj"
    gender_name_tags = "n|n-adj|adj"
    gender_tags = "art|pron|pron-pers|pron-det|pron-indp|n|n-adj|adj|num"
    number_tags = "n|n-adj|adj|art|pron|pron-pers|pron-det|pron-indp"
    superlative_tags = "n|n-adj|adj"
    verb_tags = "v|v-fin|v-ger|v-pcp|v-inf"

    return adverb_tags, augmentative_tags, diminutive_tags, gender_name_tags, gender_tags, number_tags, superlative_tags, verb_tags


def nlpyport_lematizer_loader(LemPort_config_file):
    get_paths_lematizador(LemPort_config_file)
    adverb_norm, number_norm, superlative_norm, augmentative_norm, diminutive_norm, gender_norm, gender_name_norm, verb_norm, ranking, novo_dict = load_lematizador()
    return adverb_norm, number_norm, superlative_norm, augmentative_norm, diminutive_norm, gender_norm, gender_name_norm, verb_norm, ranking, novo_dict


def nlpyport_lematizer_preload(token, tag, LemPort_config_file, adverb_norm, number_norm, superlative_norm,
                               augmentative_norm, diminutive_norm, gender_norm, gender_name_norm, verb_norm, ranking,
                               novo_dict):
    valor = []
    for i in range(len(token)):
        res = all_normalizations(adverb_norm, number_norm, superlative_norm, augmentative_norm, diminutive_norm,
                                 gender_norm, gender_name_norm, verb_norm, token[i].lower(), tag[i].lower(), ranking,
                                 novo_dict)
        valor.append(res)
    return valor


def nlpyport_lematizer(token, tag, LemPort_config_file):
    get_paths_lematizador(LemPort_config_file)
    adverb_norm, number_norm, superlative_norm, augmentative_norm, diminutive_norm, gender_norm, gender_name_norm, verb_norm, ranking, novo_dict = load_lematizador()
    valor = []
    for i in range(len(token)):
        res = all_normalizations(adverb_norm, number_norm, superlative_norm, augmentative_norm, diminutive_norm,
                                 gender_norm, gender_name_norm, verb_norm, token[i].lower(), tag[i].lower(), ranking,
                                 novo_dict)
        valor.append(res)
    return valor