
import os
import nltk
#import nltk.corpus
import numpy as np
import pandas as pd
from PIL import Image
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

from pathlib import Path
from os import listdir
from os.path import isfile, join
import docx2txt
import re
from os import path
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import word_tokenize, blankline_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.util import bigrams, trigrams, ngrams
from nltk.probability import FreqDist
from nltk import ne_chunk
import matplotlib.pyplot as plt
import time
import config
import janitor as jn
import tf_idf
import spacy
from spacy import displacy

def cut_title(text):
    indx = text.find("Lovecraft") + len("Lovecraft")
    eol = True
    while eol:
        if text[indx] != "\n" and text[indx] != " ":
            eol = False
        else:
            indx += 1
    return text[indx:]



def extract_books(path):
    file_list = [f for f in listdir(path) if isfile(join(path, f))]
    list = []
    for file_name in file_list:
        correct_regex = re.match("\A\((\d*)\)\[(\d*)\] (.*)\.docx\Z", file_name)
        if not correct_regex:
            print(file_name, " is not loaded (regex problem)")
            return 1

        text = docx2txt.process(path / file_name)

        book = {
            "id": int(correct_regex.group(1)),
            "year": int(correct_regex.group(2)),
            "title": correct_regex.group(3),
            "original": {
                "text": text
            }
        }

        # cut title
        book["original"]["text"] = cut_title(book["original"]["text"])
        list.append(book)

    # ORDER DOCUMENTS
    def sortLogic(obj):
        return obj["id"]

    list.sort(key=sortLogic)

    return list


def main():

    start_time = time.time()

    #stopwords_txt = set(open(config.general['stopwords']).read().split())
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords.update(set(STOPWORDS))

    pickle_book_file = config.general["pickle_path"] / "book_list.pickle"

    # try to load pickel
    book_list = jn.pickle_load(pickle_book_file)

    # if there is no pickels extract corpus
    if book_list == None:
        if not jn.create_dir(config.general["pickle_path"]):
            return 1

        book_list = extract_books(config.general["corpus_path"])

        # cut title, add paragraphs and sentences
        for book in book_list:
            book["original"]["tokens"] = word_tokenize(book["original"]["text"])
            book["original"]["paragraphs"] = blankline_tokenize(book["original"]["text"])
            book["original"]["sentences"] = sent_tokenize(book["original"]["text"])

        # add tokens bigrams and trigrams
        for book in book_list:
            token_list, bigram_list, trigram_list = [], [], []
            for sntc in book["original"]["sentences"]:
                tokens = word_tokenize(sntc)
                bigrams = list(nltk.bigrams(tokens))
                trigrams = list(nltk.trigrams(tokens))
                token_list.append(tokens)
                bigram_list.append(bigrams)
                trigram_list.append(trigrams)

            book["original"]["token_list"] = token_list
            book["original"]["bigram_list"] = bigram_list
            book["original"]["trigram_list"] = trigram_list

        #for word in book.lower().split():
        #preprocessed text
        punctuation = re.compile(r'[.,?!:;()|0-9]') #-
        for book in book_list:
            preprocessed_sentences = []
            preprocessed_token_list, preprocessed_bigram_list, preprocessed_trigram_list = [], [], []
            cleaned_token_list = []

            new_text = re.sub(r'[^\w\s]', '', book["original"]["text"])
            preprocessed_text = new_text.lower()

            for sntc_tokenized in book["original"]["sentences"]:
                new_sentence = re.sub(r'[^\w\s]', '', sntc_tokenized)
                new_sentence = new_sentence.lower()

                new_tokens = word_tokenize(new_sentence)
                new_bigrams = list(nltk.bigrams(new_tokens))
                new_trigrams = list(nltk.trigrams(new_tokens))

                cleaned_tokens = []
                for word in new_tokens:
                    if word not in stopwords:
                        cleaned_tokens.append(word)

                preprocessed_sentences.append(new_sentence)
                preprocessed_token_list.append(new_tokens)
                preprocessed_bigram_list.append(new_bigrams)
                preprocessed_trigram_list.append(new_trigrams)
                cleaned_token_list.append(cleaned_tokens)
            book["original"]["tokens"] = word_tokenize(book["original"]["text"])
            book["preprocess"] = {
                "text": preprocessed_text,
                "tokens": word_tokenize(preprocessed_text),
                "sentences": preprocessed_sentences,
                "token_list": preprocessed_token_list,
                "brigram_list": preprocessed_bigram_list,
                "trigram_list": preprocessed_trigram_list,
                "cleaned_tokens": cleaned_token_list
            }

        # add word freq

        for book in book_list:
            fdist_original = FreqDist(word for word in word_tokenize(book["original"]["text"]))
            book["original"]["token_frequency"] = dict(fdist_original.items())

            fdist_preprocess = FreqDist(word.lower() for word in word_tokenize(book["preprocess"]["text"]))
            book["preprocess"]["token_frequency"] = dict(fdist_preprocess.items())



        jn.pickle_save(book_list, pickle_book_file)

    # TF.IDF
    pickle_tfidf_file = config.general["pickle_path"] / "tf_idf_dictionary.pickle"
    # try to load pickel
    tf_idf_dictionary = jn.pickle_load(pickle_tfidf_file)

    # if there is no pickels extract corpus
    if tf_idf_dictionary == None:

        token_set = [book["preprocess"]["tokens"] for book in book_list]
        tf_idf_dictionary = tf_idf.get_tf_idf(token_set)
        jn.pickle_save(tf_idf_dictionary, pickle_tfidf_file)

    # TF.IDF without stopwords
    pickle_tfidf_nsw_file = config.general["pickle_path"] / "tf_idf_dictionary_nsw.pickle"
    # try to load pickel
    tf_idf_dictionary_nsw = jn.pickle_load(pickle_tfidf_nsw_file)

    # if there is no pickels extract corpus
    if tf_idf_dictionary_nsw == None:

        token_set = [book["preprocess"]["tokens"] for book in book_list]

        token_set_nsw = []
        for tokens in token_set:
            tokens_nsw = []
            for word in tokens:
                if word not in stopwords:
                    tokens_nsw.append(word)
            token_set_nsw.append(tokens_nsw)

        tf_idf_dictionary_nsw = tf_idf.get_tf_idf(token_set_nsw)
        jn.pickle_save(tf_idf_dictionary_nsw, pickle_tfidf_nsw_file)

    print("--- Preprocessing lasts %s seconds ---" % (time.time() - start_time))



    atmom = book_list[56]
    dagon = book_list[1]

    asd = tf_idf_dictionary_nsw[1]

    asd = {k: v for k, v in asd.items() if v > 0.002}

    text = dagon["original"]["text"]
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    displacy.serve(doc, style="ent")  #
    return 1
    tkn = atmom["preprocess"]["token_list"][0]
    # POS
    hey = nltk.pos_tag(tkn)
    # NER
    hoy = ne_chunk(hey)

    tkn2 = atmom["original"]["token_list"][0]
    # POS
    hey2 = nltk.pos_tag(tkn2)
    # NER
    hoy2 = ne_chunk(hey2)

    print(hey)
    print(hoy)
    print(hey2)
    print(hoy2)
    asd = 1
    #pst = LancasterStemmer()
    #print(atmom["sentences"][0])
    #print(pst.stem(atmom["sentences"][0]))

    q1 = "The big cat ate the little mouse who was after fresh cheese"
    nw_tk = nltk.pos_tag(word_tokenize(q1))
    print(nw_tk)

    grammar_np = r"NP: {<DT>?<JJ>*<NN>}"
    chunk_parser = nltk.RegexpParser(grammar_np)
    chunk_result = chunk_parser.parse(nw_tk)
    print(chunk_result)
    return 1
    #data = [
    #    [(word.replace(",", "")
    #      .replace(".", "")
    #      .replace("(", "")
    #      .replace(")", ""))
    #     for word in row[2].lower().split()]
    #    for row in reader]

    ## Removes header
    #data = data[1:]


    all_sentences = ""
    all_preprocessed_sentences = ""
    for book in book_list:
        for sntc in book["original"]["sentences"]:
            all_sentences = all_sentences + "\n" + sntc

        for sntc in book["preprocess"]["sentences"]:
            all_preprocessed_sentences = all_preprocessed_sentences + "\n" + sntc


    print("There are {} words in the combination of all review.".format(len(all_sentences)))

    # Create and generate a word cloud image:
    #wordcloud = WordCloud().generate(text)
    #wordcloud = WordCloud(max_words=30, background_color="white", collocations=False).generate(text)

    #wordcloud.to_file("img/first_review.png")

    #plt.imshow(wordcloud, interpolation='bilinear')
    #plt.axis("off")
    #plt.show()

    wordcloud = WordCloud(stopwords=stopwords, max_words=50, background_color="white", collocations=False).generate(all_sentences)

    wordcloud.to_file("img/review.png")

    # Display the generated image:
    #plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


    wordcloud = WordCloud(stopwords=stopwords, max_words=50, background_color="white", collocations=False).generate(all_preprocessed_sentences)

    wordcloud.to_file("img/refined_review.png")

    # Display the generated image:
    #plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


main()