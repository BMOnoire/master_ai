import os
import nltk
import nltk.corpus
import numpy as np
import pandas as pd
from PIL import Image

from pathlib import Path
from os import listdir
from os.path import isfile, join
import docx2txt
import re
from os import path
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import word_tokenize, blankline_tokenize, sent_tokenize
from nltk.util import bigrams, trigrams, ngrams
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import time
import pickle
import config

def create_dir(path):
    path = str(path)
    if os.path.exists(path):
        print("Directory %s exists" % path)
        return True
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
        return False
    else:
        print("Successfully created the directory %s " % path)
        return True


def picke_load(pickle_path):
    if pickle_path.exists():
        pickle_in = open(str(pickle_path), "rb")
        return pickle.load(pickle_in)
    else:
        return None


def pickle_save(object, pickle_path):
    pickle_out = open(str(pickle_path), "wb")
    pickle.dump(object, pickle_out)
    pickle_out.close()


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
            "text": text
        }
        list.append(book)

    # ORDER DOCUMENTS
    def sortLogic(obj):
        return obj["id"]

    list.sort(key=sortLogic)

    return list


def split_paragraphs_and_sentences(book):
        text = book["text"]
        indx = text.find("Lovecraft") + len("Lovecraft")
        eol = True
        while eol:
            if text[indx] != "\n" and text[indx] != " ":
                eol = False
            else:
                indx += 1
        text = text[indx:]
        book["text"] = text
        book["paragraphs"] = blankline_tokenize(text)
        book["sentences"] = sent_tokenize(text)


def tokenize_sentence(book):
    token_list, bigram_list, trigram_list = [], [], []
    for i, sntc in enumerate(book["sentences"]):
        tokens = word_tokenize(sntc)
        bigrams = list(nltk.bigrams(tokens))
        trigrams = list(nltk.trigrams(tokens))
        token_list.append(tokens)
        bigram_list.append(bigrams)
        trigram_list.append(trigrams)

    fdist = FreqDist(word.lower() for word in word_tokenize(book["text"]))
    all_token_frequency = list(fdist.items())

    book["token_list"] = token_list
    book["bigram_list"] = bigram_list
    book["trigram_list"] = trigram_list
    book["token_frequency"] = all_token_frequency


def preprocessing_corpus(book_list):
    for book in book_list:
        split_paragraphs_and_sentences(book)

    for book in book_list:
        tokenize_sentence(book)

    return book_list


def main():
    start_time = time.time()

    pickle_file = config.general["pickle_path"] / "book_list.pickle"

    # try to load pickel
    book_list = picke_load(pickle_file)

    # if there is no pickels extract corpus
    if book_list == None:
        if not create_dir(config.general["pickle_path"]):
            return 1

        book_list = extract_books(config.general["corpus_path"])

        book_list = preprocessing_corpus(book_list)

        pickle_save(book_list, pickle_file)

    print("--- %s seconds ---" % (time.time() - start_time))


    #WORD CLOUD
    all_text = ""
    for book in book_list:
        all_text = all_text + book["text"]
    print("There are {} words in the combination of all review.".format(len(all_text)))

    # Create and generate a word cloud image:
    #wordcloud = WordCloud().generate(text)
    #wordcloud = WordCloud(max_words=30, background_color="white", collocations=False).generate(text)

    #wordcloud.to_file("img/first_review.png")

    #plt.imshow(wordcloud, interpolation='bilinear')
    #plt.axis("off")
    #plt.show()

    # Create stopword list:
    stopwords = set(STOPWORDS)
    stopwords.update([
        "hi",
        "was",
        "then",
        "Then",
        "seemed",
        "it",
        "wa",
        "whose",
        "almost",
        "still",
        "without",
        "long",
        "half",
        "day",
        "must",
        "thought",
        "came",
        "much",
        "now",
        "found",
        "made",
        "year",
        "men",
        "many",
        "saw",
        "even",
        "certain",
        "thought",
        "upon",
        "first",
        "heard",
        "see",
        "know",
        "seen",
        "come",
        "might",
        "new",
        "last",
        "Street",
        "knew",
        "way",
        "tell",
        "Told",
        " told",
        "told",
        "told "
        "yet",
        "say",
        "said",
        "never",
        "two",
        "well",
        "light",
        "will",
        "little"
    ])

    wordcloud = WordCloud(stopwords=stopwords, max_words=50, background_color="white", collocations=False).generate(all_text)

    wordcloud.to_file("img/refined_review.png")

    # Display the generated image:
    #plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


main()