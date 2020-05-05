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
from nltk.tokenize import word_tokenize, blankline_tokenize
from nltk.util import bigrams, trigrams, ngrams
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import time


path = Path("./fictions/")

def main():
    file_list = [f for f in listdir(path) if isfile(join(path, f))]

    # EXTRACT TEST
    book_list = []
    for file_name in file_list:
        correct_regex = re.match("\A\((\d*)\)\[(\d*)\] (.*)\.docx\Z",file_name)
        if not correct_regex:
            print(file_name, " is not loaded (regex problem)")
            return 1

        text = docx2txt.process(path / file_name)

        book = {
            "id"    : int(correct_regex.group(1)),
            "year"  : int(correct_regex.group(2)),
            "title" : correct_regex.group(3),
            "text"  : text
        }
        book_list.append(book)


    #ORDER DOCUMENTS
    def sortLogic(obj):
        return obj["id"]
    book_list.sort(key = sortLogic)



    # PREPROCESS TEXT

    # tokenization
    def preprocess_text(text):
        paragraphs = blankline_tokenize(text)
        tokens = word_tokenize(text)
        bigrams = list(nltk.bigrams(tokens))
        fdist = FreqDist(word.lower() for word in tokens)
        token_frequency = fdist.items()


        time.sleep(500)


    preprocess_text(book_list[0]["text"])



    for book in book_list:
        token = word_tokenize(book["text"])
        book["token"] = token









    #WORD CLOUD

    # Start with one review:
    #text = book_list[5]["text"]#df.description[0]
    #text = " ".join(review for review in df.description)
    text = ""
    for book in book_list:
        text = text + book["text"]
    print("There are {} words in the combination of all review.".format(len(text)))
    # Create and generate a word cloud image:
    #wordcloud = WordCloud().generate(text)
    wordcloud = WordCloud(max_words=30, background_color="white").generate(text)

    wordcloud.to_file("img/first_review.png")

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # Create stopword list:
    stopwords = set(STOPWORDS)
    stopwords.update([
        "wa",
        "hi",
        "was",
        "then",
        "Then",
        "seemed",
        "it",
        "it wa",
        "It wa",
        "whose",
        "almost",
        "still",
        "without",
        "long",
        "half"
    ])

    wordcloud = WordCloud(stopwords=stopwords, max_words=30, background_color="white").generate(text)

    wordcloud.to_file("img/refined_review.png")

    # Display the generated image:
    #plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()




main()