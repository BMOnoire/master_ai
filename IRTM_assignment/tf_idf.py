import math

def computeReviewTFDict(review):
    """ Returns a tf dictionary for each review whose keys are all
    the unique words in the review and whose values are their
    corresponding tf.
    """

    reviewTFDict = {}
    for word in review:
        if word in reviewTFDict:
            reviewTFDict[word] += 1
        else:
            reviewTFDict[word] = 1

    for word in reviewTFDict:
        reviewTFDict[word] = reviewTFDict[word] / len(review)
    return reviewTFDict


def computeCountDict(tfDict):
    """ Returns a dictionary whose keys are all the unique words in
    the dataset and whose values count the number of reviews in which
    the word appears.
    """
    countDict = {}

    for review in tfDict:
        for word in review:
            if word in countDict:
                countDict[word] += 1
            else:
                countDict[word] = 1
    return countDict

def computeIDFDict(countDict, data_size):
    """ Returns a dictionary whose keys are all the unique words in the
    dataset and whose values are their corresponding idf.
    """
    idfDict = {}
    for word in countDict:
        idfDict[word] = math.log(data_size / countDict[word])
    return idfDict


def computeReviewTFIDFDict(reviewTFDict, idfDict):
    """ Returns a dictionary whose keys are all the unique words in the
    review and whose values are their corresponding tfidf.
    """
    reviewTFIDFDict = {}
    #For each word in the review, we multiply its tf and its idf.
    for word in reviewTFDict:
        reviewTFIDFDict[word] = reviewTFDict[word] * idfDict[word]
    return reviewTFIDFDict


def get_tf_idf(token_set):
    # Counts the number of times the word appears in a book and return tf for each word
    reviewTFDict = []
    for tokens in token_set:
        tf = computeReviewTFDict(tokens)
        reviewTFDict.append(tf)

    # Run through each review's tf dictionary and increment countDict's (word, doc) pair
    countDict  = computeCountDict(reviewTFDict)

    #Returns a dictionary whose keys are all the unique words in the dataset
    idfDict = computeIDFDict(countDict, len(token_set))

    tf_idf_dict = [computeReviewTFIDFDict(review, idfDict) for review in reviewTFDict]
    return tf_idf_dict