r"""
Core Concepts
=============

This tutorial introduces Documents, Corpora, Vectors and Models: the basic concepts and terms needed to understand and use gensim.
"""

# mock data format - create corpus map
# have tagged/training corpus and use untagged corpus as query
# 
import json
# read corpus file
#  tagged map
tagged_corpus = open('data/input_tagged_corpus_id.json', 'r')
tagged_corpus = json.load(tagged_corpus)

#  untagged map

untagged_corpus = open('data/input_untagged_corpus_id.json', 'r')
untagged_corpus = json.load(untagged_corpus)

# create topic map

# remove stoplist
# remove infrequent words
# word to int id map using dictionary
# convert corpus to list of vectors
# train tfidf model with the bow corpus
# find similarity between desc
# pick the document with most similarity and add them to topic map
#  add tag to corpus
#  if no sim found, use topic 
# create speaker map


import pprint
from gensim.parsing.preprocessing import remove_stopwords
# Create a set of frequent words
# Lowercase each document, split it by white space and filter out stopwords
texts = [remove_stopwords(seminars["description"].lower()).split() for seminars in tagged_corpus]
texts = [[token for token in text if token.isalpha() and len(token) >=3 ] for text in texts]

# print(texts)

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 3] for text in texts]
# pprint.pprint(processed_corpus)

topic_map = {}
#  map where keys represent the tags in each document and its corresponding document
#  example:
#  {'engineering': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11', '12', '13', '14', '15', '17', '19'], 'teaching': ['10', '16', '18', '20']}
for seminars in tagged_corpus: 
    if seminars["tag"].lower() not in topic_map:
        topic_map[seminars["tag"].lower()] = []
    # else:
    #     topic_map[seminars["tag"].lower()].append(seminars["id"])


speaker_map = {}
#  map where keys represent the speaker name in each document and its corresponding document
#  example: {'bennett harrison': [1], 'urmila m. diwekar': [2], 'bill wescott': [3], 'mary jane': [4], 'jim boshears': [5], 'andrew c. barrett': [6], 'erik devereux': [7]}
for seminars in tagged_corpus: 
    if seminars["speaker"].lower() not in speaker_map:
        speaker_map[seminars["speaker"].lower()] = [seminars["id"]]
    else:
        speaker_map[seminars["speaker"].lower()].append(seminars["id"])

###############################################################################
# Before proceeding, we want to associate each word in the corpus with a unique
# integer ID. We can do this using the :py:class:`gensim.corpora.Dictionary`
# class.  This dictionary defines the vocabulary of all words that our
# processing knows about.
#
from gensim import corpora

dictionary = corpora.Dictionary(processed_corpus)
# print(dictionary)

###############################################################################
# Because our corpus is small, there are only 12 different tokens in this
# :py:class:`gensim.corpora.Dictionary`. For larger corpuses, dictionaries that
# contains hundreds of thousands of tokens are quite common.
#

###############################################################################
# .. _core_concepts_vector:
#
# Vector
# ------
#
# To infer the latent structure in our corpus we need a way to represent
# documents that we can manipulate mathematically. One approach is to represent
# each document as a vector of *features*.
# For example, a single feature may be thought of as a question-answer pair:
#
# 1. How many times does the word *splonge* appear in the document? Zero.
# 2. How many paragraphs does the document consist of? Two.
# 3. How many fonts does the document use? Five.
#
# The question is usually represented only by its integer id (such as `1`, `2` and `3`).
# The representation of this document then becomes a series of pairs like ``(1, 0.0), (2, 2.0), (3, 5.0)``.
# This is known as a *dense vector*, because it contains an explicit answer to each of the above questions.
#
# If we know all the questions in advance, we may leave them implicit
# and simply represent the document as ``(0, 2, 5)``.
# This sequence of answers is the **vector** for our document (in this case a 3-dimensional dense vector).
# For practical purposes, only questions to which the answer is (or
# can be converted to) a *single floating point number* are allowed in Gensim.
#
# In practice, vectors often consist of many zero values.
# To save memory, Gensim omits all vector elements with value 0.0.
# The above example thus becomes ``(2, 2.0), (3, 5.0)``.
# This is known as a *sparse vector* or *bag-of-words vector*.
# The values of all missing features in this sparse representation can be unambiguously resolved to zero, ``0.0``.
#
# Assuming the questions are the same, we can compare the vectors of two different documents to each other.
# For example, assume we are given two vectors ``(0.0, 2.0, 5.0)`` and ``(0.1, 1.9, 4.9)``.
# Because the vectors are very similar to each other, we can conclude that the documents corresponding to those vectors are similar, too.
# Of course, the correctness of that conclusion depends on how well we picked the questions in the first place.
#
# Another approach to represent a document as a vector is the *bag-of-words
# model*.
# Under the bag-of-words model each document is represented by a vector
# containing the frequency counts of each word in the dictionary.
# For example, assume we have a dictionary containing the words
# ``['coffee', 'milk', 'sugar', 'spoon']``.
# A document consisting of the string ``"coffee milk coffee"`` would then
# be represented by the vector ``[2, 1, 0, 0]`` where the entries of the vector
# are (in order) the occurrences of "coffee", "milk", "sugar" and "spoon" in
# the document. The length of the vector is the number of entries in the
# dictionary. One of the main properties of the bag-of-words model is that it
# completely ignores the order of the tokens in the document that is encoded,
# which is where the name bag-of-words comes from.
#
# Our processed corpus has 12 unique words in it, which means that each
# document will be represented by a 12-dimensional vector under the
# bag-of-words model. We can use the dictionary to turn tokenized documents
# into these 12-dimensional vectors. We can see what these IDs correspond to:
#
# pprint.pprint(dictionary.token2id)

###############################################################################
# For example, suppose we wanted to vectorize the phrase "Human computer
# interaction" (note that this phrase was not in our original corpus). We can
# create the bag-of-word representation for a document using the ``doc2bow``
# method of the dictionary, which returns a sparse representation of the word
# counts:
#

# new_doc = "Human computer interaction"
# new_vec = dictionary.doc2bow(new_doc.lower().split())
# print(new_vec)

###############################################################################
# The first entry in each tuple corresponds to the ID of the token in the
# dictionary, the second corresponds to the count of this token.
#
# Note that "interaction" did not occur in the original corpus and so it was
# not included in the vectorization. Also note that this vector only contains
# entries for words that actually appeared in the document. Because any given
# document will only contain a few words out of the many words in the
# dictionary, words that do not appear in the vectorization are represented as
# implicitly zero as a space saving measure.
#
# We can convert our entire original corpus to a list of vectors:
#
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
# pprint.pprint(bow_corpus)

###############################################################################
# Note that while this list lives entirely in memory, in most applications you
# will want a more scalable solution. Luckily, ``gensim`` allows you to use any
# iterator that returns a single document vector at a time. See the
# documentation for more details.
#
# .. Important::
#   The distinction between a document and a vector is that the former is text,
#   and the latter is a mathematically convenient representation of the text.
#   Sometimes, people will use the terms interchangeably: for example, given
#   some arbitrary document ``D``, instead of saying "the vector that
#   corresponds to document ``D``", they will just say "the vector ``D``" or
#   the "document ``D``".  This achieves brevity at the cost of ambiguity.
#
#   As long as you remember that documents exist in document space, and that
#   vectors exist in vector space, the above ambiguity is acceptable.
#
# .. Important::
#   Depending on how the representation was obtained, two different documents
#   may have the same vector representations.
#
# .. _core_concepts_model:
#
# Model
# -----
#
# Now that we have vectorized our corpus we can begin to transform it using
# *models*. We use model as an abstract term referring to a *transformation* from
# one document representation to another. In ``gensim`` documents are
# represented as vectors so a model can be thought of as a transformation
# between two vector spaces. The model learns the details of this
# transformation during training, when it reads the training
# :ref:`core_concepts_corpus`.
#
# One simple example of a model is `tf-idf
# <https://en.wikipedia.org/wiki/Tf%E2%80%93idf>`_.  The tf-idf model
# transforms vectors from the bag-of-words representation to a vector space
# where the frequency counts are weighted according to the relative rarity of
# each word in the corpus.
#
# Here's a simple example. Let's initialize the tf-idf model, training it on
# our corpus and transforming the string "system minors":
#

from gensim import models

# train the model
tfidf = models.TfidfModel(bow_corpus)

# transform the "system minors" string
# words = "system minors".lower().split()
# print(tfidf[dictionary.doc2bow(words)])

###############################################################################
# The ``tfidf`` model again returns a list of tuples, where the first entry is
# the token ID and the second entry is the tf-idf weighting. Note that the ID
# corresponding to "system" (which occurred 4 times in the original corpus) has
# been weighted lower than the ID corresponding to "minors" (which only
# occurred twice).
#
# You can save trained models to disk and later load them back, either to
# continue training on new training documents or to transform new documents.
#
# ``gensim`` offers a number of different models/transformations.
# For more, see :ref:`sphx_glr_auto_examples_core_run_topics_and_transformations.py`.
#
# Once you've created the model, you can do all sorts of cool stuff with it.
# For example, to transform the whole corpus via TfIdf and index it, in
# preparation for similarity queries:
#
from gensim import similarities


###############################################################################
# and to query the similarity of our query document ``query_document`` against every document in the corpus:
# run a loop for seminars and find similar seminars add them to topic map

# query_document ="The Center for Cultural Analysis will host a lecture by Richard Maddox entitled 'The Best of Possible Islands: Seville, Expo '92, and the Politics of Culture in the 'New Spain', at 3:30 p.m., Friday, March 17, in Baker Hall 235A.All are welcome"
# # print(query_document)
# query_bow = dictionary.doc2bow(query_document.lower().split())
# sims = index[tfidf[query_bow]]
# print(list(enumerate(sims)))


# index = similarities.MatrixSimilarity(lsi[corpus])
f = open("output/output_tf_idf.txt", "w")
for seminars in untagged_corpus:
    doc = seminars["description"]
    vec_bow = dictionary.doc2bow(doc.lower().split())
    index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=300)
    sims = index[tfidf[vec_bow]]
    f.write(str(seminars["id"]) +',  ')
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    count = 0
    for doc_position, doc_score in sims:
        if count == 0:
            f.write('(' + str(doc_score) + ', ' + str(tagged_corpus[doc_position]["tag"]) + ', ' + str(seminars["tag"]) + ')')
            f.write("\n")
            topic_map[str(tagged_corpus[doc_position]["tag"]).lower()].append(seminars["id"])
        count += 1
f.write('\n' + str(topic_map))
f.write('\n' + str(speaker_map))



# Evaluation metrics
#  Using average precision and mean average precision over 3 users.

user_preference = [
    {
        'user_id' : 1,
        'seminars_attended' : 65
    },
    {
        'user_id' : 2,
        'seminars_attended' : 106
    },
    {
        'user_id' : 3,
        'seminars_attended' : 4
    }
]

#  run the model to find top k = 3 recommendations
f = open("output/top_k_tf_idf.txt", "w")
# merged_corpus = tagged_corpus + untagged_corpus

for users in user_preference:
    doc = ''
    for seminar in tagged_corpus:
        if seminar['id'] == users['seminars_attended']:
            doc = seminar['description']
            seminars = seminar
    vec_bow = dictionary.doc2bow(doc.lower().split())
    index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=300)
    sims = index[tfidf[vec_bow]]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    count = 0
    f.write(str(users['user_id']) + ' ' + str(users['seminars_attended'])  + '\n')
    for doc_position, doc_score in sims:
        if count < 11 and tagged_corpus[doc_position]["id"] != users['seminars_attended']:
            f.write('(' + str(doc_score) + ', ' + str(tagged_corpus[doc_position]["tag"]) + ', ' + str(seminars["tag"]) + ',' + str(tagged_corpus[doc_position]["id"]) + ')')
            f.write("\n")
            # topic_map[str(tagged_corpus[doc_position]["tag"]).lower()].append(seminars["id"])
        count += 1
# f.write('\n' + str(topic_map))
# f.write('\n' + str(speaker_map))


###############################################################################
# How to read this output?
# Document 3 has a similarity score of 0.718=72%, document 2 has a similarity score of 42% etc.
# We can make this slightly more readable by sorting:

# for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
#     print(document_number, score)

###############################################################################
# Summary
# -------
#
# The core concepts of ``gensim`` are:
#
# 1. :ref:`core_concepts_document`: some text.
# 2. :ref:`core_concepts_corpus`: a collection of documents.
# 3. :ref:`core_concepts_vector`: a mathematically convenient representation of a document.
# 4. :ref:`core_concepts_model`: an algorithm for transforming vectors from one representation to another.
#
# We saw these concepts in action.
# First, we started with a corpus of documents.
# Next, we transformed these documents to a vector space representation.
# After that, we created a model that transformed our original vector representation to TfIdf.
# Finally, we used our model to calculate the similarity between some query document and all documents in the corpus.
#
# What Next?
# ----------
#
# There's still much more to learn about :ref:`sphx_glr_auto_examples_core_run_corpora_and_vector_spaces.py`.

# # import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# img = mpimg.imread('run_core_concepts.png')
# imgplot = plt.imshow(img)
# _ = plt.axis('off')
