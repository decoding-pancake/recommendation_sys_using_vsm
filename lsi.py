r"""
Similarity Queries
==================

Demonstrates querying a corpus for similar documents.
"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

###############################################################################
#
# Creating the Corpus
# -------------------
#
# First, we need to create a corpus to work with.
# This step is the same as in the previous tutorial;
# if you completed it, feel free to skip to the next section.

from collections import defaultdict
from gensim import corpora
import json
# read corpus file
#  tagged map
tagged_corpus = open('data/input_tagged_corpus_id.json', 'r')
tagged_corpus = json.load(tagged_corpus)

#  untagged map
untagged_corpus = open('data/input_untagged_corpus_id.json', 'r')
untagged_corpus = json.load(untagged_corpus)


# remove common words and tokenize
from gensim.parsing.preprocessing import remove_stopwords
texts = [remove_stopwords(seminars["description"].lower()).split() for seminars in tagged_corpus]
texts = [[token for token in text if token.isalpha()] for text in texts]


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

# remove words that appear only once
# frequency = defaultdict(int)
# for text in texts:
#     for token in text:
#         frequency[token] += 1

# texts = [
#     [token for token in text if frequency[token] > 0]
#     for text in texts
# ]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

###############################################################################
# Similarity interface
# --------------------
#
# In the previous tutorials on
# :ref:`sphx_glr_auto_examples_core_run_corpora_and_vector_spaces.py`
# and
# :ref:`sphx_glr_auto_examples_core_run_topics_and_transformations.py`,
# we covered what it means to create a corpus in the Vector Space Model and how
# to transform it between different vector spaces. A common reason for such a
# charade is that we want to determine **similarity between pairs of
# documents**, or the **similarity between a specific document and a set of
# other documents** (such as a user query vs. indexed documents).
#
# To show how this can be done in gensim, let us consider the same corpus as in the
# previous examples (which really originally comes from Deerwester et al.'s
# `"Indexing by Latent Semantic Analysis" <http://www.cs.bham.ac.uk/~pxt/IDA/lsa_ind.pdf>`_
# seminal 1990 article).
# To follow Deerwester's example, we first use this tiny corpus to define a 2-dimensional
# LSI space:

from gensim import models
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=100)

###############################################################################
# For the purposes of this tutorial, there are only two things you need to know about LSI.
# First, it's just another transformation: it transforms vectors from one space to another.
# Second, the benefit of LSI is that enables identifying patterns and relationships between terms (in our case, words in a document) and topics.
# Our LSI space is two-dimensional (`num_topics = 2`) so there are two topics, but this is arbitrary.
# If you're interested, you can read more about LSI here: `Latent Semantic Indexing <https://en.wikipedia.org/wiki/Latent_semantic_indexing>`_:
#
# Now suppose a user typed in the query `"Human computer interaction"`. We would
# like to sort our nine corpus documents in decreasing order of relevance to this query.
# Unlike modern search engines, here we only concentrate on a single aspect of possible
# similarities---on apparent semantic relatedness of their texts (words). No hyperlinks,
# no random-walk static ranks, just a semantic extension over the boolean keyword match:
from gensim import similarities
# index = similarities.MatrixSimilarity(lsi[corpus])
f = open("output/output.txt", "w")
for seminars in untagged_corpus:
    doc = seminars["description"]
    vec_bow = dictionary.doc2bow(doc.lower().split())
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    index = similarities.MatrixSimilarity(lsi[corpus])
    index.save('/tmp/deerwester.index')
    index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')
    sims = index[vec_lsi]
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

#
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
f = open("output/top_k_lsi.txt", "w")
# merged_corpus = tagged_corpus + untagged_corpus

for users in user_preference:
    doc = ''
    for seminar in tagged_corpus:
        if seminar['id'] == users['seminars_attended']:
            doc = seminar['description']
            seminars = seminar
    vec_bow = dictionary.doc2bow(doc.lower().split())
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    index = similarities.MatrixSimilarity(lsi[corpus])
    index.save('/tmp/deerwester.index')
    index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    f.write(str(users['user_id']) + ' ' + str(users['seminars_attended'])  + '\n')
    count = 0
    for doc_position, doc_score in sims:
        if count < 11 and tagged_corpus[doc_position]["id"] != users['seminars_attended']:
            f.write('(' + str(doc_score) + ', ' + str(tagged_corpus[doc_position]["tag"]) + ', ' + str(seminars["tag"]) + ',' + str(tagged_corpus[doc_position]["id"]) + ')')
            f.write("\n")
            # topic_map[str(tagged_corpus[doc_position]["tag"]).lower()].append(seminars["id"])
        count += 1

###############################################################################
# In addition, we will be considering `cosine similarity <http://en.wikipedia.org/wiki/Cosine_similarity>`_
# to determine the similarity of two vectors. Cosine similarity is a standard measure
# in Vector Space Modeling, but wherever the vectors represent probability distributions,
# `different similarity measures <http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence>`_
# may be more appropriate.
#
# Initializing query structures
# ++++++++++++++++++++++++++++++++
#
# To prepare for similarity queries, we need to enter all documents which we want
# to compare against subsequent queries. In our case, they are the same nine documents
# used for training LSI, converted to 2-D LSA space. But that's only incidental, we
# might also be indexing a different corpus altogether.


# index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it

###############################################################################
# .. warning::
#   The class :class:`similarities.MatrixSimilarity` is only appropriate when the whole
#   set of vectors fits into memory. For example, a corpus of one million documents
#   would require 2GB of RAM in a 256-dimensional LSI space, when used with this class.
#
#   Without 2GB of free RAM, you would need to use the :class:`similarities.Similarity` class.
#   This class operates in fixed memory, by splitting the index across multiple files on disk, called shards.
#   It uses :class:`similarities.MatrixSimilarity` and :class:`similarities.SparseMatrixSimilarity` internally,
#   so it is still fast, although slightly more complex.
#
# Index persistency is handled via the standard :func:`save` and :func:`load` functions:



###############################################################################
# This is true for all similarity indexing classes (:class:`similarities.Similarity`,
# :class:`similarities.MatrixSimilarity` and :class:`similarities.SparseMatrixSimilarity`).
# Also in the following, `index` can be an object of any of these. When in doubt,
# use :class:`similarities.Similarity`, as it is the most scalable version, and it also
# supports adding more documents to the index later.
#
# Performing queries
# ++++++++++++++++++
#
# To obtain similarities of our query document against the nine indexed documents:

# sims = index[vec_lsi]  # perform a similarity query against the corpus
# print(list(enumerate(sims)))  # print (document_number, document_similarity) 2-tuples

###############################################################################
# Cosine measure returns similarities in the range `<-1, 1>` (the greater, the more similar),
# so that the first document has a score of 0.99809301 etc.
#
# With some standard Python magic we sort these similarities into descending
# order, and obtain the final answer to the query `"Human computer interaction"`:

# sims = sorted(enumerate(sims), key=lambda item: -item[1])
# for doc_position, doc_score in sims:
#     print(doc_score, documents[doc_position])

###############################################################################
# The thing to note here is that documents no. 2 (``"The EPS user interface management system"``)
# and 4 (``"Relation of user perceived response time to error measurement"``) would never be returned by
# a standard boolean fulltext search, because they do not share any common words with ``"Human
# computer interaction"``. However, after applying LSI, we can observe that both of
# them received quite high similarity scores (no. 2 is actually the most similar!),
# which corresponds better to our intuition of
# them sharing a "computer-human" related topic with the query. In fact, this semantic
# generalization is the reason why we apply transformations and do topic modelling
# in the first place.
#
# Where next?
# ------------
#
# Congratulations, you have finished the tutorials -- now you know how gensim works :-)
# To delve into more details, you can browse through the :ref:`apiref`,
# see the :ref:`wiki` or perhaps check out :ref:`distributed` in `gensim`.
#
# Gensim is a fairly mature package that has been used successfully by many individuals and companies, both for rapid prototyping and in production.
# That doesn't mean it's perfect though:
#
# * there are parts that could be implemented more efficiently (in C, for example), or make better use of parallelism (multiple machines cores)
# * new algorithms are published all the time; help gensim keep up by `discussing them <http://groups.google.com/group/gensim>`_ and `contributing code <https://github.com/piskvorky/gensim/wiki/Developer-page>`_
# * your **feedback is most welcome** and appreciated (and it's not just the code!):
#   `bug reports <https://github.com/piskvorky/gensim/issues>`_ or
#   `user stories and general questions <http://groups.google.com/group/gensim/topics>`_.
#
# Gensim has no ambition to become an all-encompassing framework, across all NLP (or even Machine Learning) subfields.
# Its mission is to help NLP practitioners try out popular topic modelling algorithms
# on large datasets easily, and to facilitate prototyping of new algorithms for researchers.

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# img = mpimg.imread('run_similarity_queries.png')
# imgplot = plt.imshow(img)
# _ = plt.axis('off')
