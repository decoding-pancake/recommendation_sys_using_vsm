from collections import defaultdict
from gensim import corpora
import json
# read corpus file
#  tagged map
tagged_corpus = open('data/input_tagged_corpus.json', 'r')
tagged_corpus = json.load(tagged_corpus)
count = 1
for seminars in tagged_corpus: 
    seminars["id"] = count
    count += 1

with open('data/input_tagged_corpus_id.json', 'w') as outfile:
    json.dump(tagged_corpus, outfile)

#  untagged map
untagged_corpus = open('data/input_untagged_corpus.json', 'r')
untagged_corpus = json.load(untagged_corpus)

for seminars in untagged_corpus: 
    seminars["id"] = count
    count += 1

with open('data/input_untagged_corpus_id.json', 'w') as outfile:
    json.dump(untagged_corpus, outfile)