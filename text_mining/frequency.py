import pandas as pd
import numpy as np
from konlpy.tag import Okt
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(document):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(document)

    word2id = defaultdict(lambda: 0)
    for idx, feature in enumerate(vectorizer.get_feature_names()):
        word2id[feature] = idx

    for i, sent in enumerate(document):
        print('document[%d]' % i)
        print([(token, tfidf[i, word2id[token]]) for token in sent.split()])


nlp = Okt()

musinsa_file = pd.read_csv("musinsa.csv")
gio_file = pd.read_csv("FashionGio47.csv")
snapp_file = pd.read_csv("FashionWebzineSnpp(text).csv")

musinsa_text = musinsa_file['Text']
list_musinsa = musinsa_text.head(100).tolist()
doc_musinsa = '\n'.join(str(e) for e in list_musinsa)

gio_title = gio_file['Title']
list_gio_title = gio_title.head(100).tolist()
doc_gio_title = '\n'.join(str(e) for e in list_gio_title)

gio_text = gio_file['Text']
list_gio_text = gio_text.head(100).tolist()
doc_gio_text = '\n'.join(str(e) for e in list_gio_text)

snapp_title = snapp_file['Title']
list_snapp_title = snapp_title.head(100).tolist()
doc_snapp_title = '\n'.join(str(e) for e in list_snapp_title)

snapp_text = snapp_file['Text']
list_snapp_text = snapp_text.head(100).tolist()
doc_snapp_text = '\n'.join(str(e) for e in list_snapp_text)

test = np.asarray(list_musinsa)

tf_idf(test[test != 'nan'])

nouns = nlp.nouns(doc_musinsa)

count = Counter(nouns)

tag_count = []
tags = []
print("Result\n")
for n, c in count.most_common(100):
    dics = {'tag': n, 'count': c}
    if len(dics['tag']) >= 2 and len(tags) <= 15:
        tag_count.append(dics)
        tags.append(dics['tag'])

for tag in tag_count:
    print("{:<14}".format(tag['tag']), end='\t')
    print("{}".format(tag['count']))
