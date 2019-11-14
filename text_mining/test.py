import konlpy
from konlpy.tag import Okt
from collections import Counter

nlp = Okt()

doc = konlpy.corpus.kolaw.open('constitution.txt').read()
nouns = nlp.nouns(doc)

# print(nouns)

count = Counter(nouns)

tag_count = []
tags = []

for n, c in count.most_common(100):
    dics = {'tag': n, 'count': c}
    if len(dics['tag']) >= 2 and len(tags) <= 15:
        tag_count.append(dics)
        tags.append(dics['tag'])

for tag in tag_count:
    print("{:<14}".format(tag['tag']), end='\t')
    print("{}".format(tag['count']))


