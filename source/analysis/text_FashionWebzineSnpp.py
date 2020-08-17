from konlpy.tag import Okt
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import konlpy.tag as kt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

origin = pd.read_csv("FashionWebzineSnpp(text).csv")
text_list = origin['Text'].tolist()

sentence = []
for x in text_list:
    if x is not np.nan:
        temp = x.split("에 ")
        if int(np.shape(temp)[0]) is 1:
            temp = ["drop"]
        elif len(temp[0]) > 20:
            temp = ["drop"]
        sentence.append(temp[0])
    else:
        sentence.append("drop")
result = pd.DataFrame(data=sentence, columns=['class_name'])
# print(result)

sub = []
for y in text_list:
    if y is not np.nan:
        temp = y.split("매치하여")
        if int(np.shape(temp)[0]) is 1:
            temp = ["drop"]
        temp = temp[0].replace("를 ", "+").replace("을 ", "+")\
            .replace("에 ", "+").replace("와 ", "+").replace("과 ", "+")
        sp = temp.split("+")
        sub.append(sp)
result2 = pd.DataFrame(data=sub)
print(result2)
document = '\n'.join(str(e) for e in sub)
nouns = nlp.nouns(document)
print(nouns)
count = Counter(nouns)
print(count)

tag_count = []
tags = []
for n, c in count.most_common(100):
    dics = {'tag': n, 'count': c}
    if len(dics['tag']) >= 2 and len(tags) <= 100:
        tag_count.append(dics)
        tags.append(dics['tag'])

for tag in tag_count:
    print("{:<14}".format(tag['tag']), end='\t')
    print("{}".format(tag['count']))

wordcloud = WordCloud(max_font_size=50, width=600, height=300, background_color='white', font_path='BMYEONSUNG_ttf.ttf').generate(document)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("FashionWebzineSnpp", fontsize=15)
plt.axis("off")
# plt.show()


##########################################
df = pd.read_csv("../url_data/FashionGio47.csv")
fg = df['Text'].tolist()

count = list(set(count.elements()))

dic = []
for k in count:
    if len(k) is not 1:
        dic.append(k)
print(dic)

result3 = []
result3_index = []
for x in range(len(fg)):
    if fg[x] is not np.nan:
        for y in dic:
            if y in fg[x]:
                result3_index.append(x)
                result3.append(y)
asd = pd.DataFrame(data=result3, index=result3_index, columns=['keyword'])
print(asd)


"""
document = '\n'.join(str(e) for e in text_list)
nouns = nlp.nouns(document)
# print(nouns)
count = Counter(nouns)
# print(count)

tag_count = []
tags = []
print("Result\n")
for n, c in count.most_common(100):
    dics = {'tag': n, 'count': c}
    if len(dics['tag']) >= 2 and len(tags) <= 100:
        tag_count.append(dics)
        tags.append(dics['tag'])

for tag in tag_count:
    print("{:<14}".format(tag['tag']), end='\t')
    print("{}".format(tag['count']))
"""
"""
test = np.asarray(text_list)
tf_idf(test[test != 'nan'])
"""

"""
komoran = kt.Komoran()
twitter = kt.Okt()
kkma = kt.Kkma()
hannanum = kt.Hannanum()

for x in text_list:
    if x is not np.nan:
        temp = twitter.pos(x)
        a = np.asarray(temp)
        print(a.reshape(-1, 2))
"""