from pathlib import Path
import pandas as pd
import json
import warnings
import sys
import os

warnings.filterwarnings(action='ignore')

# Initialize DATA_DIR, ROOT_DIR
os.chdir('..')
DATA_DIR = Path('')
ROOT_DIR = Path('')
sys.path.append(ROOT_DIR/'Mask_RCNN')

# Initialize NUM_CATS, IMAGE_SIZE
NUM_CATS = 46
IMAGE_SIZE = 512

# Load Label Descriptions to label_descriptions
with open(DATA_DIR/"label_descriptions.json") as f:
    label_descriptions = json.load(f)

# From label_descriptions['categories'] to label_names
label_names = [x['name'] for x in label_descriptions['categories']]

####################################
path = ''
train_df = pd.read_csv(path + 'train.csv')
label_description = open(path + 'label_descriptions.json').read()
label_description = json.loads(label_description)
label_description_info = label_description['info']
label_description_categories = pd.DataFrame(label_description['categories'])
label_description_attributes = pd.DataFrame(label_description['attributes'])

train_df_ImageId_count = train_df['ImageId'].value_counts()
print(train_df_ImageId_count)

# sample_df = train_df[train_df['ImageId'] == '361cc7654672860b1b7c85fe8e92b38a.jpg'].reset_index()
# image_df = train_df.groupby('ImageId')['EncodedPixels', 'ClassId']
# image = sample_df['EncodedPixels']
# print(sample_df['ClassId'].all)


'''
plt.figure(figsize=(20, 7))
plt.title('image labels count', size=20)
plt.xlabel('', size=15);plt.ylabel('', size=15)
sns.countplot(train_df_ImageId_count)
plt.show()
####################################
train_classid = pd.DataFrame({'ClassId':train_df['ClassId'].apply(lambda x: x[:2].replace('_', ''))})
label_merge = label_description_categories[['id', 'name']].astype(str).astype(object)
train_df_name = train_classid.merge(label_merge, left_on='ClassId', right_on='id', how='left')
sum1 = train_df_name.shape[0]
ratio1 = np.round(train_df_name.groupby(['ClassId', 'name']).count().sort_values(by='id', ascending=False).rename(columns = {'id':'count'})/sum1 * 100, 2)
train_df_name_stat = train_df_name.groupby(['ClassId', 'name']).count().sort_values(by='id', ascending=False).rename(columns = {'id':'count'}).reset_index()
train_df_name_stat['ratio(%)'] = ratio1.values
print(train_df_name_stat)

text = ''
for idx, name in enumerate(train_df_name_stat['name']):
    text += (name + ' ') * train_df_name_stat.loc[idx, 'count']
text = text[:-1]

wordcloud = WordCloud(max_font_size=50, width=600, height=300, background_color='white').generate(text)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("label_description_attributes in the images", fontsize=15)
plt.axis("off")
plt.show()

####################################
train_classid = pd.DataFrame({'ClassId':[j for i in train_df['ClassId'][train_df['ClassId'].apply(lambda x: '_' in x)].apply(lambda x: x.split('_')[1:]) for j in i]})
label_merge = label_description_attributes[['id', 'name']].astype(str).astype(object)
train_df_name = train_classid.merge(label_merge, left_on='ClassId', right_on='id', how='left')
sum1 = train_df_name.shape[0]
ratio1 = np.round(train_df_name.groupby(['ClassId', 'name']).count().sort_values(by='id', ascending=False).rename(columns = {'id':'count'})/sum1 * 100, 3)
train_df_name_stat = train_df_name.groupby(['ClassId', 'name']).count().sort_values(by='id', ascending=False).rename(columns = {'id':'count'}).reset_index()
train_df_name_stat['ratio(%)'] = ratio1.values
print(train_df_name_stat)

text = ''
for idx, name in enumerate(train_df_name_stat['name']):
    text += (name + ' ') * train_df_name_stat.loc[idx, 'count']
text = text[:-1]

wordcloud = WordCloud(max_font_size=50, width=600, height=300, background_color='white').generate(text)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("label_description_attributes in the images", fontsize=15)
plt.axis("off")
plt.show()
'''