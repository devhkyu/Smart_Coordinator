# Import Modules
import numpy as np
import pandas as pd
import os
import cv2
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from PIL import Image
import math
import seaborn as sns


def classid2label(class_id):
    category, *attribute = class_id.split("_")
    return category, attribute


def print_dict(dictionary, name_dict):
    print("{}{}{}{}{}".format("rank".ljust(5), "id".center(8), "name".center(40), "amount".rjust(10), "ratio(%)".rjust(10)))
    all_num = sum(dictionary.values())
    for i, (key, val) in enumerate(sorted(dictionary.items(), key=lambda x: -x[1])):
        print("{:<5}{:^8}{:^40}{:>10}{:>10.3%}".format(i+1, key, name_dict[key], val, val/all_num))


def print_img_with_labels(img_name, labels, category_name_dict, attribute_name_dict, ax):
    img = np.asarray(Image.open("train/" + img_name))
    label_interval = (img.shape[0] * 0.9) / len(labels)
    ax.imshow(img)
    for num, attribute_id in enumerate(labels):
        x_pos = img.shape[1] * 1.1
        y_pos = (img.shape[0] * 0.9) / len(labels) * (num + 2) + (img.shape[0] * 0.1)
        if(num == 0):
            ax.text(x_pos, y_pos-label_interval*2, "category", fontsize=12)
            ax.text(x_pos, y_pos-label_interval, category_name_dict[attribute_id], fontsize=12)
            if(len(labels) > 1):
                ax.text(x_pos, y_pos, "attribute", fontsize=12)
        else:
            ax.text(x_pos, y_pos, attribute_name_dict[attribute_id], fontsize=12)


def print_img(img_name, ax):
    img_df = train_df[train_df.ImageId == img_name]
    labels = list(set(img_df["ClassId"].values))
    print_img_with_labels(img_name, labels, category_name_dict, attribute_name_dict, ax)


def json2df(data):
    df = pd.DataFrame()
    for index, el in enumerate(data):
        for key, val in el.items():
            df.loc[index, key] = val
    return df


pd.set_option("display.max_rows", 101)
plt.rcParams["font.size"] = 15
os.chdir('../..')
train_df = pd.read_csv("train.csv")
with open("label_descriptions.json") as f:
    label_description = json.load(f)

print("this dataset info")
print(json.dumps(label_description["info"], indent=2))
category_df = json2df(label_description["categories"])
category_df["id"] = category_df["id"].astype(int)
category_df["level"] = category_df["level"].astype(int)
attribute_df = json2df(label_description["attributes"])
attribute_df["id"] = attribute_df["id"].astype(int)
attribute_df["level"] = attribute_df["level"].astype(int)
print("Category Labels")
print(category_df)
print("Attribute Labels")
print(attribute_df)
print("We have {} categories, and {} attributes.".format(len(label_description['categories']), len(label_description['attributes'])))
print("Each label　have ID, name, supercategory, and level.")

image_label_num_df = train_df.groupby("ImageId")["ClassId"].count()
fig, ax = plt.subplots(figsize=(25, 7))
x = image_label_num_df.value_counts().index.values
y = image_label_num_df.value_counts().values
z = zip(x, y)
z = sorted(z)
x, y = zip(*z)
index = 0
x_list = []
y_list = []
for i in range(1, max(x)+1):
    if(i not in x):
        x_list.append(i)
        y_list.append(0)
    else:
        x_list.append(i)
        y_list.append(y[index])
        index += 1
for i, j in zip(x_list, y_list):
    ax.text(i-1, j, j, ha="center", va="bottom", fontsize=13)
sns.barplot(x=x_list, y=y_list, ax=ax)
ax.set_xticks(list(range(0, len(x_list), 5)))
ax.set_xticklabels(list(range(1, len(x_list), 5)))
ax.set_title("the number of labels per image")
ax.set_xlabel("the number of labels")
ax.set_ylabel("amout")
plt.show()

counter_category = Counter()
counter_attribute = Counter()
for class_id in train_df["ClassId"]:
    category, attribute = classid2label(class_id)
    counter_category.update([category])
    counter_attribute.update(attribute)

category_name_dict = {}
for i in label_description["categories"]:
    category_name_dict[str(i["id"])] = i["name"]
attribute_name_dict = {}
for i in label_description["attributes"]:
    attribute_name_dict[str(i["id"])] = i["name"]
print("Category label frequency")
print_dict(counter_category, category_name_dict)

attribute_num_dict = {}
none_key = str(len(counter_attribute))
k = list(map(str, range(len(counter_attribute) + 1)))
v = [0] * (len(counter_attribute) + 1)
zipped = zip(k, v)
init_dict = dict(zipped)
for class_id in train_df["ClassId"].values:
    category, attributes = classid2label(class_id)
    if category not in attribute_num_dict.keys():
        attribute_num_dict[category] = init_dict.copy()
    if attributes == []:
        attribute_num_dict[category][none_key] += 1
        continue
    for attribute in attributes:
        attribute_num_dict[category][attribute] += 1

fig, ax = plt.subplots(math.ceil(len(counter_category)/2), 2,\
                       figsize=(8*2, 6*math.ceil(len(counter_category)/2)), sharey=True)
for index, key in enumerate(sorted(map(int, attribute_num_dict.keys()))):
    x = list(map(int, attribute_num_dict[str(key)].keys()))
    total = sum(attribute_num_dict[str(key)].values())
    y = list(map(lambda x: x / total, attribute_num_dict[str(key)].values()))
    sns.barplot(x, y, ax=ax[index//2, index%2])
    ax[index//2, index%2].set_title("category:{}({})".format(key, category_name_dict[str(key)]))
    ax[index//2, index%2].set_xticks(list(range(0, int(none_key), 5)))
    ax[index//2, index%2].set_xticklabels(list(range(0, int(none_key), 5)))
print("the ratio of attribute per category(x=92 means no attribute)")
plt.show()