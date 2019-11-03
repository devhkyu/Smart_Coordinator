# iMaterialist Fashion 2019 at FGVC6
# EDA: supercategories, attributes, correctness
# https://www.kaggle.com/latticetower/eda-supercategories-attributes-correctness

# Import Modules
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Set directory path
os.chdir('..')
BASE_DIR = Path('')
IMAGES_TRAIN_DIR = f"{BASE_DIR}/train"
IMAGES_TEST_DIR = f"{BASE_DIR}/test"
TRAIN_CSV = f"{BASE_DIR}/train.csv"
LABEL_DESCRIPTIONS = f"{BASE_DIR}/label_descriptions.json"

# Read csv, json file to DataFrame
train_df = pd.read_csv(TRAIN_CSV)
with open(LABEL_DESCRIPTIONS) as f:
    image_info = json.load(f)
categories = pd.DataFrame(image_info['categories'])
attributes = pd.DataFrame(image_info['attributes'])
print("There are descriptions for", categories.shape[0],"categories and", attributes.shape[0], "attributes")

train_df['hasAttributes'] = train_df.ClassId.apply(lambda x: x.find("_") > 0)
train_df['CategoryId'] = train_df.ClassId.apply(lambda x: x.split("_")[0]).astype(int)
train_df = train_df.merge(categories, left_on="CategoryId", right_on="id")
print(categories.head())
print("Fraction of mask annotations with any attributes within train data:", train_df.hasAttributes.mean())

# Supercategories with no attributes
subset = train_df[~train_df.hasAttributes]
supercategory_names = np.unique(subset.supercategory)
plt.figure(figsize=(10, 10))
g = sns.countplot(x='supercategory', data=subset, order=supercategory_names)
ax = g.axes
tl = [x.get_text() for x in ax.get_xticklabels()]
ax.set_xticklabels(tl, rotation=90)
for p, label in zip(ax.patches, supercategory_names):
    c = subset[(subset['supercategory'] == label)].shape[0]
    ax.annotate(str(c), (p.get_x(), p.get_height() + 1000))
plt.title("Supercategories with no attributes")
plt.show()

# Supercategories with any attributes
subset = train_df[train_df.hasAttributes]
supercategory_names = np.unique(subset.supercategory)
g = sns.countplot(x = 'supercategory', data=subset, order=supercategory_names)
ax = g.axes
tl = [x.get_text() for x in ax.get_xticklabels()]
ax.set_xticklabels(tl, rotation=90)
for p, label in zip(ax.patches, supercategory_names):
    c = subset[(subset['supercategory'] == label)].shape[0]
    ax.annotate(str(c), (p.get_x()+0.3, p.get_height() + 50))
plt.title("Supercategories with any attributes")
plt.show()

supercategory_names = train_df[['supercategory', 'name']].groupby('supercategory').agg(
    lambda x: x.unique().shape[0]).reset_index().sort_values("name", ascending=False).set_index('name')
print(supercategory_names)


def buildPlot(**kwargs):
    data = kwargs['data']
    g = sns.countplot(y="name", data=data)
    g.set_yticklabels(data['name'].unique())  # , rotation=90)


idx = train_df.supercategory.isin(['decorations', 'garment parts', 'upperbody'])
g = sns.FacetGrid(data=train_df[idx], col="supercategory", sharey=False)
g = g.map_dataframe(buildPlot)
plt.show()

idx = train_df.supercategory.isin(supercategory_names.supercategory.loc[4].values)
g = sns.FacetGrid(data=train_df[idx], col="supercategory", sharey=False)
g = g.map_dataframe(buildPlot)
plt.show()

total = train_df.ImageId.unique().shape[0]
print(f"There are {total} images in train dataset.")
images_with_shoes = train_df[train_df.name=="shoe"].ImageId.unique().shape[0]
images_with_legs = train_df[train_df.supercategory=="legs and feet"].ImageId.unique().shape[0]
print(f"However, only {images_with_legs} images have associated legs and feet annotation, and only {images_with_shoes} have any shoes on it.")

idx = train_df.supercategory.isin(supercategory_names.supercategory.loc[3].values)
g = sns.FacetGrid(data=train_df[idx], row="supercategory", sharey=False)
g = g.map_dataframe(buildPlot)
plt.show()

idx = train_df.supercategory.isin(supercategory_names.supercategory.loc[2].values)
g = sns.FacetGrid(data=train_df[idx], col="supercategory", sharey=False)
g = g.map_dataframe(buildPlot)
plt.show()

idx = train_df.supercategory.isin(supercategory_names.supercategory.loc[1].values)
g = sns.FacetGrid(data=train_df[idx], col="supercategory", sharey=False)
g = g.map_dataframe(buildPlot)
plt.show()

# extract all available attributes and create separate table
cat_attributes = []
for i in train_df[train_df.hasAttributes].index:
    item = train_df.loc[i]
    xs = item.ClassId.split("_")
    for a in xs[1:]:
        cat_attributes.append({'ImageId': item.ImageId, 'category': int(xs[0]), 'attribute': int(a)})
cat_attributes = pd.DataFrame(cat_attributes)

cat_attributes = cat_attributes.merge(
    categories, left_on="category", right_on="id"
).merge(attributes, left_on="attribute", right_on="id", suffixes=("", "_attribute"))

# helper objects and methods
scat_x, count_x = np.unique(cat_attributes['supercategory'], return_counts=True)
categories_by_x = {
    x: dict(cat_attributes[cat_attributes['supercategory'] == x][['name', 'category']].drop_duplicates().values)
    for x in scat_x}
scat_y, count_y = np.unique(cat_attributes['supercategory_attribute'], return_counts=True)
categories_by_y = {
    y: dict(cat_attributes[cat_attributes['supercategory_attribute'] == y][['name_attribute', 'attribute']].drop_duplicates().values)
    for y in scat_y}
vals = cat_attributes.groupby(['category', 'attribute']).count().reset_index(drop=True).values[:,0]
scale_min, scale_max = vals.min(), vals.max()


def get_scatter_data(x, y, cat, attr):
    ids_x = {cat[k]: i for i, k in enumerate(cat)}
    ids_y = {attr[k]: i for i, k in enumerate(attr)}
    data = np.zeros((len(cat), len(attr)), dtype=np.uint)
    for k, v in zip(x, y):
        data[ids_x[k], ids_y[v]]+=1
    ii, jj = np.where(data > 0)
    sizes = [data[i, j] for i, j in zip(ii, jj)]
    return ii, jj, sizes


def drawPunchcard(**kwargs):
    data = kwargs['data']
    x = data["category"]
    y = data["attribute"]
    supercategory_x = data["supercategory"].values[0]
    cat = categories_by_x[supercategory_x]
    supercategory_y = data["supercategory_attribute"].values[0]
    attr = categories_by_y[supercategory_y]
    ii, jj, sizes = get_scatter_data(
        x, y,
        cat,
        attr)
    g = sns.scatterplot(ii, jj, size=sizes, sizes=(20, 200), hue=np.log(sizes)+1)
    g.set_xticks(np.arange(len(cat)))
    g.set_xticklabels(list(cat), rotation=90)
    g.set_yticks(np.arange(len(attr)))
    g.set_yticklabels(list(attr))


sns.color_palette("bright")
sns.set(font_scale=1.0)
sns.set_style("white")
width_ratios=[len(categories_by_x[x]) for x in categories_by_x]
height_ratios=[len(categories_by_y[x]) for x in categories_by_y]
g = sns.FacetGrid(data=cat_attributes, col="supercategory",  row="supercategory_attribute",
                  #margin_titles=True,
                  gridspec_kws={'height_ratios': height_ratios, 'width_ratios': width_ratios},
                  sharex="col", sharey="row",
                  col_order=list(categories_by_x),
                  row_order=list(categories_by_y))#.set_titles('{col_name}', '{row_name}')
g = g.map_dataframe(drawPunchcard).set_titles('{col_name}', '{row_name}')
g.fig.set_size_inches(10, 20)
for ax, cat_name in zip(g.axes, list(categories_by_y)):
    ax[-1].set_ylabel(cat_name, labelpad=10, rotation=-90)
    ax[-1].yaxis.set_label_position("right")
plt.show()

#########################################################
# Closer look at train data
images = train_df[['ImageId', "Width", "Height"]].drop_duplicates()
print("Number of unique triplets (ImageId, Width, Height):", images.shape[0])
print("Unique image names: ", images['ImageId'].unique().shape[0])


def read_image_dimensions(path):
    "returns real width and height"
    with Image.open(path) as image:
        dimensions = image.size
    return dimensions


images_with_incorrect_size = {}
for ImageId, width, height in images.values:
    image_path = os.path.join(IMAGES_TRAIN_DIR, ImageId)
    (real_width, real_height) = read_image_dimensions(image_path)
    if real_width != width or real_height!=height:
        images_with_incorrect_size[ImageId] = (real_width, real_height)
print("Number of images with incorrect dimensions:", len(images_with_incorrect_size))

for ImageId in images_with_incorrect_size:
    (width, height) = images_with_incorrect_size[ImageId]
    idx = train_df['ImageId'] == ImageId
    print(ImageId, train_df.loc[idx, "Width"].values[0], train_df.loc[idx, "Height"].values[0], "Real dimensions:", width, height)
    train_df.loc[idx, "Width"] = width
    train_df.loc[idx, "Height"] = height


df = train_df[["ImageId", "EncodedPixels", "ClassId"]].drop_duplicates()
grouped_df = df.groupby(["EncodedPixels", "ImageId"]).count().reset_index()
grouped_df = grouped_df[grouped_df.ClassId > 1]
print("Number of images with duplicated EncodedPixels:", grouped_df.shape[0])

duplicated_data = df[df.ImageId.isin(grouped_df.ImageId) & df.EncodedPixels.isin(grouped_df.EncodedPixels)].sort_values(["ImageId", "EncodedPixels"])
duplicated_data.to_csv("images_with_duplicated_masks.csv", index=None) # you can look at these images, if you want

duplicates = dict()
xlabels, ylabels = set(), set()

for (ImageId, EncodedPixels), x in duplicated_data.groupby(["ImageId", "EncodedPixels"]):
    pair = tuple(sorted(x.ClassId.values))
    s, e = pair
    xlabels.add(s)
    ylabels.add(e)
    if not pair in duplicates:
        duplicates[pair] = 0
    duplicates[pair] +=1

xlabels = {x: i for i, x in enumerate(sorted(xlabels))}
ylabels = {x: i for i, x in enumerate(sorted(ylabels))}
matrix = np.zeros((len(ylabels), len(xlabels)), dtype=np.int)
for (s, e) in duplicates:
    matrix[ylabels[e], xlabels[s]] = duplicates[(s, e)]


plt.figure(figsize=(10,10))
annot = np.array([[(str(x) if x >0 else "") for x in line]for line in matrix])
sns.heatmap(matrix, annot=annot, fmt="s",xticklabels=sorted(list(xlabels)), yticklabels=sorted(list(ylabels)), square=True, cbar=False)
plt.show()
print(categories[categories.id.isin((32, 35))])

################################################
# Masking


def sum_mask_pixels(encoded_pixels):
    pixels = [np.int(x) for x in encoded_pixels.split(" ")]
    return np.sum(pixels[1::2])


def compute_mask_percentage(row):
    s = sum_mask_pixels(row['EncodedPixels'])
    return 1.0* s/row["Width"]/row["Height"]


train_df['mask_fraction'] = train_df.EncodedPixels.apply(sum_mask_pixels).astype(np.float)
train_df['mask_fraction'] = train_df['mask_fraction']/train_df["Width"]/train_df["Height"]

plt.figure(figsize=(10, 8))
g = sns.stripplot(y="mask_fraction", data=train_df, x="supercategory")
labels = [x.get_text() for x in g.get_xticklabels()]
g = g.set_xticklabels(labels, rotation=90)
plt.show()


def draw_images(data=None,**kwargs):
    plt.axis("off")
    path = os.path.join(IMAGES_TRAIN_DIR, data['ImageId'].values[0])
    with Image.open(path) as image:
        data = np.asarray(image)
    plt.imshow(data)


subset = train_df[train_df.mask_fraction > 0.7]
grid = sns.FacetGrid(subset, col="name", col_wrap=4)
grid.map_dataframe(draw_images)
plt.show()