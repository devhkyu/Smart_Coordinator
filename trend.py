# Import Modules
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import json

# Initialize
DATA_DIR = Path('')
csv_name = 'pred_fashionwebzine.csv'

# Load Label Descriptions to label_descriptions
with open(DATA_DIR / "label_descriptions.json") as f:
    label_descriptions = json.load(f)

# From label_descriptions['categories'] to label_names
label_names = [x['name'] for x in label_descriptions['categories']]

# Read csv to dataFrame
df = pd.read_csv(csv_name, names=['img', 'index', 'class'])
print(df.head())

# Counting with class_ids
value_counts = df['class'].value_counts()
df_class_count = value_counts.rename_axis('class').reset_index(name='count')
print(df_class_count)

# Counting with label_name
list_class = []
for x in range(df.__len__()):
    list_class.append(label_names[df['class'][x]])
list_class = pd.DataFrame(list_class)
value_counts = list_class[0].value_counts()
df_class_count = value_counts.rename_axis('class').reset_index(name='count')
print(df_class_count)

# Splitting values to x, y
x = df_class_count['class'].values
y = df_class_count['count'].values

# Bar chart
plt.bar(x, y, width=1)
plt.xlabel("class")
plt.ylabel("count")
plt.title("Bar chart: %s" % csv_name, fontsize=15)
plt.show()

# Reformatting to dictionary
dic = {}
for i in x:
    if i not in dic.keys():
        dic[i] = 0
    dic[i] += 1

# Generate a word cloud image
wordcloud = WordCloud().generate_from_frequencies(frequencies=dic)

# Display the generated image
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud)
plt.title("WordCloud: %s" % csv_name, fontsize=30)
plt.axis("off")
plt.show()