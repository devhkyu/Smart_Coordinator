from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from glob import glob
from scipy.spatial import distance
import mahotas as mh
import numpy as np

images = glob('FashionImageDataset/*.jpg')

features = []
labels = []
for im in images:
    labels.append(im[19:-len('00.jpg')])
    im = mh.imread(im)
    im = mh.colors.rgb2gray(im, dtype=np.uint8)
    features.append(mh.features.haralick(im).ravel())

features = np.array(features)
labels = np.array(labels)
clf = Pipeline([('preproc', StandardScaler()), ('classifier', LogisticRegression())])
print(labels)
scores = cross_val_score(clf, features, labels)
print('Accuracy: {:.2%}'.format(scores.mean()))

sc = StandardScaler()
features = sc.fit_transform(features)
dists = distance.squareform(distance.pdist(features))


def selectImage(n, m, dists, images):
    image_position = dists[n].argsort()[m]
    image = mh.imread(images[image_position])
    return image


def plotImages(n):
    plt.figure(figsize=(15, 5))

    plt.subplot(141)
    plt.imshow(selectImage(n, 0, dists, images))
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(142)
    plt.imshow(selectImage(n, 1, dists, images))
    plt.title('1st simular one')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(143)
    plt.imshow(selectImage(n, 2, dists, images))
    plt.title('2nd simular one')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(144)
    plt.imshow(selectImage(n, 3, dists, images))
    plt.title('3rd simular one')
    plt.xticks([])
    plt.yticks([])

    plt.show()


plotImages(3)