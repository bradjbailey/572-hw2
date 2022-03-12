import torchvision.datasets as dset
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

from keras.utils import np_utils
from keras.datasets import cifar10
import torch

import torchvision
from torchvision import datasets

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# normalize images
x_train = x_train/255.0
x_test = x_test/255.0

#flatten images
x_train_flat = x_train.reshape(-1,3072)
x_test_flat = x_test.reshape(-1,3072)

#pca
pca = PCA()
pca.fit_transform(x_train_flat)

#find "optimal" components
k = 0
total = sum(pca.explained_variance_)
current_sum = 0

while(current_sum / total < 0.98):
    current_sum += pca.explained_variance_[k]
    k += 1
print("Optimal value of k:", k)

#rerun PCA with new k value
pca = PCA(n_components=k, svd_solver='randomized')

#obtain transformed data based on pca result
x_train_pca = pca.fit_transform(x_train_flat)
x_test_pca = pca.transform(x_test_flat)
print("PCA Train Shape:", x_train_pca.shape, "PCA Test Shape:", x_test_pca.shape)

#not sure if I need to do this part?
print("y train shape and type:\n", y_train.shape, type(y_train), "y test shape and type:\n", y_test.shape, type(y_test))
y_train_cat = np_utils.to_categorical(y_train)
y_test_cat = np_utils.to_categorical(y_test)
print("y cat train shape and type:\n", y_train_cat.shape, type(y_train_cat), "y cat test shape and type:\n", y_test_cat.shape, type(y_test_cat))

#linear classifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


