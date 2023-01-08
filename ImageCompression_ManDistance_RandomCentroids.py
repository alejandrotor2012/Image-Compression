import numpy as np
import os
import time
from scipy import misc
from PIL import Image
from matplotlib.pyplot import imshow
from numpy.random import choice

'''
This version of the algorithm assumes an l1 norm (aka Manhattan distance) with randomly initialized centroids.
Refer to "KMeansClusteringResults_RandomCentroids_withOneNorm.xlsx" spreadsheet attached with submission for detail on results.
'''

path = os.sys.path[0] + "\\MuchSurpriseVeryWow.bmp" #Path to load football image from active directory

def read_img(path):
     """
     Read image and store it as an array, given the image path.
     Returns the 3 dimensional image array.
     """
     img = Image.open(path)
     img_arr = np.array(img, dtype='int32')
     img.close()
     return img_arr


def display_image(arr):# Function to display the image of a three dimensional array
    arr = arr.astype('uint8')
    img = Image.fromarray(arr, "RGB")
    imshow(np.asarray(img))

def init_centers(X, k): #Randomly selects k points as centroids from X
    samples = choice(len(X), size = k, replace = False)
    return X[samples,:]


def compute_dis(X, centers):
    m = len(X)
    k = len(centers)

    S = np.empty((m,k))

    for i in range(m):
        d_i = np.linalg.norm(X[i, :] - centers, ord = 1, axis=1)
        S[i,:] = d_i **2

    return S

def assign_cluster_labels(S):
    return np.argmin(S, axis= 1)

def update_centers(X, y): #Assignment of new centroids per new clusters after above labeling function
    m, d = X.shape
    k = max(y) + 1

    centers = np.empty((k,d))
    for j in range(k):
        centers[j,:d] = np.mean(X[y == j , :], axis =0)
    return centers

def cluster_SS(S):
    return np.sum(np.amin(S, axis= 1))

def converge_test(old_centers, centers):
    return set([tuple(x) for x in old_centers]) == set([tuple(x) for x in centers])

def kmeans(X, k, starting_centers = None, max_steps = np.inf):
    if starting_centers is None:
        centers = init_centers(X, k)
    else:
        centers = starting_centers
    converged = False
    labels = np.zeros(len(X))
    i = 1

    while (not converged) and (i <= max_steps):
        old_centers = centers

        S = compute_dis(X,centers)
        labels = assign_cluster_labels(S)
        centers = update_centers(X, labels)
        converged = converge_test(old_centers, centers)

        print("--   iteration %d \n" % i)
        i += 1
    return labels


image = read_img(path) #Using function defined above to load image file and save as a three dimensional array
# display_image(image) #Using function defined above to display image

#Reshaping the 3 dimensional matrix to a two dimensional matrix consisting of (pixels, RGB values)
row, col, leng = image.shape
img_reshaped = image.reshape((row*col, leng), order = "C")

tic = time.time()
labels = kmeans(img_reshaped,8) #   <--------------- DETERMINE NUM. OF CLUSTERS HERE BY ASSINGING A VALUE FOR K
toc = time.time()
print('Elapsed time is %f seconds \n' % float(toc - tic))

ind = np.column_stack((img_reshaped, labels))
centers = {}
for i in set(labels):
    c = ind[ind[:,3] == i].mean(axis=0)
    centers[i] = c[:3]

img_clustered = np.array([centers[i] for i in labels])

row, col, leng = image.shape
img_disp = np.reshape(img_clustered, (row,col,leng) , order= "C")
display_image(img_disp)