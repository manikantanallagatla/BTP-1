def getfeatures(lbp):
	features = np.zeros(256)
	length = lbp.shape[0]
	breadth = lbp.shape[1]
	for x in range(length):
		for y in range(breadth):
			features[lbp[x][y]] = features[lbp[x][y]]+1.0
	for x in range(256):
		features[x] = features[x]/(length*breadth)	
	return features



# OpenCV bindings
import cv2
# To performing path manipulations 
import os
# import cv
import matplotlib.pyplot as plt
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram 
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
# Utility package -- use pip install cvutils to install
import cvutils
# To read class from file
import csv
import numpy as np

print 'importing train data...'
###import train data
# Store the path of training images in train_images
train_images = cvutils.imlist("images_train1/")
# Dictionary containing image paths as keys and corresponding label as value
train_dic = {}
with open('class_train1.txt', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        train_dic[row[0]] = int(row[1])

# print train_dic

print 'constructing LBP for train set...'
###constructing LBP histograms for training images

# List for storing the LBP Histograms, address of images and the corresponding label 
X_test = []
X_name = []
y_test = []

english_train_images = 0
hindi_train_images = 0
telugu_train_images = 0
english_hist = 0
hindi_hist = 0 
telugu_hist = 0 

# For each image in the training set calculate the LBP histogram
# and update X_test, X_name and y_test
for train_image in train_images:
    # Read the image
    im = cv2.imread(train_image)
    # Convert to grayscale as LBP works on grayscale image
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    radius = 3
    # Number of points to be considered as neighbourers 
    no_points = 8 * radius
    # Uniform LBP is used
    lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
    
    # Calculate the histogram
    # x = itemfreq(lbp.ravel())
    # # print x
    # # Normalize the histogram
    # hist = x[:, 1]/sum(x[:, 1])
    hist = getfeatures(lbp)
    ##displaying
    print train_image
    # print hist

    ##viewing histograms
    # if('english' in str(train_image)):
    #     if(english_train_images == 0):
    #         english_hist = hist
    #     else:
    #         english_hist += hist
    #     english_train_images+=1

    #     plt.plot(hist)
    #     plt.axis([0,23,0,0.03])
    #     # plt.show()
    #     plt.savefig('english_'+str(english_train_images)+'.png')
    #     plt.clf()
    # if('hindi' in str(train_image)):
    #     if(hindi_train_images == 0):
    #         hindi_hist = hist
    #     else:
    #         hindi_hist += hist
    #     hindi_train_images+=1
    #     plt.plot(hist)
    #     plt.axis([0,23,0,0.03])
    #     # plt.show()
    #     plt.savefig('hindi_'+str(hindi_train_images)+'.png')
    #     plt.clf()
    # if('telugu' in str(train_image)):
        # if(telugu_train_images == 0):
        #     telugu_hist = hist
        # else:
        #     telugu_hist += hist
        # telugu_train_images+=1
        # plt.plot(hist)
        # plt.axis([0,23,0,0.03])
        # # plt.show()
        # plt.savefig('telugu_'+str(telugu_train_images)+'.png')
        # plt.clf()
    # plt.plot(hist)
    # plt.axis([0,23,0,0.03])
    # plt.show()
    # plt.clf()
    # Append image path in X_name
    X_name.append(train_image)
    # Append histogram to X_name
    X_test.append(hist)
    # Append class label in y_test
    y_test.append(train_dic[os.path.split(train_image)[1]])

# print X_test

# print 'X_test'
# print X_test
# print 'y_test'
# print y_test

###plotting
# english_hist = english_hist / english_train_images
# hindi_hist = hindi_hist / hindi_train_images
# telugu_hist = telugu_hist / telugu_train_images
# plt.plot(english_hist)
# plt.axis([0,25,0,0.03])
# plt.show()
# plt.clf()    # plt.plot(hist)
# plt.plot(hindi_hist)
# plt.axis([0,25,0,0.03])
# plt.show()
# plt.clf()    # plt.plot(hist)
# plt.plot(telugu_hist)
# plt.axis([0,25,0,0.03])
# plt.show()
# plt.clf()

# print X_test
# print y_test

print 'Importing test data...'
###import test data
# Store the path of testing images in test_images
test_images = cvutils.imlist("images_test1/")
# Dictionary containing image paths as keys and corresponding label as value
test_dic = {}
with open('class_test1.txt', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        test_dic[row[0]] = int(row[1])

# print test_dic	
import matplotlib.pyplot as plt
english_total = 0
hindi_total = 0
telugu_total = 0

english_got = 0
hindi_got = 0
telugu_got = 0

###chi square distance

# print 'Calculating chisquare for test set...'
# ###calculating chisquare distance
# for test_image in test_images:
#     # Read the image
#     im = cv2.imread(test_image)
#     # Convert to grayscale as LBP works on grayscale image
#     im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     radius = 3
#     # Number of points to be considered as neighbourers 
#     no_points = 8 * radius
#     # Uniform LBP is used
#     lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
#     # Calculate the histogram
#     x = itemfreq(lbp.ravel())
#     # Normalize the histogram
#     hist = x[:, 1]/sum(x[:, 1])
#     # Display the query image
#     # cvutils.imshow("** Query Image -> {}**".format(test_image), im)
#     results = []
#     # For each image in the training dataset
#     # Calculate the chi-squared distance and the sort the values
#     for index, x in enumerate(X_test):
#         score = cv2.compareHist(np.array(x, dtype=np.float32), np.array(hist, dtype=np.float32),1)
#         # score = cv2.compareHist(np.array(x, dtype=np.float32), np.array(hist, dtype=np.float32), cv2.cv.CV_COMP_CHISQR)
#         results.append((X_name[index], round(score, 3)))
#     results = sorted(results, key=lambda score: score[1])
	
#     # print test_image
#     expected = ''
#     print str(test_image)
#     print 'Expected: ',
#     if('english' in str(test_image)):
#         expected = 'english'
#         english_total+=1
#         print 'english'
#     else:
#         if('hindi' in str(test_image)):
#             print 'hindi'
#             hindi_total+=1
#             expected = 'hindi'
#         else:
#             print 'telugu'
#             telugu_total+=1
#             expected = 'telugu'
#     print 'Got: ',
#     got = ''
#     if('english' in str(results[0][0])):
#         print 'english'
#         got = 'english'
#     else:
#         if('hindi' in str(results[0][0])):
#             print 'hindi'
#             got = 'hindi'
#         else:
#             print 'telugu'
#             got = 'telugu'

#     if(expected == got):
#         if(expected == 'english'):
#             english_got+=1
#         else:
#             if(expected == 'hindi'):
#                 hindi_got+=1
#             else:
#                 telugu_got+=1

# print 'These are the results...'
# ###results
# print 'english accuracy = ', str((english_got+0.0)/english_total*100)
# print 'hindi accuracy = ', str((hindi_got+0.0)/hindi_total*100)
# print 'telugu accuracy = ', str((telugu_got+0.0)/telugu_total*100)
# print 'total accuracy = ', str((telugu_got+hindi_got+english_got+0.0)/(telugu_total+hindi_total+english_total)*100)



###trying nearest centroid algotrithm

english_total = 0
hindi_total = 0
telugu_total = 0

english_got = 0
hindi_got = 0
telugu_got = 0

from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np

X = np.array(X_test)
y = np.array(y_test)

# print X
clf = NearestCentroid()
# print X
# print X.shape
# print y
# print y.shape


clf.fit(X, y)
X_test1 = []
X_name1 = []
# print 'executed'
for test_image in test_images:

    # Read the image
    im = cv2.imread(test_image)
    # Convert to grayscale as LBP works on grayscale image
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    radius = 3
    # Number of points to be considered as neighbourers 
    no_points = 8 * radius
    # Uniform LBP is used
    lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
    # Calculate the histogram
    # x = itemfreq(lbp.ravel())
    # # Normalize the histogram
    # hist = x[:, 1]/sum(x[:, 1])
    hist = getfeatures(lbp)
    # hist = np.array(hist)
    # hist = [hist]
    # print hist
    # got_output = clf.predict(hist)
    print test_image
    # print got_output

    X_name1.append(test_image)
    # Append histogram to X_name
    X_test1.append(hist)

    # print test_image
    # print hist
#     expected = ''
#     print str(test_image)
#     print 'Expected: ',
#     if('english' in str(test_image)):
#         expected = 'english'
#         english_total+=1
#         print 'english'
#     else:
#         if('hindi' in str(test_image)):
#             print 'hindi'
#             hindi_total+=1
#             expected = 'hindi'
#         else:
#             print 'telugu'
#             telugu_total+=1
#             expected = 'telugu'
#     print 'Got: ',
#     got = ''
#     if('english' in str(results[0][0])):
#         print 'english'
#         got = 'english'
#     else:
#         if('hindi' in str(results[0][0])):
#             print 'hindi'
#             got = 'hindi'
#         else:
#             print 'telugu'
#             got = 'telugu'

#     if(expected == got):
#         if(expected == 'english'):
#             english_got+=1
#         else:
#             if(expected == 'hindi'):
#                 hindi_got+=1
#             else:
#                 telugu_got+=1
X_test2 = np.array(X_test1)
# print X_test1
# print X_test2
print clf.predict(X_test2)
# print 'These are the results...'
# ###results
# print 'english accuracy = ', str((english_got+0.0)/english_total*100)
# print 'hindi accuracy = ', str((hindi_got+0.0)/hindi_total*100)
# print 'telugu accuracy = ', str((telugu_got+0.0)/telugu_total*100)
# print 'total accuracy = ', str((telugu_got+hindi_got+english_got+0.0)/(telugu_total+hindi_total+english_total)*100)

# print X
# print y
# print X_test1


np.savetxt("foox.csv", X, delimiter=",")
np.savetxt("foox1.csv", X_test2, delimiter=",")

print X.shape
print X_test2.shape