import cv2
import math
import csv
from scipy.misc import toimage
import matplotlib.pyplot as plt

def process(file_input,file_output):
	img = cv2.imread(file_input, cv2.IMREAD_GRAYSCALE)
	# img = cv2.medianBlur(img, 3)
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.medianBlur(img, 3)

	###resize img for hindi
	# img = img[0:4500,200:3100]

	# toimage(img).show()
	import numpy as np
	img = np.asarray(img)
	print img.shape
	kernel = np.ones((3,3),np.uint8)
	#if want do
	# img = cv2.erode(img,kernel,iterations = 1)
	# img = cv2.dilate(img,kernel,iterations = 1)

	#process noise
	# import Tkinter

	horizontalprojection = np.mean(img, axis=1)



	plt.plot(horizontalprojection)

	sum1 = 0
	countPizelsPeak = 0

	for i in horizontalprojection:
		if(i != 255):
			sum1+=i
			countPizelsPeak+=1

	threshold = (sum1+0.0)/countPizelsPeak


	#finding peaks
	numberPeaks = 0
	temp_width = 0
	sumWidth = 0
	foundPeakStart = 0
	horizontalprojection1 = np.copy(horizontalprojection)
	for i in range(len(horizontalprojection1)):
		if(horizontalprojection1[i]<=threshold):
			if(foundPeakStart == 0):
				foundPeakStart = 1
				temp_width+=1
				numberPeaks+=1
				horizontalprojection1[i] = 0
			else:
				temp_width+=1
				horizontalprojection1[i] = 0
		else:
			if(foundPeakStart == 0):
				#no worries
				horizontalprojection1[i] = 255
			else:
				foundPeakStart = 0
				sumWidth+=temp_width
				temp_width = 0
				horizontalprojection1[i] = 255

	averageWidth = (sumWidth+0.0)/numberPeaks
	print averageWidth, numberPeaks
	# plt.plot(horizontalprojection1)
	# plt.show()
	temp_width = 0
	foundPeakStart = 0
	indexPeakStart = 0
	i = 0
	while(i <(len(horizontalprojection1))):
		if(horizontalprojection1[i]==0):
			if(foundPeakStart == 0):
				foundPeakStart = 1
				indexPeakStart = i
				temp_width+=1
			else:
				temp_width+=1
		else:
			if(foundPeakStart == 0):
				x = 0
			else:
				foundPeakStart = 0
				if(temp_width<0.5*averageWidth or temp_width>1.5*averageWidth):
					j = indexPeakStart
					while(j<=i):
						horizontalprojection1[j] = 255
						j+=1
				temp_width = 0
		i+=1

	# plt.plot(horizontalprojection1)
	# plt.show()
	#finding width
	foundPeakStart = 0
	indexPeakStart = 0
	removePeak = 1
	i = 0
	while(i <(len(horizontalprojection))):
		if(horizontalprojection[i]!=255):
			if(foundPeakStart == 0):
				foundPeakStart = 1
				indexPeakStart = i
			if(horizontalprojection1[i] == 0):
				removePeak = 0
		else:
			if(foundPeakStart == 0):
				x = 0
			else:
				foundPeakStart = 0
				if(removePeak == 1):
					j = indexPeakStart
					while(j<=i):
						horizontalprojection[j] = 255
						j+=1
				removePeak = 1
		i+=1

	plt.plot(horizontalprojection)
	# plt.show()

	rows = (img.shape)[0]
	columns = (img.shape)[1]

	# toimage(img).show()

	#can change line space in future
	required_space_between_lines = int(15*averageWidth)

	#constructing uniform space lines
	new_img = np.zeros((20,columns))
	for i in range(20):
		for j in range(columns):
			new_img[i][j] = 255
	foundPeakStart = 0
	indexPeakStart = 0
	i = 0
	z = 0

	first_time_block = 1
	last_block_half_size = 0

	while(i <(len(horizontalprojection))):
		if(horizontalprojection[i]!=255):
			if(foundPeakStart == 0):
				foundPeakStart = 1
				indexPeakStart = i
		else:
			if(foundPeakStart == 1):
				foundPeakStart = 0
				z = np.asarray(img[indexPeakStart:i+1])

				#process z for equal spaced words
				indexlefttravel = 0
				z_temp = np.copy(z)
				for i_z in range((z.shape)[0]):
					for j_z in range(5):
						z_temp[i_z][j_z] = 255
				while(indexlefttravel<columns):
					allwhite = 1
					for i_z in range((z.shape)[0]):
						if(z[i_z][indexlefttravel] <128):
							allwhite = 0
							break
					if(allwhite == 0):
						break

					indexlefttravel+=1
				sizeslice = columns - indexlefttravel
				z_temp[:,5:(5+sizeslice)] = z[:,indexlefttravel:columns]
				#right padding individual lines
				paddi = columns-1
				for padd_travel in range(indexlefttravel-5):
					for padd_row in range((z_temp.shape)[0]):
						z_temp[padd_row][paddi] = 255
					paddi-=1
				if(first_time_block == 1):
					dummy = np.concatenate((new_img,z_temp),axis = 0)
					new_img = dummy
					first_time_block = 0
					last_block_half_size = ((z_temp.shape)[0])/2
				else:
					present_block_half_size = ((z_temp.shape)[0])/2
					padding_rows_need = (required_space_between_lines - (present_block_half_size + last_block_half_size))
					padd_block = np.zeros((padding_rows_need,(z_temp.shape)[1]))
					for padd_blocki in range(padding_rows_need):
						for padd_blockj in range((z_temp.shape)[1]):
							padd_block[padd_blocki][padd_blockj] = 255
					dummy = np.concatenate((new_img,padd_block),axis = 0)
					new_img = dummy
					dummy = np.concatenate((new_img,z_temp),axis = 0)
					new_img = dummy
					last_block_half_size = present_block_half_size
					

		i+=1
	# toimage(new_img).show()


	#bottom padding
	padd_block = np.zeros((20,columns))
	for padd_blocki in range(20):
		for padd_blockj in range(columns):
			padd_block[padd_blocki][padd_blockj] = 255
	dummy = np.concatenate((new_img,padd_block),axis = 0)
	new_img = dummy

	# toimage(new_img).show()


	with open('img'+str(i)+'.csv', 'w') as csvfile:
	    writer = csv.writer(csvfile)
	    [ writer.writerow(r) for r in img[:,:]]

	# print new_img
	# toimage(new_img).show()
	#doing right padding
	rows = (new_img.shape)[0]
	columns = (new_img.shape)[1]

	index = columns-1
	while(index>=0):
		# index1 = index-5
		allwhite = 1
		for i in range(rows):
			if(new_img[i][index]!=255):
				allwhite = 0
				break
		if(allwhite == 0):
			break
		index-=1

	new_img1 = new_img[:,:index+5]
	# print new_img1
	new_img = np.copy(new_img1)

	# print new_img

	# toimage(new_img).show()

	import scipy.misc
	scipy.misc.imsave(file_output, new_img)

# count = 0
# for i in range(110,250):
# 	file_input = '/media/manikantanallagatla/D666D33D66D31D55/ManikantaBackup3/study/semVII/BTP/data/hindi/hindi_content/file-page'+str(i)+'.jpg'
# 	file_output = '/media/manikantanallagatla/D666D33D66D31D55/ManikantaBackup3/study/semVII/BTP/data/hindi/hindi_content/'+str(i)+'_clean.jpg'
# 	# process(file_input,file_output)
# 	try:
# 		process(file_input,file_output)
# 	except Exception, e:
# 		count+=1
# 		print 'small error '+str(count)
# 		print e

file_input = 'result3.jpg'
file_output = 'result4.jpg'
process(file_input,file_output)