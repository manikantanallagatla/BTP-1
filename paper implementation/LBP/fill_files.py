f = open('class_train.txt','w')
##filling english
for i in range(1,41):
	f.write('english_'+str(i)+'.jpg 0\n')

##filling hindi
for i in range(1,28):
	f.write('hindi_'+str(i)+'.jpg 1\n')

##filling telugu
for i in range(1,41):
	f.write('telugu_'+str(i)+'.jpg 2\n')


f = open('class_test.txt','w')
##filling english
for i in range(41,51):
	f.write('english_'+str(i)+'.jpg 0\n')

##filling hindi
for i in range(28,37):
	f.write('hindi_'+str(i)+'.jpg 1\n')

##filling telugu
for i in range(41,51):
	f.write('telugu_'+str(i)+'.jpg 2\n')