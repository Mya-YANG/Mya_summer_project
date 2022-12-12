import os

os.chdir('ground_truth')

old = os.listdir()

for i in old:
	
	print (i)
	
	os.rename(i,i+'txt')
			
	tmp_file = open(i+'txt','r')
	
	tmp_file = tmp_file.readlines()
	
	tmp_tmp_file = open(i,'w')
	
	for j in tmp_file:
		
		j = '0' + j[2::]
		
		tmp_tmp_file.write(j)
		
	tmp_tmp_file.close()
	
	os.remove(i+'txt')


