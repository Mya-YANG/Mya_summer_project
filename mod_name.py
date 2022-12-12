
import os

os.chdir('ground_truth')

files = os.listdir()

for i in files:
	
	os.rename(i,i.replace('.xml',''))
	
	