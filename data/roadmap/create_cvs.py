import os
import csv

skip = []
folders = [name for name in os.listdir(".") if os.path.isdir(name)]

ofile  = open('data_file.csv', "w")
writer = csv.writer(ofile, delimiter=',')

for folder in folders:
	if folder in skip: continue
	folders2 = os.listdir(folder + '/')
	for folder2 in folders2:
		writer.writerow([folder + '/' + folder2])
