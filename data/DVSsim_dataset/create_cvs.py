import os
import csv

# get the name of the folders
folders = [name for name in os.listdir(".") if os.path.isdir(name)]

ofile  = open('data_file.csv', "w")
writer = csv.writer(ofile, delimiter=',')

for folder in folders:
	writer.writerow([folder])
