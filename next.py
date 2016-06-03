#encoding=utf8

import csv

with open('data') as csv_file:
    data_file = csv.reader(csv_file)
    # header = next(data_file)
    for i,ir in enumerate(data_file):
        print i,ir[0]
