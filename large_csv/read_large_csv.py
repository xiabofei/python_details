#encoding=utf8

"""
http://pythondata.com/working-large-csv-files-python/
how use pandas to load large csv file
    1. not that large needs distributed network
    2. too large for once loading in memory but can fit hard-disk
"""
import pandas as pd

path = './test_data.csv'

# print pd.read_csv(path, nrows=1)

i = 0
for df in pd.read_csv(path, chunksize=100000, iterator=True):
    # df.to_pickle('./df_output_'+str(i)+'.pkl')
    df.to_csv('./df_output_'+str(i)+'.csv')
    i += 1
