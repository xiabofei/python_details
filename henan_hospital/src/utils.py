# encoding=utf8

NAN = 'nan'

def fill_na(df, column):
    return df[column].fillna(NAN)

