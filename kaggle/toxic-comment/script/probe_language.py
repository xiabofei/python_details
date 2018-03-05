# encoding=utf8

import numpy as np
import pandas as pd
from collections import defaultdict
import regex

script_list = [
    r'\p{Arabic}', r'\p{Armenian}', r'\p{Bengali}', r'\p{Bopomofo}', r'\p{Braille}',
    r'\p{Buhid}', r'\p{Canadian_Aboriginal}', r'\p{Cherokee}', r'\p{Cyrillic}',
    r'\p{Devanagari}', r'\p{Ethiopic}', r'\p{Georgian}', r'\p{Greek}', r'\p{Gujarati}',
    r'\p{Gurmukhi}', r'\p{Han}', r'\p{Hangul}', r'\p{Hanunoo}', r'\p{Hebrew}', r'\p{Hiragana}',
    r'\p{Inherited}', r'\p{Kannada}', r'\p{Katakana}', r'\p{Khmer}', r'\p{Lao}', r'\p{Latin}',
    r'\p{Limbu}', r'\p{Malayalam}', r'\p{Mongolian}', r'\p{Myanmar}', r'\p{Ogham}', r'\p{Oriya}',
    r'\p{Runic}', r'\p{Sinhala}', r'\p{Syriac}', r'\p{Tagalog}', r'\p{Tagbanwa}',
    r'\p{TaiLe}', r'\p{Tamil}', r'\p{Telugu}', r'\p{Thaana}', r'\p{Thai}', r'\p{Tibetan}',
    r'\p{Yi}', r'\p{Common}'
]

df_train_toxic = pd.read_csv('../data/input/train_toxic.csv')

script_occ = pd.DataFrame(
    [regex.sub(r'\\p\{(.+)\}', r'\g<1>', reg) for reg in script_list], columns=["script"]
)

script_occ["train"] = [
    df_train_toxic["comment_text"].apply(lambda x: len(regex.findall(reg, x))).sum()
    for reg in script_list
]

script_occ["train_docs"] = [
    (df_train_toxic["comment_text"].apply(lambda x: len(regex.findall(reg, x))) > 0).sum()
    for reg in script_list
]

df_train_toxic['latin'] = df_train_toxic['comment_text'].apply(lambda x: len(regex.findall(r'\p{Arabic}', x))>0)

df_train_latin_toxic = pd.DataFrame()
df_train_latin_toxic['id'] = df_train_toxic[df_train_toxic.latin == True]['id']
df_train_latin_toxic['comment_text'] = df_train_toxic[df_train_toxic.latin == True]['comment_text']
df_train_latin_toxic.to_csv('../data/input/latin_toxic.csv', index=False)
