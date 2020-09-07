# -*- coding: utf-8 -*-
"""preprocessing_data_&_baseline_models.ipynb

Original file is located at
    https://colab.research.google.com/drive/1Bzm1S12fS3aPdiVNXzsLLNF78IGFn21b

## Preprocessing and character level models
"""

import pandas as pd
import numpy as np
import os
import dill
import pandas as pd
import glob, csv
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt') # Download this as this allows you to tokenize words in a string.
lemmatizer = WordNetLemmatizer()


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import re
import string
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


!pip install num2words

from nltk.stem import PorterStemmer
from collections import Counter
import num2words

import os
import copy
import pickle
import math


from google.colab import drive
drive.mount('/content/drive')

!unzip '/content/drive/My Drive/519/Project/C50.zip'

"""## TRAIN PRE-PROCESS Sentence level"""

train_file_df = pd.DataFrame()
temp_df = pd.DataFrame()
for i, filepath in enumerate(glob.iglob('/content/C50train/*/*.txt')):
    print('Analysing', i , ' -> ', ' of Author ', str(filepath.split("/")[3]))
    temp_df = pd.read_csv(filepath, delimiter="\t", header = None, error_bad_lines=False, quoting = csv.QUOTE_NONE, encoding='utf-8')
    temp_df['Author'] = filepath.split("/")[3]
    temp_frame = [train_file_df, temp_df]
    train_file_df = pd.concat(temp_frame)

train_file_df = train_file_df.rename(columns={ 0 : 'text'})

train_file_df

"""## Train Preprocess Article level"""

# article level

train_file_article_df = pd.DataFrame(columns=['text','Author'])
temp_df = pd.DataFrame(columns=['text','Author'])

for i, filepath in enumerate(glob.iglob('/content/C50train/*/*.txt')):
  print('Analysing', i , ' -> ', ' of Author ', str(filepath.split("/")[3]))
  with open(filepath, 'r') as myfile:
    data = myfile.read()
  # tokens = word_tokenize(data)
  train_file_article_df = train_file_article_df.append({'text' : data ,
                                                        'Author' : str(filepath.split("/")[3])} , ignore_index=True)

# article level test

test_file_article_df = pd.DataFrame(columns=['text','Author'])
temp_df = pd.DataFrame(columns=['text','Author'])

for i, filepath in enumerate(glob.iglob('/content/C50test/*/*.txt')):
  print('Analysing', i , ' -> ', ' of Author ', str(filepath.split("/")[3]))
  with open(filepath, 'r') as myfile:
    data = myfile.read()
  # tokens = word_tokenize(data)
  test_file_article_df = test_file_article_df.append({'text' : data ,
                                                        'Author' : str(filepath.split("/")[3])} , ignore_index=True)

train_file_article_df = train_file_article_df.sample(frac=1).reset_index(drop=True)
# data = data.drop(columns='level_0')
train_file_article_df

test_file_article_df = test_file_article_df.sample(frac=1).reset_index(drop=True)
# data = data.drop(columns='level_0')
test_file_article_df

data = pd.concat([train_file_article_df, test_file_article_df], ignore_index=True)

freq = pd.DataFrame({'Author':list(set(train_file_article_df.Author)),
                     'nostoextract':[70]*50, })

def bootstrap(data, freq):
    freq = freq.set_index('Author')

    # This function will be applied on each group of instances of the same
    # class in `data`.
    def sampleClass(classgroup):
        cls = classgroup['Author'].iloc[0]
        nDesired = freq.nostoextract[cls]
        nRows = len(classgroup)

        nSamples = min(nRows, nDesired)
        return classgroup.sample(nSamples)

    samples = data.groupby('Author').apply(sampleClass)

    # If you want a new index with ascending values
    # samples.index = range(len(samples))

    # If you want an index which is equal to the row in `data` where the sample
    # came from
    samples.index = samples.index.get_level_values(1)

    # If you don't change it then you'll have a multiindex with level 0
    # being the class and level 1 being the row in `data` where
    # the sample came from.

    return samples
mega_train = bootstrap(data,freq)

mega_train

mega_test = None
mega_train = mega_train.reset_index()
data = data.reset_index()

my_tot = range(5000)
row_list = list(mega_train['index'])

sep_list = []
for i in my_tot:
  if (i not in row_list):
    sep_list.append(1)
  else:
    sep_list.append(0)


data['hopeful_test'] = sep_list
mega_test = data[data['hopeful_test'] == 1]
mega_test.Author.value_counts()

mega_train.Author.value_counts()

mega_train['index']

data

row_list = list(mega_train['index'])
data.loc[row_list]



mega_test = data[data['hopeful_test'] == 1]

mega_test.Author.value_counts()

my_tot = range(5000)

sep_list = []
for i in my_tot:
  if (i not in row_list):
    sep_list.append(1)
  else:
    sep_list.append(0)


data['hopeful_test'] = sep_list
mega_test = data[data['hopeful_test'] == 1]
mega_test.Author.value_counts()

train_file_article_df['period'] = train_file_article_df['text'].apply(lambda x : x.count("."))

# train_file_article_df.iloc[0][0].count('.')

def apply_to_all_dfs(df):
  df['period'] = df['text'].apply(lambda x : x.count("."))
  df['comma'] = df['text'].apply(lambda x : x.count(","))

  etc, etc ................

with open('/content/C50train/AaronPressman/2537newsML.txt', 'r') as myfile:
  data = myfile.read()
# data.count('\n\n') + 1

lines = data.split('\n')
count = 0
if not line.strip() == '':
    count += 1

print(count)

data

"""## Test Preprocess Article Level"""

# article level

test_file_article_df = pd.DataFrame(columns=['text','Author'])
temp_df = pd.DataFrame(columns=['text','Author'])

for i, filepath in enumerate(glob.iglob('/content/C50test/*.txt')):
  print('Analysing', i , ' -> ', ' of Author ', str(filepath.split("/")[3]))
  with open(filepath, 'r') as myfile:
    data = myfile.read()
  tokens = word_tokenize(data)
  test_file_article_df = test_file_article_df.append({'text' : tokens ,
                                                        'Author' : str(filepath.split("/")[3])} , ignore_index=True)


test_file_article_df.to_csv('test_article_lvl.csv', header=True, index=False)

test_file_article_df

# create neew col with author number mappings for ML
auth_sort = sorted(train_file_df['Author'].unique())
dictOfAuthors = { i : auth_sort[i] for i in range(0, len(auth_sort) ) }
swap_dict = {value:key for key, value in dictOfAuthors.items()}
train_file_df['Author_num'] = train_file_df['Author'].map(swap_dict)

"""##TEST PRE-PROCESS"""

test_file_df = pd.DataFrame()
temp_df = pd.DataFrame()

for i, filepath in enumerate(glob.iglob('/content/C50test/*/*.txt')):
    print('Analysing', i , ' -> ', ' of Author ', str(filepath.split("/")[3]))
    temp_df = pd.read_csv(filepath, delimiter="\t", header = None, error_bad_lines=False, quoting = csv.QUOTE_NONE, encoding='utf-8')
    temp_df['Author'] = filepath.split("/")[3]
    temp_frame = [test_file_df, temp_df]
    test_file_df = pd.concat(temp_frame)

test_file_df = test_file_df.rename(columns={ 0 : 'text'})
test_file_df.head(3)

# avg number of sentences of each author in test data
test_file_df.groupby('Author').count().mean()

# count of sentences of each author in test data
test_file_df.groupby('Author').count()

test_file_df['Author_num'] = test_file_df['Author'].map(swap_dict)

test_file_df['Author'].unique()

"""## Ranjani Stop Words"""



nltk.download('stopwords')
vocab_final = test_file_df['text']


stop = stopwords.words('english')
#print(stop)
test_file_df['text'] = test_file_df.text.str.lower()    #converting to lower case
test_file_df["text"] = test_file_df['text'].str.replace('[^\w\s]','')     #removing punctuations
test_file_df['tokenized_sents'] = test_file_df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
print(test_file_df['tokenized_sents'])
print(" ")
print(test_file_df['tokenized_sents'].apply(lambda x: [item for item in x if item not in stop]))


#Creating Bag of words
def make_bag_of_words(input_df):

  import collections, re
  texts = input_df
  bagsofwords = [ collections.Counter(re.findall(r'\w+', txt))
              for txt in texts]
  temp = remove_stopwords(input_df)
  dict_bag = collections.Counter([y for x in input_df.text.values.flatten() for y in x.split()])
  dict_bag = pd.DataFrame.from_dict(dict_bag, orient='index').reset_index()

  return dict_bag

make_bag_of_words(test_file_df)

"""## ARTH Hist of characters"""

def get_char_hist(auth_name, which_set):
  data = []
  path = '/content/C50' + str(which_set) + '/' + str(auth_name) + '/*.txt'
  for i, filepath in enumerate(glob.iglob(path)):
      # print('Analysing', i , ' -> ', ' of Author ', str(filepath.split("/")[3]))
      with open(filepath, 'r') as file:
        data.append(file.read().replace('\n', ''))

  seperator = ','
  all_data = seperator.join(data)
  figure(num=None, figsize=(15, 7), dpi=80, facecolor='w', edgecolor='k')
  valid_all_data = re.sub(r"[^A-Za-z]+", '', all_data)
  sentence = valid_all_data.lower()

  # Convert the string to an array of integers
  numbers = np.array([ord(c) for c in sentence])
  u = np.unique(numbers)
  # Make the integers range from 0 to n so there are no gaps in the histogram
  # [0][0] was a hack to make sure `np.where` returned an int instead of an array.
  ind = [np.where(u==n)[0][0] for n in numbers]
  bins = range(0,len(u)+1)
  hist, bins = np.histogram(ind, bins)

  plt.bar(bins[:-1], hist, align='center')
  plt.xticks(np.unique(ind)[:], [str(chr(n)) for n in set(numbers)])
  plt.grid()
  plt.show()

get_char_hist('AlanCrosby', 'test')

get_char_hist('LydiaZajc', 'train')

master_author_list = ['KevinMorrison', 'AlanCrosby', 'ToddNissen', 'KevinDrawbaugh',
                      'KouroshKarimkhany', 'JohnMastrini', 'KeithWeir', 'AaronPressman',
                      'ScottHillis', 'HeatherScoffield', 'PeterHumphrey',
                      'PatriciaCommins', 'BernardHickey', 'MarcelMichelson',
                      'GrahamEarnshaw', 'JoeOrtiz', 'SarahDavison', 'SimonCowell',
                      'WilliamKazer', 'JoWinterbottom', "LynneO'Donnell", 'JimGilchrist',
                      'MatthewBunce', 'LydiaZajc', 'AlexanderSmith', 'EricAuchard',
                      'TheresePoletti', 'BradDorfman', 'KarlPenhaul', 'NickLouth',
                      'PierreTran', 'MichaelConnor', 'LynnleyBrowning', 'EdnaFernandes',
                      'MarkBendeich', 'MartinWolk', 'RobinSidel', 'RogerFillion',
                      'SamuelPerry', 'DavidLawder', 'JanLopatka', 'BenjaminKangLim',
                      'DarrenSchuettler', 'MureDickie', 'TanEeLyn', 'TimFarrand',
                      'KirstinRidley', 'JonathanBirt', 'JaneMacartney', 'FumikoFujisaki']

def give_me_everything(author):
  data = []
  list_of_dicts= []
  list_alpha = list(string.ascii_lowercase)
  path = '/content/C50train/' + str(author) + '/*.txt'
  for i, filepath in enumerate(glob.iglob(path)):
      with open(filepath, 'r') as file:
        read_file = file.read().replace('\n', '')
        clean_file = re.sub(r"[^A-Za-z]+", '', read_file)
        arth = {}
        for keys in clean_file.lower():
          arth[keys] = arth.get(keys, 0) + 1
        for item in list_alpha:
          if (item not in arth.keys()):
            arth.update( {str(item) : 0} )
        sorted_dict = {k: arth[k] for k in sorted(arth)}
        list_of_dicts.append(sorted_dict)
  aut_feature = []
  for i in range(0,26,1):
    a_list = []
    for dict_ in list_of_dicts:
      a_list.append(list(dict_.values())[i])
    # print("median value appended for char", list(dict_.keys())[i], "is",  np.median(a_list))
    aut_feature.append(np.median(a_list))
  return aut_feature

feature_final = []
for author in master_author_list:
  feature_final.append(give_me_everything(author))
features_df = pd.DataFrame(feature_final)
features_df = features_df.drop(columns={0, 4 ,8 ,14 ,20}) # vowles

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(features_df)
# print(scaler.mean_)

transformed_df = scaler.transform(features_df)

test_feature_final = scaler.transform(test_feature_final)



X = transformed_df
y = range(0,50,1)

max_val = 0
for i in range(0,2500,1):
  ttt = np.dot

"""## Arth TEST SVC Model"""

def give_me_everything_to_test(author):
  data = []
  list_of_dicts= []
  list_alpha = list(string.ascii_lowercase)
  path = '/content/C50test/' + str(author) + '/*.txt'
  for i, filepath in enumerate(glob.iglob(path)):
      with open(filepath, 'r') as file:
        read_file = file.read().replace('\n', '')
        clean_file = re.sub(r"[^A-Za-z]+", '', read_file)
        arth = {}
        for keys in clean_file.lower():
          arth[keys] = arth.get(keys, 0) + 1
        for item in list_alpha:
          if (item not in arth.keys()):
            arth.update( {str(item) : 0} )
        sorted_dict = {k: arth[k] for k in sorted(arth)}
        list_of_dicts.append(sorted_dict)
  return list_of_dicts

test_feature_final = pd.DataFrame()
for author in master_author_list:
  aut_df = give_me_everything_to_test(str(author))
  aut_df = pd.DataFrame(aut_df)
  frames = [test_feature_final, aut_df]
  test_feature_final = pd.concat(frames)
test_feature_final = test_feature_final.drop(columns={'a', 'e' ,'i' ,'o' ,'u'})

test_feature_final

transformed_test_df = scaler.transform(test_feature_final)

fin_temp = []
for i in range(0,50,1):
  for j in range(0,50,1):
    fin_temp.append(i)

y_pred = clf.predict(transformed_test_df)

(fin_temp == y_pred).sum()/ len(fin_temp)

"""## new training approach"""

def give_me_everything_to_train(author):
  data = []
  list_of_dicts= []
  list_alpha = list(string.ascii_lowercase)
  path = '/content/C50train/' + str(author) + '/*.txt'
  for i, filepath in enumerate(glob.iglob(path)):
      with open(filepath, 'r') as file:
        read_file = file.read().replace('\n', '')
        clean_file = re.sub(r"[^A-Za-z]+", '', read_file)
        arth = {}
        for keys in clean_file.lower():
          arth[keys] = arth.get(keys, 0) + 1
        for item in list_alpha:
          if (item not in arth.keys()):
            arth.update( {str(item) : 0} )
        sorted_dict = {k: arth[k] for k in sorted(arth)}
        list_of_dicts.append(sorted_dict)
  return list_of_dicts

train_feature_final = 0
train_feature_final = pd.DataFrame()
for author in master_author_list:
  aut_df = give_me_everything_to_train(str(author))
  aut_df = pd.DataFrame(aut_df)
  frames = [test_feature_final, aut_df]
  train_feature_final = pd.concat(frames)

scaler = StandardScaler()
scaler.fit(train_feature_final)
# print(scaler.mean_)

transformed_train_df = scaler.transform(train_feature_final)

"""## K means"""

# train
X_train = transformed_train_df
y_train = range(0,50,1)
#test
X_test = transformed_test_df
y_test = fin_temp

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=26, init='k-means++',  n_init = 20, random_state=0).fit(X_train)

y_predict_k = kmeans.predict(X_test)
(y_test == y_predict_k).sum()/ len(y_test)

(X_train.shape, len(y_test))

from sklearn.svm import SVC

clf_new = SVC(gamma='auto')
clf_new.fit(X_train, y_test)

y_predict_svm = clf_new.predict(X_test)
(y_test == y_predict_svm).sum()/ len(y_test)

"""## TF - IDF Approach..."""

# %load_ext autotime

def convert_lower_case(data):
    return np.char.lower(data)

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def stemming(data):
    stemmer= PorterStemmer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text

def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = stemming(data) #needed again as we need to stem the words
    data = remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one
    return data

for i in dataset[:N]:
    file = open(i[0], 'r', encoding="utf8", errors='ignore')
    text = file.read().strip()
    file.close()

    processed_text.append(word_tokenize(str(preprocess(text))))
    processed_title.append(word_tokenize(str(preprocess(i[1]))))
