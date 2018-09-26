import os
import codecs
import chardet
import numpy as np
import math
import re


path='G:/PYCHARM/untitled3/20news-18828'
packs=os.listdir(path)
result1=[]
result2=[]
for pack in packs:
     path1=path+"/"+pack
     print(path1)
     files=os.listdir(path1)
     result1.append(path1)
     for file in files:
          path2=path1+"/"+file
          result2.append(path2)

print('文档数:',len(result2))
#-------------------------------------------------------------------------------------------------------------
#data pretreatment
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer as ss
stp = stopwords.words('english')
stm = ss('english')
symbols= [',','.',':','_','!','?','/','\'','\"','*','>','<','@','~','-','(',')','%','=','\\','^'
     ,'&','|','#','$','0','1','2','3','4','5','6','7','8','9','10','[',']','+','{','}',';','`','~']
glossary={}
wordlist={}
vocab=[]
for d_path in result2:
         d = open(d_path,'rb')
         data = d.read()
         encode = chardet.detect(data)['encoding']
         with codecs.open(d_path, encoding=encode) as d:
             words = d.read()
             for symbol in symbols:
                 words = words.replace(symbol,'')
             words = words.split()
             for word in words:
                 word = stm.stem(word)
                 if word not in stp:
                     if word in wordlist:
                         wordlist[word]+=1
                     else:
                         wordlist[word]=1
print(len(wordlist))
#--------------------------------------------------------------------------------------------------------------------

#过滤掉词频过低以及过高的词
for k in wordlist:
    if wordlist[k] >= 4 and wordlist[k] <= 1000:
        glossary[k]=wordlist[k]
        vocab.append("%s,%s\n" %(k,wordlist[k]))

with open('bagpack.txt','w+',encoding='utf-8') as f:
    f.writelines(vocab)

print(glossary)
print(len(glossary))

#------------------------ ------------------------------------------------------------------------------------------
keys = list(glossary.keys())
D_vec=np.zeros(len(glossary))
for d_path in result2:
    d=open(d_path,'rb')
    lines=d.read()
    lines=set(lines.split())

    for item in lines:
        if item in keys:
            D_vec[keys.index(item)] +=1
    d.close()
print(D_vec)
#-----------------------------------------------------------------------------------------------------------
vec_save ='G:/PYCHARM/untitled3/TF_IDF_VEC.npy'
idf=np.log10(int(18828)/(D_vec+1))

vector=[]
for d_path in result2:
    d=open(d_path,'rb')
    tf=np.zeros(len(glossary))
    lines=d.read()
    lines=lines.split()
    keys_2=list(glossary.keys())
    for item in lines:
        if item in keys_2:
            tf[keys_2.index(item)]+=1
    tf/=len(lines)
    tf_idf=idf*tf
    vector.append(tf_idf)
    d.close()

np.save(vec_save,vector)
