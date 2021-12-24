# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:50:25 2019

@author: Mohammad_Younesi
"""

import numpy as np
import pandas as pd
from collections import Counter 
import matplotlib.pyplot as plt


comments = pd.read_csv("data.csv" , encoding="utf8")
comments.head()
dataC = comments.comment;
dataA = comments.advantages;
dataD = comments.disadvantages;
label = comments.verification_status;

Nanads=0
adsWithNodis=0
addNodisVer=0
Nandis=0
adWithAcc=0
oddCm = []
badA = ['[\\"ندارد\\"]','[\\"نداره\\"]']
veryOdCm=[]
khoobever=0
m=m1=k1=k=0
ajab=[]
for i in range(len(dataA)):
     if (pd.isnull(dataA[i])):
         Nanads += 1
     else:
         if(pd.isnull(dataD[i])):
             adsWithNodis +=1
             
             
             if(label[i] == 'verified'):
                 addNodisVer += 1
             else:
                 veryOdCm.append([dataC[i]])
        
         if((dataA[i] in badA) & pd.isnull(dataD[i])):
             m+=1
             if(label[i] == 'rejected'):
                 k+=1
             else:
                 ajab.append(dataC[i])
         if((dataA[i] in badA)):
             m1+=1
             if(label[i] == 'rejected'):
                 k1+=1
                         
         if(label[i] == 'verified'):
             adWithAcc += 1
         else:
             oddCm.append([dataC[i]])




from hazm import *

import codecs
stopwords = set(line.strip() for line in codecs.open('stopwords-Farsi2.txt','r',encoding='utf8'))

split_C=[];
dataCC = [];
j=0;
k=0;

badComments = pd.Series();
goodComments = pd.Series();
for i in range(len(label)):
    if (label[i] == 'rejected'):
        badComments.set_value(j, dataC[i])
        j+=1
    else:
        goodComments.set_value(k,dataC[i])
        k+=1
        
normalizer = Normalizer()
for word in set(stopwords):
    word = normalizer.normalize(word)

bad_split = []
good_split = []
A = []
split_C=[];
j=0;
k=0;



for i in range(len(dataC)):
    if (not pd.isnull(dataC[i]) ):
        dataC[i] = normalizer.normalize(dataC[i])
        split_C.append(str(dataC[i].split()))
        

for i in range(len(badComments)):
    if (not pd.isnull(badComments[i])):
        badComments[i] = normalizer.normalize(badComments[i])
        bad_split.append([str(badComments[i].split())])


for i in range(len(bad_split)):
    bad_split[i] = bad_split[i][0].split(",")
    bad_split[i][0] = bad_split[i][0][1:]
    bad_split[i][len(bad_split[i])-1] = bad_split[i][len(bad_split[i])-1][:-1]
    bad_split[i][0] = bad_split[i][0][1:-1]
    for m in range(len(bad_split[i])-1):
        bad_split[i][m+1] = bad_split[i][m+1][2:-1]
   
allBad = []
for i in range(15000):
    allBad = allBad + bad_split[i]   
        
for i in range(len(goodComments)):
    if (not pd.isnull(goodComments[i])):
        goodComments[i] = normalizer.normalize(goodComments[i])
        good_split.append([str(goodComments[i].split())])


for i in range(len(good_split)):
    good_split[i] = good_split[i][0].split(",")
    good_split[i][0] = good_split[i][0][1:]
    good_split[i][len(good_split[i])-1] = good_split[i][len(good_split[i])-1][:-1]
    good_split[i][0] = good_split[i][0][1:-1]
    for m in range(len(good_split[i])-1):
        good_split[i][m+1] = good_split[i][m+1][2:-1]
       
allGood = []
for i in range(25000):
    allGood = allGood + good_split[i]  

counter1 = Counter(allBad) 
most_occur1 = counter1.most_common()[0:50]
counter2 = Counter(allGood) 
most_occur2 = counter2.most_common()[0:50] 
print(most_occur1)
print(most_occur2)  


import nltk
from nltk.stem.isri import ISRIStemmer
st = ISRIStemmer()
w= 'حركات'
print(st.stem(w))

from wordcloud_fa import WordCloudFa

wc = WordCloudFa(width=1200, height=800)
#mmss = wc.stopwords
#for word in set(mmss):
#    word = word.encode()
#    word = normalizer.normalize(word)
#    
#    
#stopwords = stopwords.union(mmss)

MrKhodaeeStop = set(['اتفاقا','احتراما','احتمالا','اري','از','ازجمله','اساسا','است','اش','اشكارا','اصلا','اصولا','اغلب'
                     ,'اكثرا','اكنون','الان','البته','اما','امد','امدم','امدن','امدند','امده','امدي','امديد','امديم','امروزه','امسال','امشب','ان','اند','انشاالله','انصافا','انطور','انقدر','انها','انچنان','انگار','او','اورد','اوردم'
                     ,'اوردن','اوردند','اورده','اوردي','اورديد','اورديم','اورم','اورند','اوري','اوريد','اوريم','اولا','اي','ايا','ايد','ايشان','ايم','اين','ايند','اينطور','اينقدر','اينك','اينها','اينچنين','اينگونه','ايي','اييد','اييم'
                     ,'اگر','با','بار','بارها','باز','بازهم','باش','باشد','باشم','باشند','باشي','باشيد','باشيم','بالاخره','بالطبع','بايد','بتوان','بتواند','بتواني','بتوانيم','بخواه','بخواهد','بخواهم','بخواهند'
                     ,'بخواهي','بخواهيد','بخواهيم','بد','بدون','بر','براحتي','براستي','براي','برعكس','بزودي','بسا','بسيار','بعدا','بعدها','بعضا','بكن','بكند','بكنم','بكنند','بكني','بكنيد','بكنيم','بلافاصله'
                     ,'بلي','به','بهتر','بود','بودم','بودن','بودند','بوده','بودي','بوديد','بوديم','بويژه','بي','بيا','بياب','بيابد','بيابم','بيابند','بيابي','بيابيد','بيابيم','بياور','بياورد','بياورم','بياورند'
                     ,'بياوري','بياوريد','بياوريم','بيايد','بيايم','بيايند','بيايي','بياييد','بياييم','بيشتر','بيشتري','بين','بگو','بگويد','بگويم','بگويند','بگويي','بگوييد','بگوييم','بگير'
                     ,'بگيرد','بگيرم','بگيرند','بگيري','بگيريد','بگيريم','ت','تا','تاكنون','تان','تحت','تر','تقريبا','تلويحا','تماما','تنها','تو','تواند','توانست','توانستم','توانستن'
                     ,'توانستند','توانسته','توانستي','توانستيم','توانم','توانند','تواني','توانيد','توانيم','ثانيا','جمعا','حالا','حتما','حتي','حداكثر','حدودا','خب','خصوصا','خلاصه','خواست'
                     ,'خواستم','خواستن','خواستند','خواسته','خواستي','خواستيد','خواستيم','خواهد','خواهم','خواهند','خواهي','خواهيد','خواهيم','خوب','خود','خودت','خودتان'
                     ,'خودش','خودشان','خودم','خودمان','خوشبختانه','خويش','خويشتن','خير','داد','دادم','دادند','داده','دادي','داديد','داديم','دار','دارد'
                     ,'دارم','دارند','داري','داريد','داريم','داشت','داشتم','داشتن','داشتند','داشته','داشتي','داشتيد','داشتيم','دايم','دايما','در','درباره','درمجموع','دريغ'
                     ,'دقيقا','دهد','دهم','دهند','دهي','دهيد','دهيم','دو','دوباره','دير','ديروز','ديگر','ديگري','را','راحت','راسا','راستي','رسما'
                     ,'رو','روزانه','روي','زود','زير','سالانه','ساليانه','سرانجام','سريعا','سپس','شان','شايد','شخصا','شد','شدم','شدن'
                     ,'شدند','شده','شدي','شديد','شديدا','شديم','شما','شود','شوم','شوند','شونده','شوي','شويد','شويم','صرفا'
                     ,'ضمن','طبعا','طبيعتا','طور','طي','ظاهرا','عمدا','عمدتا','عملا','غالبا','فردا','فعلا','فقط','قبلا','قدري','قطعا'
                     ,'كاش','كاملا','كتبا','كجا','كرد','كردم','كردن','كردند','كرده','كردي','كرديد','كرديم','كس','كسي','كلا','كم'
                     ,'كماكان','كمتر','كمتري','كمي','كن','كند','كنم','كنند','كننده','كنون','كني','كنيد','كنيم','كه','كو'
                     ,'كي','لااقل','لطفا','ما','مان','مانند','مبادا','متاسفانه','متعاقبا','مثل','مثلا','مجاني','مجددا','مجموعا'
                     ,'مدام','مستقيما','مسلما','مطمينا','معمولا','من','موقتا','مي','مگر','ناگاه','ناگهان','ناگهاني','نبايد'
                     ,'نخواهد','نخواهم','نخواهند','نخواهي','نخواهيد','نخواهيم','ندارد','ندارم','ندارند','نداري','نداريد'
                     ,'نداريم','نداشت','نداشتم','نداشتند','نداشتي','نداشتيد','نداشتيم','نسبتا','نشده','نظير','نمي','نه'
                     ,'نهايتا','نيز','نيست','ها','هاي','هايي','هر','هرچه','هست','هستم','هستند','هستي'
                     ,'هستيد','هستيم','هم','همان','همه','همواره','هميشه','همين','همچنان'
                     ,'همچنين','هنوز','هيچ','هيچگاه','و','واقعا','ولي','وي','ي','يا','يابد','يابم'
                     ,'يابند','يابي','يابيد','يابيم','يافت','يافتم','يافتن','يافته','يافتي','يافتيد'
                     ,'يافتيم','يقينا','يك','پارسال','پس','پيش','پيشاپيش','پيشتر','چرا'
                     ,'چطور','چقدر','چنان','چنانكه','چنانچه','چند','چنين','چه','چو','چون'
                     ,'چيز','چگونه','گاه','گاهي','گرفت','گرفتم','گرفتن','گرفتند','گرفته'
                     ,'گرفتي','گرفتيد','گرفتيم','گفت','گفتم','گفتن','گفتند','گفته','گفتي'
                     ,'گفتيد','گفتيم','گه','گهگاه','گو','گويا','گويد','گويم','گويند'
                     ,'گويي','گوييد','گوييم','گيرد','گيرم','گيرند','گيري','گيريد','گيريم'])

    
myStop = stopwords.union(MrKhodaeeStop)
badReview = [word for word in allBad
          if not word in set(myStop)]
goodReview = [word for word in allGood
          if not word in set(myStop)]

counter3 = Counter(badReview) 
myStop = [counter3.most_common()[0][0]]
myStop.append(counter3.most_common()[4][0] )
myStop.append(counter3.most_common()[6][0] )
myStop.append(counter3.most_common()[9][0] )
myStop.append(counter3.most_common()[11][0] )
myStop.append(counter3.most_common()[12][0] )
myStop.append(counter3.most_common()[14][0] )
myStop.append(counter3.most_common()[15][0] )
myStop.append(counter3.most_common()[17][0] )
myStop.append(counter3.most_common()[21][0] )
myStop.append(counter3.most_common()[23][0] )
myStop.append(counter3.most_common()[28][0] )
myStop.append(counter3.most_common()[30][0] )
myStop.append(counter3.most_common()[32][0] )
myStop.append(counter3.most_common()[31][0] )
myStop.append(counter3.most_common()[35][0] )
myStop.append(counter3.most_common()[37][0] )
myStop.append(counter3.most_common()[38][0] )
myStop.append(counter3.most_common()[39][0] )

goodReview = [word for word in goodReview
          if not word in myStop]
badReview = [word for word in badReview
          if not word in myStop]

counter3 = Counter(badReview) 
most_occur3 = counter3.most_common()[0:50] 
counter4 = Counter(goodReview) 
most_occur4 = counter4.most_common()[0:50] 

print(most_occur3) 
print(most_occur4) 



myBad = open('badReview15K.txt' , 'w', encoding="utf-8")
myGood = open('goodReview25K.txt' , 'w', encoding="utf-8")

for element in badReview:
    myBad.write(element)
    myBad.write('\n')
myBad.close()
for element in goodReview:
    myGood.write(element)
    myGood.write('\n')
myGood.close()

with codecs.open('badReview15K.txt','r',encoding='utf8') as file:
    text = file.read()


word_cloud = wc.generate(text)
image = word_cloud.to_image()
image.show()
image.save('badCloud15.png')

with codecs.open('goodReview25K.txt','r',encoding='utf8') as file:
    text = file.read()


word_cloud = wc.generate(text)
image = word_cloud.to_image()
image.show()
image.save('goodCloud25.png')
























plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off');
plt.show()
print('')
