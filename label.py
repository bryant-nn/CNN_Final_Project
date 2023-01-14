import json
from transformers import pipeline
with open('courseofficialfeedbackv2.json') as f:
    data = json.load(f)
sentences = []
indexs=[]
for i in range(len(data)):
    if ('description' in data[i]):
        if(data[i]['description']!='無'):
            s=data[i]['enDescription']
            sentence=""
            for j in range(len(s)):
                if(ord(s[j])>=0 and ord(s[j])<=127):
                    sentence+=s[j]
            sentences.append(sentence)
            indexs.append(i)
# print(indexs)
classifier = pipeline('sentiment-analysis',model="distilbert-base-uncased-finetuned-sst-2-english")
results = classifier(sentences)
for i,index in enumerate(indexs):
    data[index]['distilbert-base-uncased-finetuned-sst-2-english'] = dict()
    data[index]['distilbert-base-uncased-finetuned-sst-2-english']['label'] = results[i]['label']
    if(results[i]['label']=='POSITIVE'):
        data[index]['distilbert-base-uncased-finetuned-sst-2-english']['PositiveScore'] = results[i]['score']
        data[index]['distilbert-base-uncased-finetuned-sst-2-english']['NegativeScore'] = 1-results[i]['score']
    else:
        data[index]['distilbert-base-uncased-finetuned-sst-2-english']['PositiveScore'] = 1-results[i]['score']
        data[index]['distilbert-base-uncased-finetuned-sst-2-english']['NegativeScore'] = results[i]['score']
with open("courseofficialfeedbackv2.json", "w") as f:
    json.dump(data, f,ensure_ascii=False, indent = 4)