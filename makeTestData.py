import json
import csv
with open('courseofficialfeedbackv2.json') as f:
    data = json.load(f)
coursename=dict()
coursenameid=0
prof=dict()
profid=0
language=dict()
languagelist=[]
languageid=0
coursetype=dict()
coursetype['通識']=0
coursetypeid=1
tempdata=[]
for i in range(len(data)):
    if ('description' in data[i] and 'typeOfCredit' in data[i]['courseObj'] 
        and 'professor' in data[i] and 'lectureLanguage' in data[i]['courseObj'] and 'course_name' in data[i]):  #過濾資料到tempdata
        if(data[i]['description']!='無'):
            tempdata.append(data[i])
trainlength=int((len(tempdata)/10)*7)
for i in range(len(tempdata)):
    if(tempdata[i]['course_name'] not in coursename.keys()):    
        coursename[tempdata[i]['course_name']]=coursenameid
        coursenameid+=1
    if(tempdata[i]['professor'] not in prof.keys()):    
        prof[tempdata[i]['professor']]=profid
        profid+=1
    langstr=tempdata[i]['courseObj']['lectureLanguage'][0]+tempdata[i]['courseObj']['lectureLanguage'][1]
    languagelist.append(langstr)
    if(langstr not in language.keys()):
        language[langstr]=languageid
        languageid+=1
    if(tempdata[i]['courseObj']['typeOfCredit'] not in coursetype.keys()):
        coursetype[tempdata[i]['courseObj']['typeOfCredit']]=coursetypeid
        coursetypeid+=1

traindata=[]
testdata=[]
traindata.append(["_id", "description", "enDescription", "course_name", "professor", "courseObj.lectureLanguage", "courseObj.typeOfCredit", "label"])
testdata.append(["_id", "description", "enDescription", "course_name", "professor", "courseObj.lectureLanguage", "courseObj.typeOfCredit", "label"])
for i in range(len(tempdata)):
    data=[]
    data.append(tempdata[i]['_id']['$oid'])
    data.append(tempdata[i]['description'])
    data.append(tempdata[i]['enDescription'])
    data.append(coursename[tempdata[i]['course_name']])
    data.append(prof[tempdata[i]['professor']])
    data.append(language[languagelist[i]])
    data.append(coursetype[tempdata[i]['courseObj']['typeOfCredit']])
    data.append(tempdata[i]['distilbert-base-uncased-finetuned-sst-2-english']['PositiveScore'])
    if(i<=trainlength):
        traindata.append(data)
    else:
        testdata.append(data)
with open('traindata.csv', 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerows(traindata)
with open('testdata.csv', 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerows(testdata)