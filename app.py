 # coding=utf8
from pythainlp import word_tokenize,word_vector # ทำการเรียกตัวตัดคำ
model = word_vector.get_model()
from pythainlp.word_vector import * # ทำการเรียก thai2vec
from sklearn.metrics.pairwise import cosine_similarity  # ใช้หาค่าความคล้ายคลึง
import numpy as np
from flask import Flask, request
from flask_cors import CORS
import json 
import os
   
app = Flask(__name__) 
CORS(app)

txt_len =[]

@app.route("/")
def hello():
    return 'hello,I am flash Application'

@app.route('/checksimilarity_attacut', methods = ['POST']) 
def checking_attacut(): 
    answer_stu =""
    answer = []
    persent_checking=[]
    data = []
    data = request.get_json() 
    answer_stu = (data[0]["answer_stu"])
    wordvec1 = sentence_break_attacut(str(answer_stu))
    for i in data[1:]:
        answer.append(i["answer"])
    for x in answer:
        wordvec2 = sentence_break_attacut(str(x))
        persent_get = sentence_similarity(wordvec2,wordvec1)
        persent_checking.append(persent_get)
        if persent_get==100:
            break
    max_persent = max(persent_checking)
    data[persent_checking.index(max_persent)+1]["persent_get"]=max_persent
    result = data[persent_checking.index(max_persent)+1]
    print(result)
    return json.dumps({"result":result}) #แปลงObject Python เป็นสตริง json

@app.route('/checksimilarity_newmm', methods = ['POST']) 
def checking_newmm(): 
    answer_stu =""
    answer = []
    persent_checking=[]
    data = []
    data = request.get_json() 
    answer_stu = (data[0]["answer_stu"])
    wordvec1 = sentence_break_newmm(str(answer_stu))
    for i in data[1:]:
        answer.append(i["answer"])
    for x in answer:
        wordvec2 = sentence_break_newmm(str(x))
        persent_get = sentence_similarity(wordvec2,wordvec1)
        persent_checking.append(persent_get)
        if persent_get==100:
            break
    max_persent = max(persent_checking)
    data[persent_checking.index(max_persent)+1]["persent_get"]=max_persent
    result = data[persent_checking.index(max_persent)+1]
    print(result)
    return json.dumps({"result":result}) 

@app.route('/checksimilarity_newmmsafe', methods = ['POST']) 
def checking_newmmsafe(): 
    answer_stu =""
    answer = []
    persent_checking=[]
    data = []
    data = request.get_json() 
    answer_stu = (data[0]["answer_stu"])
    wordvec1 = sentence_break_newmmsafe(str(answer_stu))
    for i in data[1:]:
        answer.append(i["answer"])
    for x in answer:
        wordvec2 = sentence_break_newmmsafe(str(x))
        persent_get = sentence_similarity(wordvec2,wordvec1)
        persent_checking.append(persent_get)
        if persent_get==100:
            break
    max_persent = max(persent_checking)
    data[persent_checking.index(max_persent)+1]["persent_get"]=max_persent
    result = data[persent_checking.index(max_persent)+1]
    print(result)
    return json.dumps({"result":result}) 

@app.route('/checksimilarity_jaccard', methods = ['POST']) 
def checking_jaccard(): 
    answer_stu =""
    answer = []
    persent_checking=[]
    data = []
    data = request.get_json() 
    answer_stu = (data[0]["answer_stu"])
    for i in data[1:]:
        answer.append(i["answer"])
    for x in answer:
        persent_get = round(jaccard_similarity(str(x),str(answer_stu))*100,2)
        persent_checking.append(persent_get)
        if persent_get==100:
            break
    max_persent = max(persent_checking)
    data[persent_checking.index(max_persent)+1]["persent_get"]=max_persent
    result = data[persent_checking.index(max_persent)+1]
    print(result)
    return json.dumps({"result":result}) 

@app.route('/checksimilarity_attacut2', methods = ['POST']) 
def checking_attacut2(): 
    answer_stu =""
    answer = []
    persent_checking=[]
    data = []
    txt_len.clear()
    data = request.get_json() 
    answer_stu = (data[0]["answer_stu"])
    wordvec1 = sentence_break_attacut2(str(answer_stu))
    for i in data[1:]:
        answer.append(i["answer"])
    for x in answer:
        wordvec2 = sentence_break_attacut2(str(x))
        persent_get = checksimilarity_txtlen(sentence_similarity(wordvec1,wordvec2))
        persent_checking.append(persent_get)
        del txt_len[1]
        if persent_get==100:
            break
    max_persent = max(persent_checking)
    data[persent_checking.index(max_persent)+1]["persent_get"]=max_persent
    result = data[persent_checking.index(max_persent)+1]
    print(result)
    return json.dumps({"result":result}) #แปลงObject Python เป็นสตริง json


def sentence_break_attacut2(text):
    cut_text = word_tokenize(text.lower(),engine='attacut') 
    vec = np.zeros((1,300))
    for word in cut_text:
        if word in model.index_to_key:
            vec+= model.get_vector(word)
        else: pass
    txt_len.append(len(cut_text))
    vec /= len(cut_text)
    return vec

def checksimilarity_txtlen (persent_process):
    txt_avg = (round((min(txt_len)/max(txt_len))*persent_process,2))
    txt_dif = persent_process-txt_avg
    persent_process -= txt_dif
    if(persent_process<0):persent_process=0
    return round(persent_process,2)

def sentence_break_attacut(text,use_mean=True): 
    cut_text = word_tokenize(text.lower(),engine='attacut') 
    vec = np.zeros((1,300))
    for word in cut_text:
        if word in model.index_to_key:
            vec+= model.get_vector(word)
        else: pass
    if use_mean: vec /= len(cut_text)
    return vec

def sentence_break_newmm(text,use_mean=True): 
    cut_text = word_tokenize(text.lower(),engine='newmm') 
    vec = np.zeros((1,300))
    for word in cut_text:
        if word in model.index_to_key:
            vec+= model.get_vector(word)
        else: pass
    if use_mean: vec /= len(cut_text)
    return vec

def sentence_break_newmmsafe(text,use_mean=True): 
    cut_text = word_tokenize(text.lower(),engine='newmm-safe') 
    vec = np.zeros((1,300))
    for word in cut_text:
        if word in model.index_to_key:
            vec+= model.get_vector(word)
        else: pass
    if use_mean: vec /= len(cut_text)
    return vec

def sentence_similarity(wordvec1,wordvec2):
    like = cosine_similarity(wordvec1,wordvec2)*100
    return round(like[0][0],2)

def jaccard_similarity(x,y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

if __name__ == "__main__": 
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)