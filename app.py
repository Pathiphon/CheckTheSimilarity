from pythainlp import word_tokenize,word_vector # ทำการเรียกตัวตัดคำ
model = word_vector.get_model()
from pythainlp.word_vector import * # ทำการเรียก thai2vec
from sklearn.metrics.pairwise import cosine_similarity  # ใช้หาค่าความคล้ายคลึง
import numpy as np
from flask import Flask, request,render_template
from flask_cors import CORS
import json 
   
app = Flask(__name__) 
CORS(app)

@app.route("/")
def hello():
    return render_template('index.html') 

@app.route('/checksimilarity', methods = ['POST']) 
def checking(): 
    answer_stu =""
    answer = []
    persent_checking=[]
    data = []
    data = request.get_json() 
    answer_stu = (data[0]["answer_stu"])
    for i in data[1:]:
        answer.append(i["answer"])
    for x in answer:
        persent_checking.append(sentence_similarity(x,answer_stu))
    max_persent = max(persent_checking)
    data[persent_checking.index(max_persent)+1]["persent_get"]=max_persent
    result = data[persent_checking.index(max_persent)+1]
    print(result)
    return json.dumps({"result":result}) #แปลงObject Python เป็นสตริง json
   
def sentence_break(text,dim=300,use_mean=True): 
    cut_text = word_tokenize(text,engine='newmm') #mm newmm 
    vec = np.zeros((1,dim))
    for word in cut_text:
        if word in model.index_to_key:
            vec+= model.get_vector(word)
        else: pass
    if use_mean: vec /= len(cut_text)
    return vec

def sentence_similarity(text1,text2):
    like = cosine_similarity(sentence_break(str(text1)),sentence_break(str(text2)))*100
    return round(like[0][0],2)

if __name__ == "__main__": 
    app.run(debug=True)
    # app.run(port=5000)