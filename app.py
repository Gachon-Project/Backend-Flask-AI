import io
from pickletools import read_uint1
from unittest import result
from torchvision import models
import json
from flask import Flask, jsonify, request
from flask import make_response
import torchvision.transforms as transforms
import torch
from PIL import Image
import os
import requests
import logging

app = Flask(__name__)

# yolo model 불러오기
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')

# Node.js 서버 주소 저장
node_url = "Node.js 서버IP주소/ingredients"
headers = {"Content-Type": "application/json; charset=utf-8"}

# Node.js 서버에 YOLO결과값을 보내는 Post 함수
def send_data_node(node_url, data):
    res = requests.post(node_url, headers=headers, data=data)


# 이미지를 저장하는 함수
def save_image(file):
    file.save('./temp/'+ file.filename)

# 기본 URL
@app.route('/')
def web():
    return "Lungnaha's flask test page"

# POST 통신으로 들어오는 이미지를 저장하고 모델로 추론하는 과정
@app.route('/predict', methods=['POST'])
def predict():
    logging.info("predict")
    if request.method == 'POST':
        file = request.files.get('file')
        logging.info(file)

        save_image(file)
        train_img = './temp/' + file.filename
        temp = model(train_img)
        result = temp.pandas().xyxy[0]['name'].to_json(orient="records") 
        name = file.filename
        logging.info(type(result))

        result = json.loads(result)
        logging.info(type(result))
        
        M = dict(zip(range(1, len(result) + 1), result))
        M = json.dumps(M)
        logging.info(M)

        send_data_node(node_url, M)
        logging.info(type(M))
        return M

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)