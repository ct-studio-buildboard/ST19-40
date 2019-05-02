# -*- coding: utf-8 -*-
from flask import Flask
from flask import Flask, send_from_directory, jsonify
from flask import render_template
from flask import redirect
from flask import url_for
import json
import random

app = Flask(__name__)
app.config.from_object(__name__)
from flask import request
import scipy.misc
import numpy as np
from models.run_model_wrapper import Model_Handler
from models.helper_functions import show_image

model_G=Model_Handler("../full_models/gen_D")

@app.route('/')
def index():
    return send_from_directory("views", "main.html")

from binascii import a2b_base64
import base64
from base64 import b64decode
import io
@app.route('/api/generate_image',methods=['GET', 'POST'])
def get_dummy_results():
   print ("GREAT")
   print(request.form['jsonData'])
   data=json.loads(request.form['jsonData'])
   print(data)
   relevant=data["user_id"].split("base64,",1)[1].replace(" ","+")
   #relevant=#relevant.split("base64,/")[1]
   print(relevant)
   binary_data = b64decode(relevant)

   #im = Image.open(io.BytesIO(base64.b64decode(data["user_id"].split(',')[1])))
   #im.save("image.jpg")

   with open("imageToSave5.jpg", "wb") as fh:
       #fh.write(binary_data)
       fh.write(binary_data)
   fh.close()

   import os
   temp_whut = (model_G.predict("imageToSave5.jpg"))
   #show_image(temp_whut[0].T)
   #print (os.getcwd())
   scipy.misc.imsave("static/x_final.jpg", np.fliplr(np.rot90(((temp_whut[0].T+1)*255).astype(int),3)))
   #print data_fake_positive_messages
   return json.dumps({"f":1})
   #return json.dumps( "fwe:""KILL ME")



if __name__ == '__main__':
    app.run(    )
