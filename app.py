from functions import pre
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import sqlite3
import pickle
st.set_page_config(layout ="wide") 
st.write('# Welcome to Text Emotion Analyzer')
st.warning('You have only 10 times use, after the 10th try the result will be deleted')


#database functions
conn = sqlite3.connect('data.db')
c = conn.cursor()      		
def create_table():
	c.execute('CREATE TABLE IF NOT EXISTS nlptable(inputxt TEXT, text_to_analyze TEXT)')


def add_data(inputxt,text_to_analyze):
	c.execute('INSERT INTO nlptable(inputxt,text_to_analyze) VALUES (?,?)',(inputxt,text_to_analyze))
	conn.commit()  

def view_all_notes():
	c.execute('SELECT * FROM nlptable')
	data = c.fetchall()
	return data  


def remove_table():
  c.execute("DELETE FROM nlptable")
  conn.commit()
  

create_table()   
mxlen =100
numwords = 1000
model = keras.models.load_model('nlp_emotion_analysis.h5')
inputxt = st.text_input('Here the text to analyze')
sentence = [inputxt]
tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=mxlen, padding='post', truncating= 'post')
prediction_index = np.argmax(model.predict(padded)[0])
result = " That was a tone of {}".format(pre(prediction_index))



if st.button('Analyze'):  
    add_data(inputxt,result)
    results = view_all_notes() #json format
    col1, col2,col3 = st.columns(3)
    with col1:
        for i in results:
            st.write(i[0])
    with col2:
      for i in results:
        st.write(i[1])

    with col3:	
        for i, result in enumerate(results):
            i+=1
            if i ==10: #controling num of rows in database 
              remove_table()
              conn = sqlite3.connect('data.db')
              c = conn.cursor() 

            st.write("This is the result number {}".format(i))
