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
st.write('#Welcome to Text Emotion Analyzer')
st.warning('You have only 10 times use, after the 10th try the result will be deleted')


#database functions
conn = sqlite3.connect('data.db')
c = conn.cursor()      		
def create_table():
	c.execute('CREATE TABLE IF NOT EXISTS nlptable(text_to_analyze TEXT)')


def add_data(text_to_analyze):
	c.execute('INSERT INTO nlptable(text_to_analyze) VALUES (?)',(text_to_analyze,))
	conn.commit()  

def view_all_notes():
	c.execute('SELECT * FROM nlptable')
	data = c.fetchall()
	return data  


def remove_table():
  c.execute("DELETE FROM nlptable WHERE text_to_analyze like 11%  ") #delete the row that begin with 11

create_table()
mxlen =100
numwords = 1000
model = keras.models.load_model('nlp_emotion_analysis.h5')
sentence = [st.text_input('Here the text to analyze')]
tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=mxlen, padding='post', truncating= 'post')
prediction_index = np.argmax(model.predict(padded)[0])
count =1 #helping us to control database rows number
result = "{} That was a tone of {}".format(count,pre(prediction_index))


if st.button('Analyze'):  
    add_data(result)
    remove_table()
    results = view_all_notes()
    count+=1
    col1, col2 = st.columns(2)
    with col1:
        for i in results:
            st.write('##' + i[0][1:]) #this [1:] for not showing the number which help us in remove_table() function
    with col2:	
        for i, result in enumerate(results):
            i+1
            st.write("This is the result number {}".format(i))
