#!/usr/bin/env python3.6
#Copyright 11/2019 Jack K. Rasmus-Vorrath

from argparse_prompt import PromptParser

import platform
import os
import sys

import json
import re

from collections import OrderedDict

import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf

import keras
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import (EarlyStopping, 
			     ModelCheckpoint)
from keras.models import (Model, 
			  Input, 
			  load_model)
from keras.layers import (Embedding, 
			  Conv1D, 
			  LSTM, 
			  Bidirectional, 
			  Dense)
from keras.layers import (Masking, 
			  MaxPooling1D, 
			  SpatialDropout1D, 
			  BatchNormalization)

#########################################################################################################################

def get_args():

	parser = PromptParser()
	parser.add_argument("-D", "--Data", 
			    default="./Data/", 
			    type=str, 
			    help="Data Directory", 
			    prompt=True)
	parser.add_argument("-M", "--Models", 
			    default="./Models/", 
			    type=str, 
			    help="Models Directory", 
			    prompt=True)
	parser.add_argument("-T", "--Train", 
			    default=True, 
			    type=str, 
			    help="Train New Model", 
			    prompt=True)
	parser.add_argument("-P", "--Predict", 
			    default=True, 
			    type=str, 
			    help="Predict using Trained Model", 
			    prompt=True)
	parser.add_argument("-L", "--loadedModel", 
			    default="CNN_BLSTM.h5",
			    help="Pretrained Model Path", 
			    prompt=True)
	parser.add_argument("-G", "--Debug", 
			    default=False, 
			    help="Debug Mode", 
			    prompt=False)

	args = parser.parse_args()
	
	return args

def read_data(args):
	
	csv_file = '{}training.csv'.format(args.Data)
	dataframe = pd.read_csv(csv_file, 
				engine='python', 
				quotechar='|', 
				header=None)
	dataset = dataframe.sample(frac=1).values
	
	X = dataset[:,0]
	y = dataset[:,1]

	for index, item in enumerate(X):
		
		reqJson = json.loads(item, 
				     object_pairs_hook=OrderedDict)
		X[index] = json.dumps(reqJson, 
				      separators=(',', ':'))
		
	return X, y
	
def preprocess(X, 
	       to_tokenize):

	to_exclude = '!"#%&()*+,/:;<=>?@[\\]^_`|{}~\t\n'
	
	tokenizer = Tokenizer(filters=to_exclude, 
			      char_level=True, 
			      lower=False, 
			      oov_token='')
	X_clean = [re.sub(r'(['+to_tokenize+'])', r' ', x) 
		   	for x in X]
	
	tokenizer.fit_on_texts(X_clean)
	
	X_seq = tokenizer.texts_to_sequences(X_clean)
	
	n_unique = len(tokenizer.word_index)+1
	payload_len = max([len(x) 
			   	for x in X_seq])
	
	X_processed = pad_sequences(X_seq, 
				    maxlen=payload_len, 
				    padding='post', 
				    truncating='post')
	
	return X_processed, n_unique, payload_len, tokenizer
	
def build_model(n_unique, 
		payload_len):
	
	#Padded Input
	char_in = Input(shape=(payload_len,))
	
	#Embed Characters
	emb_char = Embedding(input_dim=n_unique, 
			     output_dim=16, 
			     input_length=payload_len)(char_in)

	#1D Convolution + Pooling
	conv1 = Conv1D(kernel_size=4, 
		       filters=16, 
		       padding='same', 
		       activation='relu', 
		       strides=1)(emb_char)
	pool = MaxPooling1D(strides=2)(conv1)
	
	mask = Masking(mask_value=0)(pool)

	#LSTM Hidden Layer
	char_enc = LSTM(units=32, 
			return_sequences=True, 
			dropout=0.5,
			recurrent_dropout=0.5)(mask)

	#1D Spatial Dropout + BLSTM
	x = SpatialDropout1D(0.5)(char_enc)
	main_lstm = Bidirectional(LSTM(units=32, 
				       return_sequences=True, 
				       dropout=0.5,
				       recurrent_dropout=0.5))(x)
	#LSTM Stacked
	out_lstm = LSTM(units=16, 
			return_sequences=False)(main_lstm)

	#Batch Normalization
	bn = BatchNormalization()(out_lstm)

	#Fully Connected sigmoid output
	out = Dense(1, activation="sigmoid")(bn) 

	model = Model([char_in], out)
	
	model.compile(optimizer="rmsprop", 
		      loss="binary_crossentropy",
		      metrics=["acc"])
	
	print(model.summary())
	
	return model
	
def prediction(loaded_model, 
	       val_input, 
	       to_tokenize, 
	       tokenizer, 
	       payload_len):
	
	val_input_clean = [re.sub(r'(['+to_tokenize+'])', r' ', x) 
			   	for x in [val_input]]
	
	X_val_seq = tokenizer.texts_to_sequences(val_input_clean)

	X_val_processed = pad_sequences(X_val_seq, 
					maxlen=payload_len, 
					padding='post', 
					truncating='post')
	
	res = loaded_model.predict(np.array(X_val_processed))
	
	return res
	
#########################################################################################################################

def main():

	args = get_args()

	X, y = read_data(args)
	
	to_tokenize = '"&()*+,/:@[\\]_{}|'
	
	X_processed, n_unique, payload_len, tokenizer = preprocess(X, to_tokenize)
	
	X_train, X_test, y_train, y_test = train_test_split(X_processed, 
							    y, 
							    test_size=0.2,
							    random_state=222)
	
	print('~'*80)
	print(f'\nShape of Training Set is: {X_train.shape}')
	print(f'Shape of Test Set is: {X_test.shape}\n')

	print(f'Shape of Training Target is: {y_train.shape}')
	print(f'Shape of Test Target is: {y_test.shape}\n')
	print('~'*80)
	
	if args.Train not in ['False','false','f','F']:
	
		model = build_model(n_unique, 
				    payload_len)
		
		###################################################
		#TRAINING##########################################
		
		early_stop = EarlyStopping(monitor='val_loss', 
					   patience=2)
		checkpoint = ModelCheckpoint(filepath='./CNN_BLSTM_Checkpoint.hdf5', 
					     monitor='val_loss', 
					     verbose=1, 
					     save_best_only=True)
		callbacks_list = [checkpoint, early_stop]
		
		history = model.fit([X_train], 
				    y_train, 
				    batch_size=32, 
				    epochs=10, 
				    validation_split=0.2, 
				    callbacks=callbacks_list,
				    verbose=1)
		
		model.save_weights(f'{args.Models}CNN_BLSTM.h5')
		model.save(f'{args.Models}CNN_BLSTM.h5')
		with open(f'{args.Models}CNN_BLSTM.json', 'w') as outfile:
			outfile.write(model.to_json())
		
		score, acc = model.evaluate([np.array(X_test)], 
					    np.array(y_test).reshape(len(y_test), 1), 
					    batch_size=32)
		
		print(f'Trained Model Performance on {len(X_test)} Documents of Hold-out Test Data:\n')
		print(f'Loss: {score} | Accuracy: {acc}')
	
	#####################################################
	#PREDICTION##########################################		
	
	if args.Predict not in ['False','false','f','F']:
	
		loaded_model = load_model(f'{args.Models}{args.loadedModel}')
		
		score, acc = loaded_model.evaluate([np.array(X_test)], 
						   np.array(y_test).reshape(len(y_test), 1), 
						   batch_size=32)
		
		print(f'\nPretrained Model Performance on {len(X_test)} Documents of Hold-out Test Data:\n')
		print(f'Loss: {score} | Accuracy: {acc}\n')
		
		####################################################
		
		print('~'*80)	
		val_input = ''' {"userid":"John Smith","password":"mypassword123","my_location":"123 S. Street","social":"123-45-6789"} '''		
		res =  prediction(loaded_model, 
				  val_input, 
				  to_tokenize, 
				  tokenizer, 
				  payload_len)		
		print(f'Sensitive Document Payload: {val_input}\n')
		print(f'Probability that document is sensitive: {int(res[0][0]*100)}%')		
		####################################################
		
		print('~'*80)		
		val_input = ''' {"user":"Ramona Schwarz","marital":"S","workflow":"sfYeowi09"} '''		
		res = prediction(loaded_model, 
				 val_input, 
				 to_tokenize, 
				 tokenizer, 
				 payload_len)		
		print(f'Sensitive Document Payload: {val_input}\n')
		print(f'Probability that document is sensitive: {int(res[0][0]*100)}%')
		####################################################
		
		print('~'*80)		
		val_input = ''' {"resource":"175.0.1.45","location":"NYC","workflow":"sfYeowi09"} '''		
		res = prediction(loaded_model, 
				 val_input, 
				 to_tokenize, 
				 tokenizer, 
				 payload_len)		
		print(f'Sensitive Document Payload: {val_input}\n')
		print(f'Probability that document is sensitive: {int(res[0][0]*100)}%')
		####################################################
		
		print('~'*80)		
		val_input = ''' {"resource":"gHOSkiQF","location":"(40.71, 71.00)","workflow":"sfYeowi09"} '''		
		res = prediction(loaded_model, 
				 val_input, 
				 to_tokenize, 
				 tokenizer, 
				 payload_len)		
		print(f'Sensitive Document Payload: {val_input}\n')
		print(f'Probability that document is sensitive: {int(res[0][0]*100)}%')
		####################################################
		
		print('~'*80)		
		val_input = ''' {"resource":"gHOSkiQF","location":"tvHqbbjY","birthdate":"01-01-1970"} '''		
		res = prediction(loaded_model, 
				 val_input, 
				 to_tokenize, 
				 tokenizer, 
				 payload_len)		
		print(f'Sensitive Document Payload: {val_input}\n')
		print(f'Probability that document is sensitive: {int(res[0][0]*100)}%')
		####################################################
		
		print('~'*80)		
		val_input = ''' {"gender":"F","location":"Los Angeles","workflow":"sfYeowi09"} '''		
		res = prediction(loaded_model, 
				 val_input, 
				 to_tokenize, 
				 tokenizer, 
				 payload_len)		
		print(f'Sensitive Document Payload: {val_input}\n')
		print(f'Probability that document is sensitive: {int(res[0][0]*100)}%')
		####################################################
		
		print('~'*80)		
		val_input = ''' {"AGI":"250000","location":"tvHqbbjY","workflow":"sfYeowi09"} '''		
		res = prediction(loaded_model, 
				 val_input, 
				 to_tokenize, 
				 tokenizer, 
				 payload_len)		
		print(f'Sensitive Document Payload: {val_input}\n')
		print(f'Probability that document is sensitive: {int(res[0][0]*100)}%')
		####################################################
		
		print('~'*80)		
		val_input = ''' {"CC#":"5234 7466 8967 3458","location":"tvHqbbjY","workflow":"sfYeowi09"} '''		
		res = prediction(loaded_model, 
				 val_input, 
				 to_tokenize, 
				 tokenizer, 
				 payload_len)		
		print(f'Sensitive Document Payload: {val_input}\n')
		print(f'Probability that document is sensitive: {int(res[0][0]*100)}%')
		####################################################
		
		print('~'*80)		
		val_input = ''' {"AGI":"XXXXXX","location":"XX","workflow":"XXXXX"} '''		
		res = prediction(loaded_model, 
				 val_input, 
				 to_tokenize, 
				 tokenizer, 
				 payload_len)		
		print(f'Non-Sensitive Document Payload: {val_input}\n')
		print(f'Probability that document is sensitive: {int(res[0][0]*100)}%')
		####################################################
		
		print('~'*80)		
		val_input = ''' {"resource":"localhost","location":"unknown","workflow":"unauthorized"} '''		
		res = prediction(loaded_model, 
				 val_input, 
				 to_tokenize,
				 tokenizer, 
				 payload_len)		
		print(f'Non-Sensitive Document Payload: {val_input}\n')
		print(f'Probability that document is sensitive: {int(res[0][0]*100)}%')
		####################################################
		
		print('~'*80)		
		val_input = ''' {"resource":"localhost","location":"unknown","workflow":"authorized"} '''		
		res = prediction(loaded_model, 
				 val_input, 
				 to_tokenize, 
				 tokenizer, 
				 payload_len)		
		print(f'Non-Sensitive Document Payload: {val_input}\n')
		print(f'Probability that document is sensitive: {int(res[0][0]*100)}%')
		####################################################
		
		print('~'*80)		
		val_input = ''' {"resource":"gHOSkiQF","location":"unknown","workflow":"sfYeowi09"} '''		
		res = prediction(loaded_model, 
				 val_input, 
				 to_tokenize, 
				 tokenizer, 
				 payload_len)		
		print(f'Non-Sensitive Document Payload: {val_input}\n')
		print(f'Probability that document is sensitive: {int(res[0][0]*100)}%')
		####################################################
		
		print('~'*80)		
		val_input = ''' {"resource":"gHOSkiQF","location":"tvHqbbjY","workflow":"sfYeowi09"} '''		
		res = prediction(loaded_model, 
				 val_input, 
				 to_tokenize,
				 tokenizer, 
				 payload_len)		
		print(f'Non-Sensitive Document Payload: {val_input}\n')
		print(f'Probability that document is sensitive: {int(res[0][0]*100)}%')
		
	print('~'*80)	
	print('END OF PROGRAM\n')
	
#########################################################################################################################
	
if __name__ == '__main__':
		
	print('\n')
	print('_'*80)
	print('The environment and package versions used are: \n')
	
	print(platform.platform())
	print('Python', sys.version)
	print('OS', os.name)
	print('JSON', json.__version__)
	print('RegEx', re.__version__)
	print('NumPy', np.__version__)
	print('Pandas', pd.__version__)
	print('Sklearn', sklearn.__version__)
	print('TensorFlow', tf.__version__)
	print('Keras', keras.__version__)
	
	print('~'*80)
	print('\n')

	main()	
