# Importing built-in libraries (no need to install these)
import re
import os
from os import listdir
from os.path import isfile, join

# Importing libraries you need to install
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools as it

import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en.stop_words import STOP_WORDS

from gensim.corpora import Dictionary, MmCorpus
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
from gensim.parsing.preprocessing import remove_stopwords

from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from wordcloud import WordCloud,STOPWORDS

import matplotlib.pyplot as plt 
import pyLDAvis
import pyLDAvis.gensim
import warnings
from pyLDAvis import PreparedData

def punct_space(token):
	"""
	helper function to eliminate tokens
	that are pure punctuation or whitespace
	"""
	
	return token.is_punct or token.is_space

def line_review(filename):
	"""
	generator function to read in reviews from the file
	and un-escape the original line breaks in the text
	"""
	
	with open(filename, encoding='utf_8') as f:
		for text in f:
			#text_re_stop = remove_stopwords(text)
			yield text.replace('\\n', '\n')
			
def lemmatized_sentence_corpus(filename,nlp):
	"""
	generator function to use spaCy to parse reviews,
	lemmatize the text, and yield sentences
	"""
	
	for parsed_review in nlp.pipe(line_review(filename),batch_size=100, n_process=4):
		for sent in parsed_review.sents:
			print("**************************")
			yield u' '.join([token.lemma_ for token in sent if not punct_space(token)])
	print("##################################")


def clean_raw_data(raw_file,output_path,run_or_load_flag):
	"""
	Clean raw file with punctuation or whitespace removal 
	Lemmatize the cleaned token and save to output_path

	"""
	if run_or_load_flag:
		nlp = spacy.load("en_core_web_sm")
		with open(output_path, 'w', encoding='utf_8') as f:	
			count =0
			for sentence in lemmatized_sentence_corpus(raw_file,nlp):
				f.write(sentence + '\n')
				count += 1
				print(count)
	else:
		pass

def train_phrase_model(cleaned_filepath,bigram_model_filepath,run_or_load_flag):
	
	if run_or_load_flag:
		unigram_sentences = LineSentence(cleaned_filepath)
		bigram_model = Phrases(unigram_sentences)
		bigram_model.save(bigram_model_filepath)
	else:
		bigram_model = Phrases.load(bigram_model_filepath)
	
	return bigram_model


def bigram_transform(input_file,output_path,bigram_model,run_or_load_flag):

	if run_or_load_flag:
		nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser","ner"])
		with open(output_path, 'w', encoding='utf_8') as f:
			count = 0
			for parsed_review in nlp.pipe(line_review(input_file),
										  batch_size=100, n_process=4):
				# lemmatize the text, removing punctuation and whitespace
				unigram_text = [token.lemma_  for token in parsed_review if token.lemma_ != '-PRON-']
				bigram_text = bigram_model[unigram_text]
				#cleaned_text = [term for term in bigram_text if term not in spacy.en_core_web_sm.STOPWORDS]
				cleaned_text = remove_stopwords(u' '.join(bigram_text))
				f.write(cleaned_text + '\n')
				count += 1
				print(count, end= ',')
	else:
		pass

def create_dictionary(input_file,dictionary_filepath,run_or_load_flag):
	if run_or_load_flag:
		text = LineSentence(input_file)

		# learn the dictionary by iterating over all of the reviews
		dictionary = Dictionary(text)
		
		# filter tokens that are very rare or too common from
		# the dictionary (filter_extremes) and reassign integer ids (compactify)
		dictionary.filter_extremes(no_below=10, no_above=0.4)
		dictionary.compactify()

		dictionary.save(dictionary_filepath)

	else:
		dictionary = Dictionary.load(dictionary_filepath)

	return dictionary


def trigram_bow_generator(filepath,dictionary):
	"""
	generator function to read reviews from a file
	and yield a bag-of-words representation
	"""
	
	for text in LineSentence(filepath):
		yield dictionary.doc2bow(text)

def create_corpus(input_file,corpus_filepath,dictionary,run_or_load_flag):
	# generate bag-of-words representations for
	# all reviews and save them as a matrix
	if run_or_load_flag:
		MmCorpus.serialize(corpus_filepath,
						   trigram_bow_generator(input_file,dictionary))
		corpus = MmCorpus(corpus_filepath)
	else:
		corpus = MmCorpus(corpus_filepath)

	return corpus


def train_lda(corpus,dictionary,lda_model_filepath,num_topics,run_or_load_flag):
	if run_or_load_flag:
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			
			# workers => sets the parallelism, and should be
			# set to your number of physical cores minus one
			lda = LdaMulticore(corpus,
							   num_topics=num_topics,
							   id2word=dictionary,
							   workers=3)
		lda.save(lda_model_filepath)
	else:
		lda = LdaMulticore.load(lda_model_filepath)

	return lda

def explore_topic(topic_number, topn=20):
	"""
	accept a user-supplied topic number and
	print out a formatted list of the top terms
	"""
		
	print (u'{:20} {}'.format(u'term', u'frequency') + u'\n')

	for term, frequency in lda.show_topic(topic_number, topn=20):
		print (u'{:20} {:.3f}'.format(term, round(frequency, 3)))
		
def topic_visualizer(lda,topic_number, topn=20):
	"""
	print out a wordcloud figure of the top terms 
	for the picked toptic
	"""
	
	stop_words = set(STOPWORDS) 
	topic = lda.show_topic(topic_number,topn)
	dict_topic = dict(topic)

	cloud = WordCloud(stopwords=stop_words,
				  background_color='white',
				  width=2500,
				  height=1800,
				  max_words=topn,
				  prefer_horizontal=1.0)
	
	cloud.generate_from_frequencies(dict_topic, max_font_size=300)
	
	plt.figure(figsize = (8, 8), facecolor = None) 
	plt.imshow(cloud) 
	plt.axis("off") 
	plt.tight_layout(pad = 0) 

	plt.show() 


class main():

	def __init__(self):

		self.data_path = r'/Users/henry/Documents/study/Springboard/nlp_project/data'
		self.model_path = r'/Users/henry/Documents/study/Springboard/nlp_project/models/all_year'

		self.stopwords = ['include','use','risk','factor','subject','relate','result','associate','s','significant',
             'substantial','successful','additional','report','statement','maintain','provide','evaluate',
             'annual','disclosure','identify','assessment','obtein','tax','income','million','$','December_31',
             'year','total','asset','approximately','taxable_income','effective','●','adverse','certain',
            'stockholder','transaction','right','business','holder','shareholder','officer','unit',
            'fair_value','goodwill','day','intanible_asset','maintain','Directors','Board','revenue','impact',
            'common_stock','stock','investor','value','equity','adversely_effect','per_share','exercise',
            'materially_adverse','materially','impact','adversely']

	def run(self,run_or_load_flag):

		raw_file_path = join(self.data_path,'all_txt_file.txt')
		cleaned_filepath = join(self.data_path,'all_txt_file_cleaned.txt')
		bigram_txt_filepath = join(self.model_path,'all_text_bigram_trans.txt')
		bigram_model_filepath = join(self.model_path,'bigram_model_all')
		dictionary_filepath = join(self.model_path,'dict_all.dict')
		corpus_filepath = join(self.model_path,'corpus_all.mm')
		lda_model_filepath = join(self.model_path, 'lda_model_all')

		print(f'clean_raw_data,run_or_load_flag = {run_or_load_flag}')
		clean_raw_data(raw_file_path,cleaned_filepath,0)
		
		print(f'train_phrase_model,run_or_load_flag = {run_or_load_flag}')
		bigram_model = train_phrase_model(cleaned_filepath,bigram_model_filepath,run_or_load_flag)
		
		print(f'bigram_transform,run_or_load_flag = {run_or_load_flag}')
		bigram_transform(cleaned_filepath,bigram_txt_filepath,bigram_model,run_or_load_flag)
		
		print(f'create_dictionary,run_or_load_flag = {run_or_load_flag}')
		dictionary = create_dictionary(bigram_txt_filepath,dictionary_filepath,run_or_load_flag)

		print(f'create_corpus,run_or_load_flag = {run_or_load_flag}')
		corpus = create_corpus(bigram_txt_filepath,corpus_filepath,dictionary,run_or_load_flag)
		
		print(f'train_lda,run_or_load_flag = {run_or_load_flag}')
		lda =train_lda(corpus,dictionary,lda_model_filepath,25,run_or_load_flag)
		
		print(f'topic_visualizer,run_or_load_flag = {run_or_load_flag}')
		topic_visualizer(lda,topic_number=8)

if __name__ == '__main__':
	

	path = main()

	path.run(1)
	

	

	

	
	