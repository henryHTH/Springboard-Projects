from functions import *


class lda_model():

    def __init__(self):

        self.data_path = r'./data/all_data'
        self.output_path = r'./output/all_data'
        self.model_path = r'./output/all_data/models'
        self.figure_path = r'./output/all_data/figures'
        
        self.raw_file_path = join(self.data_path,'txt_file.txt')
        self.cleaned_filepath = join(self.data_path,'txt_file_cleaned.txt')
        
        self.bigram_txt_filepath = join(self.output_path,'text_bigram_trans.txt')
        self.bigram_model_filepath = join(self.output_path,'bigram_model')
        
        self.trigram_txt_filepath = join(self.output_path,'text_trigram_trans.txt')
        self.trigram_model_filepath = join(self.output_path,'trigram_model')
        
        self.final_cleaned_path = join(self.output_path,'final_cleaned.txt')

        self.dictionary_filepath = join(self.output_path,'dict.dict')
        self.corpus_filepath = join(self.output_path,'corpus.mm')
        
        #self.lda_model_filepath = join(self.output_path, 'lda_model')
        

        self.stopwords_ = set(['include','use','risk','factor','subject','relate','result','associate','s','significant',
             'substantial','successful','additional','report','statement','maintain','provide','evaluate',
             'annual','disclosure','identify','assessment','obtein','tax','income','million','$','December_31',
             'year','total','asset','approximately','taxable_income','effective','adverse','certain',
            'stockholder','transaction','right','business','holder','shareholder','officer','unit',
            'fair_value','goodwill','day','intanible_asset','maintain','Directors','Board','revenue','impact',
            'common_stock','stock','investor','value','equity','adversely_effect','per_share','exercise',
            'materially_adverse','materially','impact','adversely','affect','3','2018','2017','2016','2015','2014',
            '2013','2012','2011','2010','2009','2008','2007','2006','2005','2004','2003','2022','item','1a.','-pron-'])

        #self.topic_number = topic_number

        self.nlp = spacy.load("en_core_web_sm")
        self.stopwords = STOP_WORDS.union(self.stopwords_)
        for word in self.stopwords_:
            lexeme = self.nlp.vocab[word]
            lexeme.is_stop = True

    def clean_raw_data(self,run_or_load_flag):
        
        """
        Clean raw file with punctuation or whitespace removal 
        Lemmatize the cleaned token and save to output_path
        Output cleaned file, one line is one document  
        """

        if run_or_load_flag:
            nlp = spacy.load("en_core_web_sm")
            with open(self.cleaned_filepath, 'w', encoding='utf_8') as f:   
                for sentence in lemmatized_sentence_corpus(self.raw_file_path,nlp):
                    f.write(sentence + '\n')
        else:
            pass

    def train_bigram_model(self,run_or_load_flag):
        
        '''
        Read cleaned file which is the output from function clean_raw_data
        Train a phrase model, which is the bigram model
        return bigram model
        '''

        if run_or_load_flag:
            #  Steam the txt file with  sentences
            unigram_sentences = LineSentence(self.cleaned_filepath)
            #  Train bigram model
            self.bigram_model = Phrases(unigram_sentences)
            #  Save bigram model
            self.bigram_model.save(self.bigram_model_filepath)
        else:
            #  Load bigram model
            self.bigram_model = Phrases.load(self.bigram_model_filepath)
        
        return self.bigram_model
        
    def bigram_transform(self,run_or_load_flag):
        
        '''
        Read the cleaned file, which is the output from function clean_raw_data
        Load the pre-trained bigram model, which is the output from function train_bigram_model
        Implement the bigram model to our txt documents
        Output bigram file, one line is one document
        '''

        if run_or_load_flag:
            nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser","ner"])
            with open(self.bigram_txt_filepath, 'w', encoding='utf_8') as f:
                for parsed_review in nlp.pipe(line_review(self.cleaned_filepath),
                                              batch_size=100, n_process=4):
                    
                    # lemmatize the text, removing '-PRON-'
                    unigram_text = [token.text  for token in parsed_review]
                    #  Implement bigram model to the lemmatized text
                    bigram_text = self.bigram_model[unigram_text]
                    #  Join all the words into one sentence
                    bigram_text = u' '.join(bigram_text)
                    f.write(bigram_text + '\n')
        else:
            pass



    def train_trigram_model(self,run_or_load_flag):
        
        '''
        Read bigram file which is the output from function bigram_transform
        Train a phrase model, which is the trigram model
        return trigram model
        '''

        if run_or_load_flag:
            #  Steam the txt file with  sentences
            bigram_sentences = LineSentence(self.bigram_txt_filepath)
            #  Train trigram model
            self.trigram_model = Phrases(bigram_sentences)
            #  Save trigram model
            self.trigram_model.save(self.trigram_model_filepath)
        else:
            #  Load trigram model
            self.trigram_model = Phrases.load(self.trigram_model_filepath)
        
        return self.trigram_model



    def trigram_transform(self,run_or_load_flag,start=0,end=None):

        '''
        Read the raw file
        Load the pre-trained bigram model, trigram model
        Implement the bigram model and trigram model to our txt documents
        Output trigram file, one line is one document
        '''

        if run_or_load_flag:
            print(start,end)
            with open(self.trigram_txt_filepath, 'a', encoding='utf_8') as f:
                for parsed_text in self.nlp.pipe(it.islice(line_review(self.raw_file_path),start,end),
                                              batch_size=1, n_process=4):
                    # lemmatize the text, removing punctuation and whitespace
                    unigram_text = [token.lemma_.lower() for token in parsed_text
                              if not punct_space(token)]
                    
                    #  Implement bigram model to the lemmatized text
                    bigram_text = self.bigram_model[unigram_text]
                    
                    #  Implement trigram model to the bigram text
                    trigram_text = self.trigram_model[bigram_text]
                    #print([term for term in trigram_text if term in self.stopwords])
                    #  Remove remaining stopwords from trigram text
                    trigram_text = [term for term in trigram_text if (term not in self.stopwords) and (len(term) > 2)]
                    
                    #  Join all the words into one sentence
                    trigram_text = u' '.join(trigram_text)
                    #  Write the sentence into txt file
                    f.write(trigram_text + '\n')            
        else:
            pass


    def create_dictionary(self,run_or_load_flag):
        
        '''
        Create dictionry from trigram file which is the output from function trigram_transform
        '''

        if run_or_load_flag:
            text = LineSentence(self.final_cleaned_path)

            # learn the dictionary by iterating over all of the documents
            dictionary = Dictionary(text)
            
            # filter tokens that are very rare or too common from
            # the dictionary (filter_extremes) and reassign integer ids (compactify)
            dictionary.filter_extremes(no_below=10, no_above=0.6)
            dictionary.compactify()

            self.dictionary = dictionary
            self.dictionary.save(self.dictionary_filepath)

        else:
            self.dictionary = Dictionary.load(self.dictionary_filepath)

        return self.dictionary


    def create_corpus(self,run_or_load_flag):

        # generate bag-of-words representations for
        # all reviews and save them as a matrix

        if run_or_load_flag:
            MmCorpus.serialize(self.corpus_filepath,
                               trigram_bow_generator(self.final_cleaned_path,self.dictionary))
            self.corpus = MmCorpus(self.corpus_filepath)
        else:
            self.corpus = MmCorpus(self.corpus_filepath)

        return self.corpus


    def train_lda(self,topic_number,alpha,run_or_load_flag):
        
        '''
        Train lda model
        '''
        
        lda_model_filepath = join(self.model_path, alpha,'lda_model_' + str(topic_number))
        
        if run_or_load_flag:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                
                # workers => sets the parallelism, and should be
                # set to your number of physical cores minus one
                self.lda = LdaMulticore(self.corpus,
                                   num_topics=topic_number,
                                   id2word=self.dictionary,
                                   alpha = alpha,
                                   workers=3)
            self.lda.save(lda_model_filepath)
        else:
            self.lda = LdaMulticore.load(lda_model_filepath)

        return self.lda


    def cal_coherence(self,lda):

        coherence_model_lda = CoherenceModel(model=lda, corpus=self.corpus, dictionary=self.dictionary, coherence='u_mass')
        coherence_lda = coherence_model_lda.get_coherence()
        return coherence_lda


    def run(self,topic_number,run_or_load_flag):

        
        print('clean_raw_data,run_or_load_flag = {}'.format(run_or_load_flag))
        self.clean_raw_data(run_or_load_flag)
        
        print('train_bigram_model,run_or_load_flag = {}'.format(run_or_load_flag))
        self.bigram_model = self.train_bigram_model(run_or_load_flag)
        
        print('bigram_transform,run_or_load_flag = {}'.format(run_or_load_flag))
        self.bigram_transform(run_or_load_flag)

        print('train_trigram_model,run_or_load_flag = {}'.format(run_or_load_flag))
        self.trigram_model = self.train_trigram_model(run_or_load_flag)
        
        print('trigram_transform,run_or_load_flag = {}'.format(run_or_load_flag))
        self.trigram_transform(run_or_load_flag)

        print('create_dictionary,run_or_load_flag = {}'.format(run_or_load_flag))
        self.dictionary = self.create_dictionary(run_or_load_flag)

        print('create_corpus,run_or_load_flag = {}'.format(run_or_load_flag))
        self.corpus = self.create_corpus(run_or_load_flag)
        
        print('train_lda,run_or_load_flag = {}'.format(run_or_load_flag))
        self.lda = self.train_lda(topic_number,'auto',run_or_load_flag)
        
        print('lda model successfully trained !')
        #print(f'topic_visualizer,run_or_load_flag = {run_or_load_flag}')
        for i in range(0,20):
            topic_visualizer(self.lda,topic_number=i)
        
if __name__ == '__main__':
    
    topic_number = 20
    model = lda_model()
    
    begin_time = datetime.datetime.now()
    print(begin_time)
    model.run(topic_number,0)
    #model.train_bigram_model(0)
    #model.train_trigram_model(0)
    #for interval in [(10000,20000),(20000,30000),(30000,None)]:
    #    model.trigram_transform(1,start=interval[0],end=interval[1])
    #    print(datetime.datetime.now() - begin_time)
    #    time.sleep(120)
    #model.run(1)
    

    

    

    
    