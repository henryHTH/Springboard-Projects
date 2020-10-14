from functions import *
class lda_model():

    def __init__(self):

        self.data_path = r'./data/all_data'
        self.output_path = r'./output/all_data'
        self.model_path = r'./output/all_data/models'

        self.raw_file_path = join(self.data_path,'txt_file.txt')
        self.cleaned_filepath = join(self.data_path,'txt_file_cleaned.txt')
        
        self.bigram_txt_filepath = join(self.output_path,'text_bigram_trans.txt')
        self.bigram_model_filepath = join(self.output_path,'bigram_model')
        
        self.trigram_txt_filepath = join(self.output_path,'text_trigram_trans.txt')
        self.trigram_model_filepath = join(self.output_path,'trigram_model')
        
        self.dictionary_filepath = join(self.output_path,'dict.dict')
        self.corpus_filepath = join(self.output_path,'corpus.mm')
        
        #self.lda_model_filepath = join(self.output_path, 'lda_model')
        

        self.stopwords_ = set(['include','use','risk','factor','subject','relate','result','associate','significant',
             'substantial','successful','additional','report','statement','maintain','provide','evaluate',
             'annual','disclosure','identify','assessment','obtein','tax','income','million','$','december_31',
             'year','total','asset','approximately','taxable_income','effective','adverse','certain',
            'stockholder','transaction','right','business','holder','shareholder','officer','unit',
            'fair_value','goodwill','day','intanible_asset','maintain','directors','board','revenue','impact',
            'common_stock','stock','investor','value','equity','adversely_effect','per_share','exercise',
            'materially_adverse','materially','impact','adversely','affect','3','2018','2017','2016','2015','2014',
            '2013','2012','2011','2010','2009','2008','2007','2006','2005','2004','2003','2022','item','1a.','-pron-',
            '-PRON-','item_1a.','january','february','march','april','may','june','july','august','september','october',
            'november','december','inc.'])

        #self.topic_number = topic_number

        self.nlp = spacy.load("en_core_web_sm")
        self.stopwords = STOP_WORDS.union(self.stopwords_)
        for word in self.stopwords_:
            lexeme = self.nlp.vocab[word]
            lexeme.is_stop = True


if __name__ == '__main__':

    model = lda_model()
    with open(join(model.output_path,'final_cleaned.txt'),'w',encoding='utf_8') as w:
        with open(model.trigram_txt_filepath, 'r', encoding='utf_8') as f:
            count = 0
            for line in f.readlines():
                text = [term.lower() for term in line.split() if (term not in model.stopwords) and (len(term) > 2)]
                if len(text) > 100:
                    text = u' '.join(text)  
                    count += 1
                    if count % 1000 == 0:
                        print(count)
                    w.write(text + '\n')
            f.close()
        w.close()