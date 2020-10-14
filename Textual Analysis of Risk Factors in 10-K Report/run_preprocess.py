from lda_model import *
from time import sleep
import json
from IPython.core.display import HTML

if __name__ == '__main__':

    model = lda_model()

    model.train_bigram_model(0)
    
    model.train_trigram_model(0)

    '''
    for (start,end) in [(18252,20000),(20000,30000),(30000,None)]:
        print(start,end)
        model.trigram_transform(1,start,end)
        sleep(300)
    '''
    
    print('create_dictionary')
    dictionary = model.create_dictionary(1)
    print('create_corpus')
    corpus = model.create_corpus(1)
    print('train_lda')
    lda = model.train_lda(25,1)

    
    LDAvis_data_filepath = join(model.output_path,'ldavis_prepared_10')

    if os.path.isfile(LDAvis_data_filepath): 
    # load the pre-prepared pyLDAvis data from disk
        with open(LDAvis_data_filepath, 'r') as json_file:
            dict_data = json.load(json_file)
            LDAvis_prepared = prepared_data_from_dict(dict_data)
        pyLDAvis.show(LDAvis_prepared)
    else:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda, corpus,
                                              dictionary,sort_topics=False)

        pyLDAvis.save_json(LDAvis_prepared, LDAvis_data_filepath)
        