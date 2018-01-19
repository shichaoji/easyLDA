# coding: utf-8




import sys
try:
    reload(sys)
    sys.setdefaultencoding("utf-8")
except:
    print('python3')
    
    
    
    
import pandas as pd
from time import time, ctime

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
from gensim.models.phrases import Phraser, Phrases

stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation) 
lemmatize = WordNetLemmatizer()

from collections import defaultdict
from stop import stop


import gensim
from gensim import corpora

import pyLDAvis.gensim

from time import time
import logging
import os




class PipelineLDA(object):
    
    """
    input a csv file with one colume (optional column name: 'text')
    every row of that csv should be one document   
    
    """
    
    def __init__(self, path, topics, grams):
        
        global logging
        try:
            mkdir('model')
        except:
            print 'model dir exists'
        self.folder = './model/'
        self.path = path
        self.topics = topics
        self.grams = grams
        self.name = self.path.split('.')[-2].split('/')[-1]+'_topics_'+str(self.topics)+'_{}_gram'.format(self.grams)
        self.df =  pd.read_csv(self.path)
        print self.df.shape
        try:
            self.series = self.df['text']
        except:
            self.series = self.df.iloc[:,0:1]
            
        self.stop = stopwords
        self.stop.update(stop)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                            filename=folder+self.name+'_log.log'.format(self.name),filemode='w')
            
    def addstop(self, wordlist):
        self.stop.update(set(wordlist))


        
        
    def clean(self, article):
        
        try:
            article = str(article).decode('unicode_escape').encode('utf-8')
        except:
            article = str(article)

        zero = "".join(i for i in article if i not in punctuation)

        one = " ".join([i for i in zero.lower().split() if i not in stopwords])

        try:
            three = " ".join(lemmatize.lemmatize(i) for i in one.split())
        except:
            three = " ".join(lemmatize.lemmatize(i.decode('unicode_escape').encode('utf-8')) for i in one.split())
        return three
    
    def split(self):
        start = time()
        n_gram = self.grams

        ap_text = self.series.apply(self.clean)
        ap_text_list = [i.split() for i in ap_text]
        print (len(ap_text_list))

        print ('used: {:.2f}s'.format(time()-start))
        if n_gram==1:
            self.prepared=ap_text_list
            
        elif n_gram==2:
            phs = Phrases(ap_text_list)
            bi_gram = Phraser(phs)
            new_bi_list = [bi_gram[i] for i in ap_text_list]
            self.prepared = new_bi_list

        else:
            phs = Phrases(ap_text_list)
            bi_gram = Phraser(phs)
            new_bi_list = [bi_gram[i] for i in ap_text_list]
                        
            
            phs3=Phrases(new_bi_list)
            tri_gram=Phraser(phs3)
            new_tri_list2 = [tri_gram[i] for i in new_bi_list]

            self.prepared=new_tri_list2


            
    def create_dictionary(self):
        # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
        start = time()
        self.dictionary = corpora.Dictionary(self.prepared)
        self.dictionary.save(folder+self.name+'_dict.dict')
        print (len(self.dictionary))
        print ('used: {:.2f}s'.format(time()-start))

        
    def create_corpus(self):
        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
        start = time()
        self.doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in self.prepared]
        corpora.MmCorpus.serialize(folder+self.name+'_corpus.mm', self.doc_term_matrix)
        print (len(self.doc_term_matrix))
        #print (doc_term_matrix[100])        
        print ('used: {:.2f}s'.format(time()-start))
        
    def train(self, passes=1):
        num_topics = self.topics
        
        start = time()
        # Creating the object for LDA model using gensim library
        Lda = gensim.models.ldamodel.LdaModel

        # Running and Trainign LDA model on the document term matrix.
        self.ldamodel = Lda(self.doc_term_matrix, num_topics=num_topics, id2word = self.dictionary, 
                        passes=passes
                      )
        
        print 'used: {:.2f}s'.format(time()-start)        
        
    def save(self):
        start = time()
        self.ldamodel.save(folder+self.name+'_lda.model')
        print 'used: {:.2f}s'.format(time()-start)
        
    def load(self, path):
        start = time()
        loading = gensim.models.ldamodel.load(path)
        self.ldamodel=loading
        print 'used: {:.2f}s'.format(time()-start)
        
    def visualize(self):
        import pyLDAvis
        try:
            pyLDAvis.enable_notebook()
        except:
            print 'not in jupyter notebook'
            
        start = time()

        self.viz = pyLDAvis.gensim.prepare(self.ldamodel, self.doc_term_matrix, self.dictionary)

        print ('used: {:.2f}s'.format(time()-start))
        print 'saving viz to '+self.name+'_viz.html'
        
        pyLDAvis.save_html(self.viz, self.name+'_viz.html')
        
        return self.viz

        
        
        

        
        
    def __repr__(self):
        return "name: "+ str(self.name)+ " doc numbers: "+ str(self.df.shape[0])
        

def main():
    path = raw_input('the PATH of the csv file with just one column and each row should be one document: ')
    topics = int(str(raw_input('how many topics would you like to fit in your model?: ')).strip())
    n_gram = int(str(raw_input('word topics or phase topics 1-word, 2-phase, 3-long phase? (1/2/3): ')).strip())
    starter = time()
    
    print '1/7: load file'
    lda = PipelineLDA(path, topics, n_gram)
    print lda
    print '2/7: preprocessing docs'
    lda.split()
    print '3/7: create doc dictionary'
    lda.create_dictionary()
    print '4/7: create doc corpus'
    lda.create_corpus()
    print '5/7: train LDA model'
    lda.train()
    print '6/7: save trained LDA model'
    lda.save()
    print '7/7: visualize LDA model result'
    print 'making Viz'
    lda.visualize()
    print 'done'
    ender = time()-starter
    print ('total used: {:.2f}s, {:.2f}mins'.format(ender, ender/60))
