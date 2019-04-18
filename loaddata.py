import sys
import csv
csv.field_size_limit(2147483647)
from preprocessing import Tokenizer, pad_sequences
from metadata import *
import fastText
import numpy as np

# small = True
small = False
if small:
    textdatafolder = 'textdata_small/'
else:
    textdatafolder = 'textdata/'
    

            
class dataset:
    def __init__(self,filename,line=3):
        self.output = []
        self.content = []
        self.columns=line
        self.loadcsv(filename)
    def loadcsv(self, filename):
        reader = csv.reader(open(filename, "rt", encoding = "utf8"))
        count = 0
        for row in reader:
            if not row:
                continue
            if self.columns==2:
                self.output.append(int(row[0])-1)
                self.content.append((row[1]).lower())
            elif self.columns==3:
                self.output.append(int(row[0])-1)
                self.content.append((row[1] + " " + row[2]).lower())           
            elif self.columns==4:
                self.output.append(int(row[0])-1)
                self.content.append((row[1] + " " + row[2] + " " + row[3]).lower())       


def loaddata(i = 0):
    datanames = ['ag_news','amazon_review_full','amazon_review_polarity','dbpedia','sogou_news','yahoo_answers','yelp_review_full','yelp_review_polarity','enron','blog-authorship-corpus']
    lines = [3,3,3,3,3,4,2,2,2,2]
    classes= [4,5,2,14,5,10,5,2,2,3]
    # if not 'blog' in datanames[i]:
    trainadd = textdatafolder+datanames[i]+'_csv/train.csv'
    testadd = textdatafolder+datanames[i]+'_csv/test.csv'
    traindata = dataset(trainadd,lines[i])
    testdata = dataset(testadd,lines[i])
    return (traindata,testdata,classes[i])
    # else:
    #     data = dataset()
    #     return (traindata,testdata,classes[i])

def load_embeddings(tokenizer,dict_size = 20000,dim=300):
    ft_model = fastText.load_model(fasttest_crawl_bin)
    word_embeddings = []
    for i in range(dict_size):
        if i <3:
            #add random
            continue
        word = tokenizer.index_word[i-3]
        embedding = ft_model.get_word_vector(word)
        word_embeddings.append(embedding)


def loaddatawithtokenize(i = 0, padding = False,nb_words = 20000, start_char = 1, oov_char=2, index_from=3, withraw = False, datalen = 300):
    (traindata,testdata,numclass) = loaddata(i)
    rawtrain = traindata.content[:]
    rawtest = testdata.content[:]
    tokenizer = Tokenizer(lower=True)
    tokenizer.fit_on_texts(traindata.content + testdata.content)
    traindata.content = tokenizer.texts_to_sequences(traindata.content)
    testdata.content  = tokenizer.texts_to_sequences(testdata.content)
    if padding:
        traindata.content = pad_sequences(traindata.content, maxlen=datalen)
        testdata.content = pad_sequences(testdata.content, maxlen=datalen)
    '''
    ft_model = fastText.load_model(fasttest_crawl_bin)
    all_embeddings = []
    for i in range(len(traindata.content)):
        tokens = traindata.content[i]
        embeddings = []
        for each_t in tokens:
            if each_t  == '[PADDING]':
                embeddings.append(np.array([0.1]*300))
            else:
                embeddings.append(ft_model.get_word_vector(each_t))
        all_embeddings.append(np.array(embeddings))
    traindata.content = np.array(all_embeddings)
    all_embeddings = []
    for i in range(len(testdata.content)):
        tokens = testdata.content[i]
        embeddings = []
        for each_t in tokens:
            if each_t  == '[PADDING]':
                embeddings.append(np.array([0.1]*300))
            else:
                embeddings.append(ft_model.get_word_vector(each_t))
        all_embeddings.append(np.array(embeddings))
    testdata.content = np.array(all_embeddings)
    '''
    if withraw:
        return traindata,testdata,tokenizer,numclass,rawtrain,rawtest
    else:
        return traindata,testdata,tokenizer,numclass


if __name__ == "__main__":
    (traindata,testdata,numclass) = loaddata(9)
    print(len(traindata.output))
    # blogdata('textdata/blog-authorship-corpus_csv/blogtext.csv',2)