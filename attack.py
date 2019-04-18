import os
# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.autograd import Variable
import argparse
import loaddata
import dataloader
import model
import scoring
import transformer
import numpy as np
import pickle
from metadata import *
import fastText
np.random.seed(7)

parser = argparse.ArgumentParser(description='Data')
parser.add_argument('--data', type=int, default=0, metavar='N',
                    help='data: can be 0,1,2,3,5,6,7 which specify a textdata file')
parser.add_argument('--externaldata', type=str, default='', metavar='S',
                    help='External database file. Default: Empty string')
parser.add_argument('--model', type=str, default='simplernn', metavar='S',
                    help='model type(simplernn, charcnn, bilstm). LSTM as default.')
parser.add_argument('--modelpath', type=str, default='models/simplernn_0_bestmodel.dat', metavar='S',
                    help='model file path')
parser.add_argument('--power', type=int, default=5, metavar='N',
                    help='Attack power')
parser.add_argument('--batchsize', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--scoring', type=str, default='replaceone', metavar='N',
                    help='Scoring function.')
parser.add_argument('--transformer', type=str, default='homoglyph', metavar='N',
                    help='Transformer function.')
parser.add_argument('--maxbatches', type=int, default=20, metavar='B',
                    help='maximum batches of adv samples generated')
parser.add_argument('--advsamplepath', type=str, default=None, metavar='B',
                    help='advsamplepath: If default, will generate one according to parameters')
parser.add_argument('--dictionarysize', type=int, default=20000, metavar='B',
                    help='Size of the dictionary used in RNN model')
parser.add_argument('--charlength', type=int, default=1014, metavar='N',
                    help='length: default 1014')
parser.add_argument('--wordlength', type=int, default=200, metavar='N',
                    help='word length: default 500')


args = parser.parse_args()

torch.manual_seed(8)
torch.cuda.manual_seed(8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


args.datatype = "word"

if args.externaldata!='':
        (data,word_index,numclass) = pickle.load(open(args.externaldata,'rb'))
        testword = dataloader.Worddata(data, getidx = True)
        test_loader = DataLoader(testword,batch_size=args.batchsize, num_workers=4,shuffle=False)  
else:
        ft_model = fastText.load_model(fasttest_crawl_bin)
        (train,test,tokenizer,numclass, rawtrain, rawtest) = loaddata.loaddatawithtokenize(args.data,padding=True, nb_words = args.dictionarysize, datalen = args.wordlength, withraw=True)
        word_index = tokenizer.word_index
        trainword = dataloader.Worddata(train,ft_model, getidx = True, rawdata = rawtrain)
        testword = dataloader.Worddata(test,ft_model, getidx = True, rawdata = rawtest)
        train_loader = DataLoader(trainword,batch_size=args.batchsize, num_workers=0, shuffle = True)
        test_loader = DataLoader(testword,batch_size=args.batchsize, num_workers=0,shuffle=True)
        maxlength =  args.wordlength

if args.model == "simplernn":
    model = model.smallRNN(classes = numclass)
elif args.model == "bilstm":
    model = model.smallRNN(classes = numclass, bidirection = True)

print(model)

state = torch.load(args.modelpath)
model = model.to(device)
try:
    model.load_state_dict(state['state_dict'])
except:
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(state['state_dict'])
    model = model.module

alltimebest = 0
bestfeature = []
def recoveradv(rawsequence, index2word, inputs, advwords):
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n '
    rear_ct = len(rawsequence)
    advsequence = rawsequence[:]
    try:
        for i in range(inputs.size()[0]-1,-1,-1):
            wordi = index2word[inputs[i].item()]
            rear_ct = rawsequence[:rear_ct].rfind(wordi)
                # print(rear_ct)
            if inputs[i].item()>=3:
                advsequence = advsequence[:rear_ct] + advwords[i] + advsequence[rear_ct + len(wordi):]
    except:
        print('something went wrong')
    return advsequence
    


def attackword(maxbatch = None):
    corrects = .0
    total_loss = 0
    model.eval()
    wordinput = []
    tgt = []
    adv = []
    origsample = []
    origsampleidx = []
    flipped_idx = []
    flipped_origin = []
    flipped_adv = []

    for dataid, data in enumerate(test_loader):
        print(dataid)
        if maxbatch!=None and dataid >= maxbatch:
            break
        inputs,target, idx, raw = data
        inputs, target = inputs.to(device), target.to(device)
        origsample.append(inputs)
        origsampleidx.append(idx)
        tgt.append(target)
        wtmp = []
        output = model(inputs)
        pred = torch.max(output, 1)[1].view(target.size())

        losses = scoring.scorefunc(args.scoring)(model, inputs, pred, numclass)

        sorted, indices = torch.sort(losses,dim = 1,descending=True)

        advinputs = inputs.clone()

        for k in range(inputs.size()[0]):
            wtmp.append([])
            for i in range(inputs.size()[1]):
                if test.content[idx[k],i] != '[PADDING]':
                    wtmp[-1].append(test.content[idx[k],i])
                else:
                    wtmp[-1].append('')

        for k in range(inputs.size()[0]):
            j = 0
            t = 0
            while j < args.power and t<inputs.size()[1]:
                if test.content[idx[k],indices[k][t]] != '[PADDING]':
                    origin_w = test.content[idx[k],indices[k][t]]
                    #word = homoglyph(origin_w)
                    word = transformer.transform(args.transformer)(origin_w)
                    advinputs[k,indices[k][t]] = torch.from_numpy(ft_model.get_word_vector(word)).float()
                    wtmp[k][indices[k][t]] = word
                    #print(word)
                    j+=1
                t+=1

        adv.append(advinputs)

        output2 = model(advinputs)
        pred2 = torch.max(output2, 1)[1].view(target.size())
        corrects += (pred2 == target).sum().item()
        for i in range(len(wtmp)):
            #print(raw[i])
            #print(pred[i].item())
            #wordinputi = recoveradv(raw[i],index2word,inputs[i], wtmp[i])
            wordinputi = ' '.join(wtmp[i])
            #print(wordinputi)
            wordinput.append(wordinputi)
            #print(pred2[i].item())
            if pred[i].equal(target[i]):
                if not pred[i].equal(pred2[i]):
                    flipped_idx.append(idx[i])
                    flipped_origin.append(target[i].item())
                    flipped_adv.append(pred2[i].item())
           

    target = torch.cat(tgt)
    advinputs = torch.cat(adv)
    origsamples = torch.cat(origsample)
    origsampleidx = torch.cat(origsampleidx)
    acc = corrects/advinputs.size(0)
    print('Accuracy %.5f' % (acc))
    f = open('attack_log.txt','a')
    f.write('%d\t%d\t%s\t%s\t%s\t%d\t%.2f\n' % (args.data,args.wordlength,args.model,args.scoring,args.transformer,args.power,100*acc))
    if args.advsamplepath == None:
        advsamplepath = 'advsamples/%s_%d_%s_%s_%d_%d.dat' % (args.model,args.data,args.scoring,args.transformer,args.power,args.wordlength)
    else:
        advsamplepath = args.advsamplepath
    torch.save({'original':origsamples,'sampleid':origsampleidx,'wordinput':wordinput,'advinputs':advinputs,'labels':target,'flipped_adv':flipped_adv,'flipped_origin':flipped_origin,'flipped_idx':flipped_idx}, advsamplepath)

    


        

index2word = {}
index2word[0] = '[PADDING]'
index2word[1] = '[START]'
index2word[2] = '[UNKNOWN]'
index2word[3] = ''
if args.dictionarysize==20000:
    for i in word_index:
        if word_index[i]+3 < args.dictionarysize:
            index2word[word_index[i]+3]=i
else:
    for i in word_index:
        if word_index[i] + 3 < args.dictionarysize:
            index2word[word_index[i]+3]=i  
attackword(maxbatch = args.maxbatches)

