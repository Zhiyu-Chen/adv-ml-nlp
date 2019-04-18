import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import loaddata
import dataloader
import shutil
import model
from metadata import *
import fastText
import numpy as np




def save_checkpoint(state, is_best, filename='checkpoint.dat'):
    torch.save(state, filename + '_checkpoint.dat')
    if is_best:
        shutil.copyfile(filename + '_checkpoint.dat', filename + "_bestmodel.dat")


parser = argparse.ArgumentParser(description='Data')
parser.add_argument('--data', type=int, default=0, metavar='N',
                    help='data 0 - 6')
parser.add_argument('--charlength', type=int, default=1014, metavar='N',
                    help='length: default 1014')
parser.add_argument('--wordlength', type=int, default=200, metavar='N',
                    help='length: default 500')
parser.add_argument('--model', type=str, default='simplernn', metavar='N',
                    help='model type: LSTM as default')
parser.add_argument('--space', type=bool, default=False, metavar='B',
                    help='Whether including space in the alphabet')
parser.add_argument('--trans', type=bool, default=False, metavar='B',
                    help='Not implemented yet, add thesausus transformation')
parser.add_argument('--backward', type=int, default=-1, metavar='B',
                    help='Backward direction')
parser.add_argument('--epochs', type=int, default=10, metavar='B',
                    help='Number of epochs')
parser.add_argument('--batchsize', type=int, default=128, metavar='B',
                    help='batch size')
parser.add_argument('--dictionarysize', type=int, default=20000, metavar='B',
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005, metavar='B',
                    help='learning rate')
parser.add_argument('--maxnorm', type=float, default=400, metavar='B',
                    help='learning rate')
args = parser.parse_args()

torch.manual_seed(7)
torch.cuda.manual_seed_all(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


args.datatype = "word"

print("Loading data..")
ft_model = fastText.load_model(fasttest_crawl_bin)
(train,test,tokenizer,numclass,rawtrain,rawtest) = loaddata.loaddatawithtokenize(args.data,padding=True,withraw = True,nb_words = args.dictionarysize, datalen = args.wordlength)
trainword = dataloader.Worddata(train,ft_model,backward = args.backward)
testword = dataloader.Worddata(test,ft_model,backward = args.backward)


train_loader = DataLoader(trainword,batch_size=args.batchsize, num_workers=4, shuffle = True)
test_loader = DataLoader(testword,batch_size=args.batchsize, num_workers=4)



if args.model == "simplernn":
    model = model.smallRNN(classes = numclass)
elif args.model == "bilstm":
    model = model.smallRNN(classes = numclass, bidirection = True)
elif args.model == "wordcnn":
    model = model.WordCNN(classes = numclass)



model = model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

bestacc = 0
for epoch in range(args.epochs+1):
    print('Start epoch %d' % epoch)
    model.train()
    for dataid, data in enumerate(train_loader):
        inputs,target = data
        inputs,target = Variable(inputs),  Variable(target)
        inputs, target = inputs.to(device), target.to(device)
        output = model(inputs)
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    correct = .0
    total_loss = 0
    model.eval()
    for dataid, data in enumerate(test_loader):
        inputs,target = data
        inputs, target = inputs.to(device), target.to(device)
        output = model(inputs)
        loss = F.nll_loss(output, target)
        total_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    acc = correct/len(test_loader.dataset)
    avg_loss = total_loss/len(test_loader.dataset)
    print('Epoch %d : Loss %.4f Accuracy %.5f' % (epoch,avg_loss,acc))
    is_best = acc > bestacc
    if is_best:
        bestacc = acc
    if args.dictionarysize!=20000:
        fname = "models/" + args.model +str(args.dictionarysize) + "_" + str(args.data)
    else:
        fname = "models/" + args.model + "_" + str(args.data)
        
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'bestacc': bestacc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename = fname)
    
