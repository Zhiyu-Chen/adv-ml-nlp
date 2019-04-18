import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

        
class Worddata(Dataset):
    def __init__(self, data, ft_model = None,tokenizer = True, length=1014, space = False, backward = -1, getidx = False, rawdata = None):
        self.backward = backward
        self.length = length
        (self.inputs,self.labels) = (data.content,data.output)
        self.labels = torch.LongTensor(self.labels)
        #self.inputs = torch.from_numpy(self.inputs).long()
        self.getidx = getidx
        self.ft_model = ft_model
        if rawdata:
            self.raw = rawdata
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self,idx):
        x = self.inputs[idx]
        y = self.labels[idx]
        embeddings = []
        for each_t in x:
            if each_t  == '[PADDING]':
                embeddings.append(np.array([0.1]*300))
            else:
                embeddings.append(self.ft_model.get_word_vector(each_t))
        x = np.array(embeddings)
        x = torch.from_numpy(x).float()
        if self.getidx==True:
            if self.raw:
                return x,y,idx,self.raw[idx]
            else:
                return x,y,idx
        else:
            return x,y


if __name__ == '__main__':
    # Example for generating external dataset
    import pickle
    import loaddata
    (train,test,tokenizer,numclass) = loaddata.loaddatawithtokenize(0)
    test.content = test.content[:100]
    test.output = test.output[:100]
    word_index = tokenizer.word_index
    
    pickle.dump((test,word_index,numclass), open('textdata/ag_news_small_word.pickle','wb'))