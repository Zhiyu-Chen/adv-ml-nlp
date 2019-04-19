## Reference

```
@INPROCEEDINGS{JiDeepWordBug18, 
author={J. Gao and J. Lanchantin and M. L. Soffa and Y. Qi}, 
booktitle={2018 IEEE Security and Privacy Workshops (SPW)}, 
title={Black-Box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers}, 
year={2018}, 
pages={50-56}, 
keywords={learning (artificial intelligence);pattern classification;program debugging;text analysis;deep learning classifiers;character-level transformations;IMDB movie reviews;Enron spam emails;real-world text datasets;scoring strategies;text input;text perturbations;DeepWordBug;black-box attack;adversarial text sequences;black-box generation;Perturbation methods;Machine learning;Task analysis;Recurrent neural networks;Prediction algorithms;Sentiment analysis;adversarial samples;black box attack;text classification;misclassification;word embedding;deep learning}, 
doi={10.1109/SPW.2018.00016}, 
ISSN={}, 
month={May},}
```


## Usage:

```bash
python train.py --data 0 --model simplernn
``` 


```bash
python attack.py --data [0-7] --model [modelname] --modelpath [modelpath] --power [power] --scoring [algorithm] 
--transformer [algorithm] --maxbatches [batches=20] --batchsize [batchsize=128] ### Generate DeepWordBug adversarial samples
#--modelpath [modelpath] #Model path, stored by train.py
#--scoring [combined, temporal, tail, replaceone, random, grad] # Scoring algorithm
#--transformer [swap, flip, insert, remove] # transformer algorithm
#--power [power] # Attack power(integer, in (0,30]) which is number of modified tokens, i.e., the edit distance
#--maxbatches [batches=20] # Number of batches of adversarial samples generated, samples are selected randomly. 
# Since some test dataset is very large, to evaluate the performance we add this parameter
# to generate on parts of data. By default it will generate 2560 samples.
```

```bash
python attack.py --data 0 --model simplernn --modelpath ./models/simplernn_0_bestmodel.dat --power 30 --scoring combined --transformer insert 

python attack.py --data 0 --model simplernn --modelpath ./models/simplernn_0_bestmodel.dat --power 30 --scoring combined --transformer swap 

python attack.py --data 0 --model simplernn --modelpath ./models/simplernn_0_bestmodel.dat --power 30 --scoring combined --transformer flip 

python attack.py --data 0 --model simplernn --modelpath ./models/simplernn_0_bestmodel.dat --power 30 --scoring combined --transformer remove 
```
