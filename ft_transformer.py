from metadata import *
import fastText
import numpy as np

model = fastText.load_model(fasttest_crawl_bin)



def checking_embedding(model,word):
	word = 'abcdefg'
	word_embedding = model.get_word_vector(word)
	subs, subinds = model.get_subwords(word)
	sub_embedings = model.get_input_matrix()[subinds]
	sub_embedding = sub_embedings.sum(axis=0)
	np.allclose(word_embedding,sub_embedding/len(subs))


def clustering_subwords(model):
	pass