import nltk.corpus, nltk.tag
import nltk
class UnigramChunker(nltk.ChunkParserI):


	def _init_(self,training):
		training_data=[[(x,y) for p,x,y in nltk.chunk.treeconlltags(sent)] for sent in training]
		self.tagger=nltk.UnigramTagger(training_data)

	def parsing(self,sent):
		postags=[pos1 for (word1,pos1) in sent]
		tagged_postags=self.tagger.tag(postags)
		chunk_tags=[chunking for (pos1,chunktag) in tagged_postags]
		conll_tags=[(word,pos1,chunktag) for ((word,pos1),chunktag) in zip(sent, chunk_tags)]
		return nltk.chunk.conlltaags2tree(conlltags)
	
	def ubt_conll_chunk_accuracy(train_sents, test_sents):
		chunks_train =conll_tag_chunks(training)
		chunks_test =conll_tag_chunks(testing)
		chunker1 =nltk.tag.UnigramTagger(chunks_train)
		print(nltk.tag.accuracy(chunker1, chunks_test))
		chunker2 =nltk.tag.BigramTagger(chunks_train, backoff=chunker1)
		print(nltk.tag.accuracy(chunker2, chunks_test))
		chunker3 =nltk.tag.TrigramTagger(chunks_train, backoff=chunker2)
		print (nltk.tag.accuracy(chunker3, chunks_test))
		chunker4 =nltk.tag.TrigramTagger(chunks_train, backoff=chunker1)
		print(nltk.tag.accuracy(chunker4, chunks_test))
		chunker5 =nltk.tag.BigramTagger(chunks_train, backoff=chunker4)
		print(nltk.tag.accuracy(chunker5, chunks_test))
# accuracy test for conll chunking
		conll_train =nltk.corpus.conll2000.chunked_sents('train.txt')
		conll_test =nltk.corpus.conll2000.chunked_sents('test.txt')
		ubt_conll_chunk_accuracy(conll_train, conll_test)
# accuracy test for treebank chunking
		treebank_sents =nltk.corpus.treebank_chunk.chunked_sents()
		ubt_conll_chunk_accuracy(treebank_sents[:2000], treebank_sents[2000:])