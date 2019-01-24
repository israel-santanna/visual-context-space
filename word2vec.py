import multiprocessing
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


class WordVector(object):
    def __init__(self, window, model_path=None, doc=False):
        super(WordVector, self).__init__()
        cores = multiprocessing.cpu_count()
        self.window = window
        self.doc = doc
        if self.doc:
            self.model = Doc2Vec(dm=1, dm_concat=1, size=100,
                                 window=self.window, negative=0, hs=1,
                                 min_count=1, workers=cores, sample=1e-4,
                                 iter=20)
        else:
            self.model = Word2Vec(size=100, window=self.window, min_count=1,
                                  workers=cores, sg=1, hs=1, negative=0)
        self.model_path = model_path if model_path else "data/trained_model"

    def create_corpus(self, coco):
        corpus = []
        for img in coco.images_ids():
            catgs_img = coco.get_image_categories(img)
            img_text = [coco.get_category_name(c) for c in catgs_img]
            if self.doc:
                corpus.append(TaggedDocument(img_text, [img]))
            else:
                corpus.append(img_text)
        return corpus

    def train(self, train_corpus, load=False):
        self.model.build_vocab(train_corpus)
        print("Vocabulary builded")
        print("Training...")
        self.model.train(train_corpus,
                         total_examples=self.model.corpus_count,
                         epochs=self.model.iter)
        print("Finished training")
        self.model.save(self.model_path)
        print("Saved")

    def load(self, load_path=None):
        load_path = load_path if load_path else self.model_path
        self.model = Doc2Vec.load(load_path) if self.doc \
            else Word2Vec.load(load_path)
