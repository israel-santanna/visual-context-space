import multiprocessing
from gensim.models import Word2Vec


class WordVector(object):
    """docstring for WordVector"""
    def __init__(self, window, model_path=None):
        super(WordVector, self).__init__()
        cores = multiprocessing.cpu_count()
        self.window = window
        self.model = Word2Vec(size=100, window=self.window, min_count=1,
                              workers=cores, sg=1)
        self.model_path = model_path if model_path else "data/trained_model"

    def create_corpus(self, df, cats_ids, imgs_ids, cats_names):
        texts = []
        for img in imgs_ids:
            cats_img = df[df.image_id == img]['category_id'].tolist()
            img_text = [cats_names[cats_ids.index(c)]
                        for c in cats_img]
            # nops = ['nop'] * (self.window - len(img_text))
            # if nops:
            #     img_text.extend(nops)
            texts.append(img_text)
        return texts

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

    def load(self):
        self.model = Word2Vec.load(self.model_path)

    def model(self):
        return self.model
