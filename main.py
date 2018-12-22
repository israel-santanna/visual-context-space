import argparse
from pycocotools.coco import COCO
from pandas import DataFrame
from cooccurrence import cooccurrence_matrix
from word2vec import WordVector


def parse_args():
    parser = argparse.ArgumentParser(description="Object's Context")
    parser.add_argument("-i", "--input",
                        help="File containing the COCO data",
                        default="data/instances_train2017.json")
    parser.add_argument("-l", "--load_path",
                        help="File containing the Word2Vec trained data",
                        default="data/trained_model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    coco = COCO(args.input)
    anns = coco.loadAnns(coco.getAnnIds())

    df = DataFrame(anns)
    imgs_ids = list(df['image_id'].unique())
    print("Number of images = ", len(imgs_ids))
    cats_ids = list(df['category_id'].unique())
    print("Number of categories = ", len(cats_ids))
    cats = coco.loadCats(cats_ids)
    cats_names = [c['name'].replace(' ', '_') for c in cats]
    # cats_names.append("nop")
    print(cats_names)
    # cooc = cooccurrence_matrix(df, cats_ids, imgs_ids)
    # print(cooc)
    max_objs = df.groupby('image_id').size().max()
    print("Max number of objects in an image = ", max_objs)
    # w2v = WordVector(max_objs, args.load_path)
    w2v = WordVector(5, args.load_path)
    print("Created W2V")
    corpus = w2v.create_corpus(df, cats_ids, imgs_ids, cats_names)
    print("Created Corpus")
    w2v.train(corpus)
    # w2v.load()
    while True:
        q = input("Type a word from the trained vocabulary, or 'q' to exit:\n")
        if q == 'q':
            break
        elif q == 'p':
            print(cats_names)
        else:
            words = q.split()
            sims = w2v.model.most_similar(words)
            print(sims)
