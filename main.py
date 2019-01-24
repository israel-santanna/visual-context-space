import argparse
from coco_model import CocoModel
from word2vec import WordVector


def parse_args():
    parser = argparse.ArgumentParser(description="An experiment with " +
                                     "object's context in the COCO dataset")
    parser.add_argument("-i", "--input",
                        help="File containing the COCO training data",
                        default="data/instances_train2017.json")
    parser.add_argument("-v", "--validation",
                        help="File containing the COCO validation data",
                        default="data/instances_val2017.json")
    parser.add_argument("-s", "--save_path",
                        help="Path where to save the Word2Vec trained data",
                        default="data/trained_model")
    parser.add_argument("-l", "--load_path",
                        help="File containing the Word2Vec trained data",
                        default="")
    parser.add_argument("-m", "--men_path",
                        help="File containing the MEN dataset for evaluation",
                        default="")
    parser.add_argument("-c", "--console", action='store_true',
                        help="Show a console where you can query W2V " +
                        "for the objects more commonly in the same context")
    return parser.parse_args()


def test_words_similarities(coco, w2v):
    while True:
        q = input("Type a word from the trained vocabulary, or 'q' to exit:\n")
        if q == 'q':
            break
        elif q == 'p':
            print(coco.categories_names())
        else:
            words = q.split()
            sims = w2v.model.most_similar(words)
            print(sims)


def validate_w2v(coco_train, coco_val, w2v):
    total = 0.0
    right_sims = 0.0
    right_cooc = 0.0
    cooc_sims = 0.0
    max_catgs = 0
    for img in coco_val.images_ids():
        catgs_img = coco_val.get_image_categories(img)
        catgs_set = set(catgs_img)
        max_catgs = max(max_catgs, len(catgs_set))
        if len(catgs_set) < 2:
            continue
        for catg in catgs_set:
            total += len(catgs_set) - 1
            # Train and Validation categories are the same,
            # so no need to check them here
            # (but their array indexes are probably different)
            sims = w2v.model.similar_by_word(coco_val.get_category_name(catg),
                                             topn=(len(catgs_set) - 1))
            sims = [coco_train.get_category_id_by_name(i[0]) for i in sims]
            right_sims += len(catgs_set.intersection(sims))

            cooc = coco_train.topn_coocurrences(catg, n=(len(catgs_set) - 1))
            right_cooc += len(catgs_set.intersection(cooc))

            cooc_sims += len(set(cooc).intersection(sims))

    print("Max categories in an image = ", max_catgs)
    print("Number of objects (not really) = ", total)
    print("Number of W2V correct objects = ", right_sims)
    print("Number of Coocurrence correct objects = ", right_cooc)
    print("Number of Coocurrence and W2V intersection = ", cooc_sims)
    print("Accuracy W2V = ", right_sims / total)
    print("Accuracy Coocurrence = ", right_cooc / total)
    print("Coocurrence and W2V intersection rate = ", cooc_sims / total)


def read_men(path):
    words = set()
    scores = []
    with open(path, "r", encoding="ISO-8859-1") as file:
        for line in file:
            word1, word2, score = line.split(" ")
            score = float(score)
            if score != 0.0:
                score /= 50
            words.add(word1)
            words.add(word2)
            scores.append([word1, word2, score])
    return scores, words


if __name__ == "__main__":
    args = parse_args()
    coco_train = CocoModel(args.input)
    print("Number of images on " + args.input + " = ",
          len(coco_train.images_ids()))
    print("Number of objects on " + args.validation + " = ",
          len(coco_train.annotations))
    print("Number of categories on " + args.input + " = ",
          len(coco_train.categories_ids()))
    print("Max number of objects in an image = ",
          coco_train.max_objects_per_image())

    if args.load_path:
        w2v = WordVector(100, args.load_path)
        w2v.load()
        print("Loaded W2V")
    else:
        w2v = WordVector(100, args.save_path)
        corpus = w2v.create_corpus(coco_train)
        print("Created Corpus")
        w2v.train(corpus)

    if args.console:
        test_words_similarities(coco_train, w2v)
    else:
        coco_val = CocoModel(args.validation)
        print("Number of images on " + args.validation + " = ",
              len(coco_val.images_ids()))
        print("Number of objects on " + args.validation + " = ",
              len(coco_val.annotations))
        print("Number of categories on " + args.validation + " = ",
              len(coco_val.categories_ids()))
        print("Max number of objects in an image = ",
              coco_val.max_objects_per_image())
        validate_w2v(coco_train, coco_val, w2v)

        if args.men_path:
            men_scores, men_words = read_men(args.men_path)
            print("# words in MEN dataset = ", len(men_words))
            print("# pairs in MEN dataset = ", len(men_scores))
            words = list(men_words.intersection(coco_train.categories_names()))
            print(words)
            print("# common words in COCO and MEN = ", len(words))
            scores = [s for s in men_scores if s[0] in words and s[1] in words]
            print("# pairs with the common words = ", len(scores))
            print(scores)
            for s in scores:
                print("-------")
                print(s[0])
                sims = w2v.model.similar_by_word(s[0], topn=10)
                print(sims)
                catg_id = coco_train.get_category_id_by_name(s[0])
                cooc = coco_train.topn_coocurrences(catg_id, n=10)
                cooc_names = [coco_train.get_category_name(i) for i in cooc]
                print(cooc_names)
                print(s[1])
                sims = w2v.model.similar_by_word(s[1], topn=10)
                print(sims)
                catg_id = coco_train.get_category_id_by_name(s[1])
                cooc = coco_train.topn_coocurrences(catg_id, n=10)
                cooc_names = [coco_train.get_category_name(i) for i in cooc]
                print(cooc_names)
