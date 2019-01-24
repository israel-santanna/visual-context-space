from pycocotools.coco import COCO
from pandas import DataFrame
import numpy as np


class CocoModel(object):
    def __init__(self, path):
        super(CocoModel, self).__init__()
        self.path = path
        self.coco = COCO(self.path)
        self.annotations = self.coco.loadAnns(self.coco.getAnnIds())
        self.df = DataFrame(self.annotations)
        self.imgs_ids = None
        self.catgs_ids = None
        self.catgs_names = None
        self.cooc_matrix = None

    def images_ids(self):
        if not self.imgs_ids:
            self.imgs_ids = list(self.df['image_id'].unique())
        return self.imgs_ids

    def categories_ids(self):
        if not self.catgs_ids:
            self.catgs_ids = list(self.df['category_id'].unique())
        return self.catgs_ids

    def categories_names(self):
        if not self.catgs_names:
            catgs = self.coco.loadCats(self.categories_ids())
            self.catgs_names = [c['name'].replace(' ', '_') for c in catgs]
        return self.catgs_names

    def max_objects_per_image(self):
        return self.df.groupby('image_id').size().max()

    def get_image_categories(self, img_id):
        return list(self.df[self.df.image_id == img_id]['category_id'])

    def get_category_name(self, catg_id):
        return self.categories_names()[self.categories_ids().index(catg_id)]

    def get_category_id_by_name(self, catg_name):
        if catg_name == "nop":
            return -1
        return self.categories_ids()[self.categories_names().index(catg_name)]

    def cooccurrence_matrix(self):
        if self.cooc_matrix is None:
            self.cooc_matrix = np.zeros((len(self.categories_ids()),
                                         len(self.categories_ids())),
                                        dtype=np.int32)
            for img in self.images_ids():
                catgs_img = self.get_image_categories(img)
                for i, c_id in enumerate(catgs_img):
                    i_id = self.categories_ids().index(c_id)
                    for j in range(i + 1, len(catgs_img)):
                        j_id = self.categories_ids().index(catgs_img[j])
                        self.cooc_matrix[i_id, j_id] += 1
            np.fill_diagonal(self.cooc_matrix, 0)
        return self.cooc_matrix

    def topn_coocurrences(self, catg_id, n=10):
        idx = self.categories_ids().index(catg_id)
        # Get the catg_id equivalent cooc_matrix row
        # Get the indices that would sort this row
        # Reverse the indices array to get coocurrences in descending order
        # Get only the n most coocurring categories indices
        # Map the indices to the categories id and return them
        sort_row = np.argsort(self.cooccurrence_matrix()[idx, :])[::-1][:n]
        return [self.categories_ids()[i] for i in sort_row]
