from pycocotools.coco import COCO
from pandas import DataFrame
import numpy as np


coco = COCO("data/instances_train2017.json")

anns = coco.loadAnns(coco.getAnnIds())
df = DataFrame(anns)
imgs_ids = list(df['image_id'].unique())
print("n_imgs = ", len(imgs_ids))
cats_ids = list(df['category_id'].unique())
print("n_cats = ", len(cats_ids))
cats = coco.loadCats(cats_ids)

matrix = np.zeros((len(cats_ids), len(cats_ids)), dtype=np.int32)
# cooccurrence = DataFrame(0, index=cats_ids, columns=cats_ids, dtype='int32')
for img in imgs_ids:
    cats_img = df[df.image_id == img]['category_id'].tolist()
    for i, c_id in enumerate(cats_img):
        i_id = cats_ids.index(c_id)
        for j in range(i + 1, len(cats_img)):
            j_id = cats_ids.index(cats_img[j])
            matrix[i_id, j_id] += 1
            matrix[j_id, i_id] += 1
            # cooccurrence[c_id][cats[j]] += 1
print(matrix)
# print(cooccurrence)
