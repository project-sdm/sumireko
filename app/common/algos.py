import math

import numpy as np

from app.common.state import PreprocessedData


def knn(descriptors: np.ndarray, data: PreprocessedData, k: int | None):
    _, labels = data.word_index.search(descriptors, 1)
    q_hist = np.bincount(labels.ravel(), minlength=len(data.words))

    n = len(data.media_files)

    def weight(word_id: int, tf: int) -> float:
        return math.log(1 + tf) * math.log((n + 1) / (data.df[word_id] + 1))

    scores: dict[int, float] = {}
    query_len_sq = 0.0

    for word_id, tf_query in enumerate(q_hist):
        if tf_query == 0:
            continue

        w_query = weight(word_id, tf_query)
        query_len_sq += w_query**2

        for img_id, w_img in data.index[word_id]:
            scores[img_id] = scores.get(img_id, 0) + w_img * w_query

    query_length = math.sqrt(query_len_sq)

    for img_id in scores:
        scores[img_id] /= data.lengths[img_id] * query_length

    result = sorted(scores.items(), key=lambda tup: tup[1], reverse=True)
    top_files = [data.media_files[i] for i, _ in result[:k]]

    return top_files
