from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[4, 5, 6], [1, 2, 3]])
res = cosine_similarity(a, b)
print(res)


def test_cosine_similarity():
    # 创建两个向量
    vector_a = np.array([[1, 2, 3], [4, 5, 6]])
    vector_b = np.array([[4, 5, 6], [1, 2, 3]])

    # 计算余弦相似度
    similarity = cosine_similarity(vector_a, vector_b)
    print(similarity)
    print("Cosine Similarity between vector_a and vector_b:", similarity[0][0])


# 调用测试函数
test_cosine_similarity()