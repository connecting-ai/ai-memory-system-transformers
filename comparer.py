from numpy import dot
from numpy.linalg import norm

def cosine_similarity(v1, v2):
    cos_sim = dot(v1, v2)/(norm(v1)*norm(v2))
    return cos_sim