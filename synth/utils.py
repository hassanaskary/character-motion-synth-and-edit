import torch

def compute_gram(matrix):
    matrix = matrix.squeeze()
    gram = torch.mm(matrix, torch.t(matrix))
    gram = gram.view(1, gram.size(0), gram.size(1))
    return gram
