from typing import Dict
import torch
import torch.nn as nn
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention

class KNRM(nn.Module):
    '''
    Paper: End-to-End Neural Ad-hoc Ranking with Kernel Pooling, Xiong et al., SIGIR'17

    '''

    def __init__(self, word_embeddings: TextFieldEmbedder, n_kernels: int):
        super(KNRM, self).__init__()

        # Assign the word embeddings to the model
        self.word_embeddings = word_embeddings

        # static - kernel size & magnitude variables
        mu = torch.FloatTensor(self.kernel_mus(n_kernels)).view(1, 1, 1, n_kernels)
        sigma = torch.FloatTensor(self.kernel_sigmas(n_kernels)).view(1, 1, 1, n_kernels)

        # This is used to register a buffer that should not be considered a model parameter.
        # These variables are constant and can be used in the model's forward pass without being optimized.
        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)

        # Create a CosineMatrixAttention module
        self.cos_module = CosineMatrixAttention()

        # Create a linear layer for the final score calculation
        self.dense = nn.Linear(n_kernels, 1)

        # Initialize the weights of the linear layer
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize the weights of the linear layer using a uniform distribution
        nn.init.uniform_(self.dense.weight, -0.014, 0.014)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        # Create a mask to ignore padding tokens in the query
        query_pad_oov_mask = (query["tokens"]["tokens"] > 0).float()

        # Create a mask to ignore padding tokens in the document
        document_pad_oov_mask = (document["tokens"]["tokens"] > 0).float()

        # Embed the query and document using the word embeddings
        query_embeddings = self.word_embeddings(query)
        document_embeddings = self.word_embeddings(document)

        # Create a binary mask indicating the alignment between query terms and document terms
        query_by_doc_mask = torch.einsum("bij, bkj -> bikj", query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1))

        # Calculate the cosine similarity matrix between query and document embeddings
        cosine_matrix = self.cos_module.forward(query_embeddings, document_embeddings)

        # Apply the query-by-document mask to the cosine similarity matrix
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask.squeeze(-1)

        # Calculate the similarity scores using the Gaussian kernels
        similarity_scores = torch.exp(-torch.pow(cosine_matrix_masked.unsqueeze(-1) - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))

        # Apply the query-by-document mask to the similarity scores
        kernel_scores = similarity_scores * query_by_doc_mask

        # Calculate the soft-TF score for each query term
        per_kernel_query = torch.sum(kernel_scores, dim=2)

        # Take the logarithm of the soft-TF scores and apply a mask to exclude padding values
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * 0.01
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1)

        # Sum the soft-TF scores for each kernel
        per_kernel = torch.sum(log_per_kernel_query_masked, dim=1)

        # Pass the summed scores through the linear layer to calculate the final score
        output = self.dense(per_kernel).squeeze(1)

        return output

    def kernel_mus(self, n_kernels: int):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)
        l_mu.append(1 - bin_size / 2)
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def kernel_sigmas(self, n_kernels: int):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.0001]
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma
