from typing import Dict, Iterator, List

import torch
import torch.nn as nn
import math

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention



class TK(nn.Module):
    '''
    Paper: S. Hofstätter, M. Zlabinger, and A. Hanbury 2020. Interpretable & Time-Budget-Constrained Contextualization for Re-Ranking. In Proc. of ECAI 
    
    student note: based on https://github.com/sebastian-hofstaetter/matchmaker/blob/master/matchmaker/models/published/ecai20_tk.py
      adapting it to the code and adding comments to showcase understanding
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_kernels: int,
                 n_layers:int,
                 n_tf_dim:int,
                 n_tf_heads:int):

        super(TK, self).__init__()

        self.word_embeddings = word_embeddings

        # static - kernel size & magnitude variables
        mu = torch.FloatTensor(self.kernel_mus(n_kernels)).view(1, 1, 1, n_kernels)
        sigma = torch.FloatTensor(self.kernel_sigmas(n_kernels)).view(1, 1, 1, n_kernels)

        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)

        #contextualized encoding, positional encoding and transformer

        #adding positional encoding, allows the model to differentiate between elements based on their position
        # which is something nn dont capture
        self.register_buffer("positional_features_q", self.get_positional_features(n_tf_dim,2000))
        self.register_buffer("positional_features_d", self.positional_features_q)

        #calculating the context with a set of Transformer layers, 
        # first, the input sequence is fused with a positional encoding, 
        # followed by a set of n_layers Transformer layers
        #Parameters explained:
        # d_model: the dimension of the input and output embeddings, which is represented by n_tf_dim in our case.
        # nhead: the number of attention heads in the multiheadattention models, which is represented by n_tf_heads in our case
        # dim_feedforward: the dimension of the feedforward network model, which is represented by n_tf_dim in our case
        # dropout: the probability of neuron dropout, which is 0 in our case, meaning no dropout is applied.
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_tf_dim, nhead=n_tf_heads, dim_feedforward=n_tf_dim, dropout=0)
        self.contextualizer = nn.TransformerEncoder(encoder_layer, n_layers, norm=None)

        # Cosine-match matrix for term x term interaction
        self.cosine_module = CosineMatrixAttention()

        # Kernel pooling
        self.kernel_bin_weights = nn.Linear(n_kernels, 1, bias=False)
        self.kernel_alpha_scaler = nn.Parameter(torch.full([1,1,n_kernels], 1, dtype=torch.float32, requires_grad=True))

    def _initialize_weights(self):
        # Initialize the weights of the linear layer using a uniform distribution
        nn.init.uniform_(self.kernel_bin_weights.weight, -0.014, 0.014)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        # shape: (batch, query_max)
        query_pad_oov_mask = (query["tokens"]["tokens"] > 0).float() # > 1 to also mask oov terms
        # shape: (batch, doc_max)
        document_pad_oov_mask = (document["tokens"]["tokens"] > 0).float()

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document)

        #contextualized encoding
        # applied to query and doc sequences individually
        # shape: (batch, query_max, emb_dim)
        query_embeddings = self.forward_representation(query_embeddings, query_pad_oov_mask,self.positional_features_q[:,:query_embeddings.shape[1],:])
        # shape: (batch, document_max, emb_dim)
        document_embeddings = self.forward_representation(document_embeddings, document_pad_oov_mask,self.positional_features_d[:,:document_embeddings.shape[1],:])

        #term x term interaction
        # the interaction match-matrix is created with pairwise cosine similarities
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)

        #kernels, kernel pooling
        # we try to detect different features with rbf kernels
        # shape: (batch, query_max, document_max, n_kernels)
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix.unsqueeze(-1) - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * document_pad_oov_mask.unsqueeze(1).unsqueeze(-1)

        # process each kernel matrix in parallel, begin by summing the document dimension j for each query term and kernel
        per_kernel_query = torch.sum(kernel_results_masked, 2)
        # log normalization applies a logarithm to each query term before summing them up
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query * self.kernel_alpha_scaler, min=1e-10)) 
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #scoring
        # kernels fed into output linear layer which predicts the final score
        score = self.kernel_bin_weights(per_kernel).squeeze(1)

        return score

    def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor,positional_features=None) -> torch.Tensor:
        #applied contextualized embedding

        if positional_features is None:
            positional_features = self.positional_features_d[:,:sequence_embeddings.shape[1],:]

        sequence_embeddings_context = self.contextualizer((sequence_embeddings + positional_features).transpose(1,0),src_key_padding_mask=~sequence_mask.bool()).transpose(1,0)

        # shape: (batch, sequence_max, emb_dim)
        return sequence_embeddings_context

    def get_positional_features(self,dimensions,
                                max_length,
                                min_timescale: float = 1.0,
                                max_timescale: float = 1.0e4):
        # pylint: disable=line-too-long
        """
        Implements the frequency-based positional encoding described
        in `Attention is all you Need
        <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .
        Adds sinusoids of different frequencies to a ``Tensor``. A sinusoid of a
        different frequency and phase is added to each dimension of the input ``Tensor``.
        This allows the attention heads to use absolute and relative positions.
        The number of timescales is equal to hidden_dim / 2 within the range
        (min_timescale, max_timescale). For each timescale, the two sinusoidal
        signals sin(timestep / timescale) and cos(timestep / timescale) are
        generated and concatenated along the hidden_dim dimension.
        Parameters
        ----------
        tensor : ``torch.Tensor``
            a Tensor with shape (batch_size, timesteps, hidden_dim).
        min_timescale : ``float``, optional (default = 1.0)
            The smallest timescale to use.
        max_timescale : ``float``, optional (default = 1.0e4)
            The largest timescale to use.
        Returns
        -------
        The input tensor augmented with the sinusoidal frequencies.
        """
        timesteps=max_length
        hidden_dim = dimensions

        timestep_range = self.get_range_vector(timesteps, 0).data.float()
        # We're generating both cos and sin frequencies,
        # so half for each.
        num_timescales = hidden_dim // 2
        timescale_range = self.get_range_vector(num_timescales, 0).data.float()

        log_timescale_increments = math.log(float(max_timescale) / float(min_timescale)) / float(num_timescales - 1)
        inverse_timescales = min_timescale * torch.exp(timescale_range * -log_timescale_increments)

        # Broadcasted multiplication - shape (timesteps, num_timescales)
        scaled_time = timestep_range.unsqueeze(1) * inverse_timescales.unsqueeze(0)
        # shape (timesteps, 2 * num_timescales)
        sinusoids = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
        if hidden_dim % 2 != 0:
            # if the number of dimensions is odd, the cos and sin
            # timescales had size (hidden_dim - 1) / 2, so we need
            # to add a row of zeros to make up the difference.
            sinusoids = torch.cat([sinusoids, sinusoids.new_zeros(timesteps, 1)], 1)
        return sinusoids.unsqueeze(0)

    def get_range_vector(self, size: int, device: int) -> torch.Tensor:
        """
        Returns a range vector with the desired size, starting at 0. The CUDA implementation
        is meant to avoid copy data from CPU to GPU.
        """
        if device > -1:
            return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
        else:
            return torch.arange(0, size, dtype=torch.long)  
    def kernel_mus(self, n_kernels: int):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
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
        l_sigma = [0.0001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma
