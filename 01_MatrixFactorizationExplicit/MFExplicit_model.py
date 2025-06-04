import torch
import torch.nn as nn


class MatrixFactorizationExplicitModel(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, use_user_bias, use_item_bias, use_global_bias):
        super().__init__()

        self.user_matrix = nn.Embedding(num_embeddings=num_users, embedding_dim=latent_dim)
        self.item_matrix = nn.Embedding(num_embeddings=num_items, embedding_dim=latent_dim)

        if use_user_bias:
            self.user_bias_matrix = nn.Embedding(num_embeddings=num_users, embedding_dim=1)
        else:
            self.register_buffer('user_bias_matrix', torch.zeros(num_users,1))

        if use_item_bias:
            self.item_bias_matrix = nn.Embedding(num_embeddings=num_items, embedding_dim=1)
        else:
            self.register_buffer('item_bias_matrix', torch.zeros(num_items,1))
        
        if use_global_bias:
            self.global_bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("global_bias", torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        # Initialise embedding matrices
        nn.init.xavier_uniform_(self.user_matrix.weight)
        nn.init.xavier_uniform_(self.item_matrix.weight)

        # Initialise bias terms
        if isinstance(self.user_bias_matrix, nn.Embedding):
            nn.init.zeros_(self.user_bias_matrix.weight)

        if isinstance(self.item_bias_matrix, nn.Embedding):
            nn.init.zeros_(self.item_bias_matrix.weight)

        if isinstance(self.global_bias, nn.Parameter):
            nn.init.zeros_(self.global_bias)

    def forward(self, user_idx, item_idx):

        # get the base predictions
        base_pred = (self.user_matrix(user_idx)*self.item_matrix(item_idx)).sum(dim=1)

        # get user bias
        if isinstance(self.user_bias_matrix, nn.Embedding):
            user_bias = self.user_bias_matrix(user_idx).squeeze()
        else:
            user_bias = self.user_bias_matrix[user_idx].squeeze()

        # get item bias
        if isinstance(self.item_bias_matrix, nn.Embedding):
            item_bias = self.item_bias_matrix(user_idx).squeeze()
        else:
            item_bias = self.item_bias_matrix[user_idx].squeeze()

        # get global bias
        global_bias = self.global_bias

        pred = base_pred + user_bias + item_bias + global_bias

        return pred

