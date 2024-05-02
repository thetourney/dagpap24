import torch


class ConvBlock(torch.nn.Module):

    def __init__(self, k_size, dim):
        """1D convolution followed by a normalization layer and activation.

        Arguments:
            k_size -- Kernel size
            dim -- Number of channels
        """
        super().__init__()
        self.conv = torch.nn.Conv1d(
            dim, dim, k_size, padding='same', bias=False
        )
        self.norm = torch.nn.LayerNorm(dim, eps=1e-6)
        self.activ = torch.nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.activ(x)
        return x


class Model(torch.nn.Module):

    def __init__(self, n_tokens, n_chars, n_attrs, n_classes, padding_idx, hidden_dim, k_size, n_layers, bidirectional, dropout):
        """Classify each token.

        Arguments:
            n_tokens -- Number of tokens in the vocabulary
            n_chars -- Number of characters in the vocabulary
            n_attrs -- Number of word attributes
            n_classes -- Number of classes
            padding_idx -- Integer used for padding
            hidden_dim -- Model hidden dimension
            k_size -- Size of the kernel used in convolution
            n_layers -- Number of recurrent layers
            bidirectional -- Whether the recurrent layers are bidirectional
            dropout -- Dropout between the recurrent layers
        """
        super().__init__()

        self.token_embed = torch.nn.Embedding(
            num_embeddings=n_tokens,
            embedding_dim=hidden_dim,
            padding_idx=padding_idx
        )

        self.char_embed = torch.nn.Embedding(
            num_embeddings=n_chars,
            embedding_dim=hidden_dim,
            padding_idx=padding_idx
        )
        self.char_conv = ConvBlock(k_size, hidden_dim)

        self.attr_embed = torch.nn.Embedding(
            num_embeddings=n_attrs,
            embedding_dim=hidden_dim,
            padding_idx=padding_idx
        )

        self.rnn = torch.nn.GRU(
            hidden_dim, hidden_dim, num_layers=n_layers, bias=False, bidirectional=bidirectional, dropout=dropout,
        )

        logits_input_dim = (1 + bidirectional) * hidden_dim
        self.logits = torch.nn.Linear(logits_input_dim, n_classes, bias=False)

    def forward(self, inputs):
        """Forward pass on a single sequence.
        The expected input is a dictionnary containing the following keys:
        - 'tokens' which value has shape (n_words,)
        - 'chars' which value has shape (n_words, n_characters)
        - 'attrs' which value has shape (n_words,)
        It works on a single sequence of tokens.

        Arguments:
            inputs -- Dictionnary containing 'tokens', 'chars' and 'attrs'.

        Returns:
            Logits for each token.
        """
        tokens, chars, attrs = inputs['tokens'], inputs['chars'], inputs['attrs']

        token_embeddings = self.token_embed(tokens)

        char_embeddings = self.char_embed(chars)
        char_embeddings = char_embeddings.permute(0, 2, 1)
        char_embeddings = self.char_conv(char_embeddings)
        char_embeddings = torch.amax(char_embeddings, dim=2)

        attr_embeddings = self.attr_embed(attrs)

        embeddings = token_embeddings + char_embeddings + attr_embeddings
        embeddings, _ = self.rnn(embeddings)

        return self.logits(embeddings)
