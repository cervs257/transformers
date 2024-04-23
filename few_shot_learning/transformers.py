import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model:int, vocab_size:int):
        """
        :param d_model: The dimension of the model
        :param vocab_size: The size of the vocabulary
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len:int, dropout: float) -> None:
        """
        :param d_model: The dimension of the model
        :param max_len: The maximum length of the input sequence
        :param dropout: The dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = seq_len # in original paper, max_len = 512
        self.dropout = nn.Dropout(p=dropout) # avoid overfitting

        # create constant 'pe' matrix of shape (max_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (max_len, 1)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply sin to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        # apply cos to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe) # register the buffer to save the tensor in the model

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # this isn't learned, positions are fixed
        return self.dropout(x)
    
class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 1e-6) -> None:
        """
        :param eps: A small number to avoid division by zero
        """
        super().__init__()
        self.eps = eps
        # create two learnable parameters to calibrate normalization
        self.alpha = nn.Parameter(torch.ones(1)) # using nn.Parameter to make it trainable
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True) # .mean usually returns a scalar, but keepdim=True returns a tensor with the same dimensions as x
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    
        def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
            """
            :param d_model: The dimension of the model
            :param d_ff: The dimension of the feedforward network
            :param dropout: The dropout rate
            """
            super().__init__()
            self.d_model = d_model
            self.d_ff = d_ff
            self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1 from the paper (bias is True by default in PyTorch, so no need to specify it)
            self.dropout = nn.Dropout(dropout)
            self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2 from the paper
            self.activation = nn.ReLU()
    
        def forward(self, x):
            # (Batch, seq_len, d_model) -> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)
            x = self.activation(self.linear_1(x))
            x = self.dropout(x)
            x = self.linear_2(x)
            return x
        
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        """
        :param d_model: The dimension of the model
        :param heads: The number of heads
        :param dropout: The dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        # the embedding vector needs to be divided into heads number of heads
        assert d_model % heads == 0, "d_model not divisible by heads"
        # define d_k so that it can be divided by heads .. follows nomencalture from the paper
        self.d_k = d_model // heads

        # matrices WQ, WK, WV from the paper
        self.w_q = nn.Linear(d_model, d_model) # WQ
        self.w_k = nn.Linear(d_model, d_model) # WK
        self.w_v = nn.Linear(d_model, d_model) # WV

        # output matrix Wo
        self.w_o = nn.Linear(d_model, d_model) # Wo

        self.dropout = nn.Dropout(dropout)

    @staticmethod # this is a static method, so it doesn't need to be called on an instance of the class
    def attention(query, key, value, dropout: nn.Dropout, mask=None):
        d_k = query.shape[-1] # get the last dimension of the query

        # (Batch, heads, seq_len, d_k) x (Batch, heads, d_k, seq_len) -> (Batch, heads, seq_len, seq_len):
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) 
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9) # fill the scores with a very small number where the mask is 0

        # softmax attention ####### what if i don't want to use softmax?
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1) # apply softmax to the last dimension (Batch, heads, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask=None): # mask hides specific words in the sequence
        query = self.w_q(q) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.w_k(k) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)

        # we want to split the d_model into heads number of heads so that we can perform self-attention
        # each of the previous is fed into the different heads
        # (Batch, seq_len, d_model) -> (Batch, seq_len, heads, d_k) -> (Batch, heads, seq_len, d_k)
        # this means that each head watches d_k smaller parts of the embedding: seq_len x d_k
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, self.dropout, mask=mask)

        # (Batch, heads, seq_len, d_k) -> (Batch, seq_len, heads, d_k) -> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.d_k)

        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    
    def __init__(self, dropout: float) -> None:
        """
        :param dropout: The dropout rate
        """
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        

class EncoderBlock(nn.Module): # in the video he calls this the EncoderBlock

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """
        :param self_attention_block: The self-attention block
        :param feed_forward_block: The feedforward block
        :param dropout: The dropout rate
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # 2 residual connections
        # self.residual = ResidualConnection(dropout)
        # self.feed_forward_residual = ResidualConnection(dropout)

    def forward(self, x, src_mask): # src_mask is the padding mask. we don't want to pay attention to the padding
        # in the same operation, perform skip connection and attention block ... input x, x, x corresponding to q, k, v
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # in the same operation, perform skip connection and feed forward block
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module): # in the video he calls this the Encoder
    '''
    Note that the Encoder can be made up of multiple EncoderBlocks (up to N according to the paper)
    '''

    def __init__(self, layers: nn.ModuleList) -> None:
        """
        :param layers: The number of layers
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers: # each layer is an EncoderBlock
            x = layer(x, mask) # EncoderBlock forward function inputs
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    '''
    Note that the DecoderBlock is made up of three parts:
    1. Self-Attention Block as previously defined
    2. Encoder-Decoder Attention Block aka Cross-Attention Block:
        - This block takes the encoder output as keys and values
        - This block takes the decoder output from the previous layer as query
    3. Feed Forward Block as previously defined
    '''

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """
        :param self_attention_block: The self-attention block
        :param cross_attention_block: The cross-attention block
        :param feed_forward_block: The feedforward block
        :param dropout: The dropout rate
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask): # src_mask comes from the encoder, tgt_mask is the mask for the decoder
        # self attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # cross attention block
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    '''
    Note that the Decoder can be made up of multiple DecoderBlocks (up to N according to the paper)
    '''

    def __init__(self, layers: nn.ModuleList) -> None:
        """
        :param layers: The number of layers
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers: # each layer is a DecoderBlock
            x = layer(x, encoder_output, src_mask, tgt_mask) # DecoderBlock forward function inputs
        return self.norm(x)
    

# we have the Encoder and Decoder. We need the Linear layer to convert the output of the Decoder to the output space (aka Projection Layer)
## as is, we're using the log_softmax function to convert the output to probabilities within the ProjectionLayer ... remove for HW?

class ProjectionLayer(nn.Module):
    '''
    This layer projects the output of the Decoder to the output space
    '''

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        :param d_model: The dimension of the model
        :param vocab_size: The size of the vocabulary
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1) # log_softmax is used for numerical stability
    
class Transformer(nn.Module):
    '''
    This is the main model that puts everything together
    '''

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        """
        :param encoder: The encoder
        :param decoder: The decoder
        :param src_embed: The source embeddings
        :param tgt_embed: The target embeddings
        :param src_pos: The positional encoding for the source
        :param tgt_pos: The positional encoding for the target
        :param projection_layer: The projection layer
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj = projection_layer

    # why not build a forward function as we've done before? During inferencing we can reuse the output of the encoder and visualizing attention

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

# we'll now define a function that, given parameters, builds the model
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, heads: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer : 
    '''
    This function builds the transformer model
    - language is based on the translation task to be performed in the video but applicable to any task
    '''
    # create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialize the parameters with xavier uniform
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer