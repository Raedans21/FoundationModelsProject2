import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Instantiate positional embedding tensor to zeros
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # Calculate interleaved sinusoidal position embeddings according to convention
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1),:]
        return self.dropout(x)

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_layers=3, num_heads=4, dropout=0.2, max_seq_len=512, pad_token_id=3):
        """
        Create a Transformer language model.
        :param vocab_size: size of the vocabulary
        :param embed_dim: size of each token's embedding vector
        :param num_layers: number of transformer encoder layers to stack
        :param num_heads: number of attention heads per encoder layer
        :param dropout: training dropout rate
        :param max_seq_len: maximum sequence length
        :param pad_token_id: token ID of <pad> token
        """
        super(TransformerLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Define embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        # Define positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout, max_len=max_seq_len)
        # Define stackable transformer encoder layer
        encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        # Define transformer encoder
        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)
        # Output layer that maps hidden state of final transformer encoder to output
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        """
        Compute model output logits given a sequence.
        :param input_ids: Sequence of input token IDs, Tensor of shape (batch_size, seq_len)
        :return: output logits, Tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()

        embeds = self.embedding(input_ids) # compute embeddings for all input tokens in parallel
        embeds = self.positional_encoding(embeds) # compute positional embeddings for all embedded tokens

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))

        transformer_output = self.transformer(embeds, mask=mask)

        logits = self.fc(transformer_output) # compute output logits
        return logits

    def predict_next_token(self, input_ids, temperature=1.0):
        """
        Predict the next token ID (and hidden state) from the last token in input_ids.
        :param input_ids: Input sequence token IDs
        :param temperature: setting for sampling
        :return: next token ID, hidden state
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature

            # Sort output logits to apply top-p filtering.
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            # Compute the probability distribution of the sorted logits.
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            # Compute the cumulative sum of the sorted logit probabilities.
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Nucleus sampling p value
            top_p = 0.9
            # Create boolean mask to remove cumulative probabilities greater than top_p
            mask = cumulative_probs > top_p
            # Avoid masking the first token to handle situations when the first token probability is greater than top_p
            mask[..., 0] = False
            # Set any masked elements to negative infinity to set probability to zero in softmax call
            sorted_logits[mask] = -float('Inf')
            # Transform the filtered logits back to probabilities, zeroing out masked token_ids
            filtered_probs = torch.softmax(sorted_logits, dim=-1)
            # Sample one token index from the filtered probability distribution using weighted sampling
            next_token_index = torch.multinomial(filtered_probs, num_samples=1)
            # Map the selected token index back to onto the original vocabulary indices
            next_token_id = sorted_indices.gather(1, next_token_index)
            return next_token_id.item()

    def generate(self, tokenizer, prompt, max_length=50, eos_token_id=None, temperature=1.0, device='cpu'):
        """
        Generate a full output sequence given a prompt.
        :param tokenizer: The trained SentencePiece tokenizer
        :param prompt: The input prompt (plain text string)
        :param max_length: Maximum number of tokens to generate auto-regressively before stopping
        :param eos_token_id: The token ID of the EOS token
        :param temperature: Temperature setting for sampling
        :param device: Device we are using to run the model
        """
        self.eval()
        input_ids = tokenizer.encode(prompt, out_type=int) # Encode the input string into token IDs
        # convert token ID list to a tensor, move to device memory, and add a batch dimension (1D to 2D)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated_ids = []

        # loop over map output tokens
        for _ in range(max_length):
            next_token_id = self.predict_next_token(input_tensor, temperature)
            # exit early if the model generated <eos> token ID
            if eos_token_id is not None and next_token_id == eos_token_id:
                break
            # keep track of generated tokens
            generated_ids.append(next_token_id)
            # the input to the next step is just the new token and the hidden state
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        # decode generated token IDs into tokens
        return tokenizer.decode(generated_ids, out_type=str)