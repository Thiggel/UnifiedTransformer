from torch import cat, hstack, unsqueeze, device, cuda, Tensor
from torch.nn import Module, \
    Linear, \
    Softmax


class MultiHeadAttention(Module):
    def __init__(self, d, n_heads=2):
        super(MultiHeadAttention, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = [Linear(d_head, d_head) for _ in range(self.n_heads)]
        self.k_mappings = [Linear(d_head, d_head) for _ in range(self.n_heads)]
        self.v_mappings = [Linear(d_head, d_head) for _ in range(self.n_heads)]
        self.d_head = d_head
        self.softmax = Softmax(dim=-1)

        self.dev = device("cuda:0" if cuda.is_available() else "cpu")

        self.attention_buffer = None

    def forward(self, sequences, mask: Tensor = None):
        # (N, seq_length, token_dim)
        # --> (N, seq_length, n_heads, token_dim / n_heads)
        # --> (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head].to(self.dev)
                k_mapping = self.k_mappings[head].to(self.dev)
                v_mapping = self.v_mappings[head].to(self.dev)

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                scores = q @ k.T / (self.d_head ** 0.5)

                if mask is not None:
                    scores = scores.masked_fill(mask.eq(0), -1e9)

                attention = self.softmax(scores)

                # save for creating attention maps
                self.attention_buffer = attention

                seq_result.append(attention @ v)
            result.append(hstack(seq_result))
        return cat([unsqueeze(r, dim=0) for r in result])
