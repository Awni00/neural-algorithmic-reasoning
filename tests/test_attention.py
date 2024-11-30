import unittest
import torch
import sys, os; sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from models.attention import Attention
from models.positional_encoding import RotaryPositionalEmbeddings, AlibiPositionalBias, T5RelativePositionBias

class TestAttention(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.n_heads = 8
        self.d_head = 64 // 8
        self.seq_len = 10
        self.batch_size = 2

    def test_attention_no_pos_enc(self):
        attn = Attention(d_model=self.d_model, n_heads=self.n_heads, pos_enc_model=None)
        self._run_attention_test(attn)

    def test_attention_rotary_pos_enc(self):
        pos_enc_model = RotaryPositionalEmbeddings(dim=self.d_head)
        attn = Attention(d_model=self.d_model, n_heads=self.n_heads, pos_enc_model=pos_enc_model)
        self._run_attention_test(attn)

    def test_attention_alibi_pos_enc(self):
        pos_enc_model = AlibiPositionalBias(heads=self.n_heads)
        attn = Attention(d_model=self.d_model, n_heads=self.n_heads, pos_enc_model=pos_enc_model)
        self._run_attention_test(attn)

    def test_attention_t5_pos_enc(self):
        pos_enc_model = T5RelativePositionBias(heads=self.n_heads)
        attn = Attention(d_model=self.d_model, n_heads=self.n_heads, pos_enc_model=pos_enc_model)
        self._run_attention_test(attn)

    def _run_attention_test(self, attn):
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)

        for is_causal in [True, False]:
            output1, scores1 = attn(query, key, value, is_causal=is_causal, need_weights=True)
            output2, scores2 = attn(query, key, value, is_causal=is_causal, need_weights=False)

            self.assertTrue(
                torch.allclose(output1, output2, atol=1e-7), # Alibi actually fails at atol 1e-8
                f"Outputs differ for is_causal={is_causal} between need_weights=True and need_weights=False"
                f"\noutput1={output1}\noutput2={output2}")
            self.assertIsNotNone(scores1, "Scores should be returned when need_weights=True")


if __name__ == "__main__":
    unittest.main()