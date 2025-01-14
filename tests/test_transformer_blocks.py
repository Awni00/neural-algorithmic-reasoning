import unittest
import torch
import sys, os; sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from models.positional_encoding import RotaryPositionalEmbeddings, AlibiPositionalBias, T5RelativePositionBias

from models.transformer_blocks import EncoderBlock, DecoderBlock

class TestTransformerBlocks(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.n_heads = 4
        self.n_heads_cross = 4
        self.dropout_rate = 0.0

        self.seq_len = 13
        self.batch_size = 2

        self.encoder_common_kwargs = dict(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate
        )

        self.decoder_common_kwargs = dict(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_heads_cross=self.n_heads_cross,
            dropout_rate=self.dropout_rate
        )

        self.positional_encoding_types = [
            None,
            RotaryPositionalEmbeddings(dim=self.d_model // self.n_heads),
            AlibiPositionalBias(self.n_heads),
            T5RelativePositionBias(self.n_heads)
        ]

    def test_encoder_block(self):
        for pe_type in self.positional_encoding_types:
            for causal in [False, True]:
                encoder_block = EncoderBlock(
                    **self.encoder_common_kwargs,
                    pos_enc_model=pe_type,
                    causal=causal
                )
                with self.subTest(positional_encoding=pe_type, causal=causal):
                    self.assertIsNotNone(encoder_block)
                    x = torch.randn(self.batch_size, self.seq_len, self.d_model)
                    output = encoder_block(x)
                    self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

                # check that need_intermediate=True works
                with self.subTest(positional_encoding=pe_type, causal=causal, need_intermediate=True):
                    x, intermediate_results = encoder_block(x, need_intermediate=True)
                    self.assertEqual(x.shape, (self.batch_size, self.seq_len, self.d_model))
                    self.assertIsInstance(intermediate_results, dict)
                    for key in ['self_attn_scores']:
                        self.assertIn(key, intermediate_results)
                        self.assertIsInstance(intermediate_results[key], torch.Tensor)
                        self.assertEqual(intermediate_results[key].shape, (self.batch_size, self.n_heads, self.seq_len, self.seq_len))

    def test_decoder_block(self):
        for pe_type in self.positional_encoding_types:
            for causal in [False, True]:
                decoder_block = DecoderBlock(
                    **self.decoder_common_kwargs,
                    pos_enc_model_sa=pe_type,
                    pos_enc_model_ca=pe_type,
                    causal=causal
                )

                with self.subTest(positional_encoding=pe_type, causal=causal):
                    self.assertIsNotNone(decoder_block)
                    x = torch.randn(self.batch_size, self.seq_len, self.d_model)
                    memory = torch.randn(self.batch_size, self.seq_len, self.d_model)
                    output = decoder_block(x, memory)
                    self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

                # check that need_intermediate=True works
                with self.subTest(positional_encoding=pe_type, causal=causal, need_intermediate=True):
                    x, intermediate_results = decoder_block(x, memory, need_intermediate=True)
                    self.assertEqual(x.shape, (self.batch_size, self.seq_len, self.d_model))
                    self.assertIsInstance(intermediate_results, dict)
                    for key in ['self_attn_scores', 'cross_attn_scores']:
                        self.assertIn(key, intermediate_results)
                        self.assertIsInstance(intermediate_results[key], torch.Tensor)
                        self.assertEqual(intermediate_results[key].shape, (self.batch_size, self.n_heads, x.size(1), memory.size(1)))

if __name__ == '__main__':
    unittest.main()
