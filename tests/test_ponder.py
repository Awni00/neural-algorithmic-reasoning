import unittest
import torch
from models.transformer_blocks import EncoderBlock
from models.ponder import PonderNetworkWrapper

class TestPonderNetworkWrapper(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 5
        self.d_model = 16
        self.n_heads = 2
        self.n_classes = 10
        self.labels = torch.randint(0, self.n_classes, (self.batch_size, self.seq_len))

    def test_forward_train_tokenwise(self):
        block = EncoderBlock(d_model=self.d_model, n_heads=self.n_heads, causal=True)
        model = PonderNetworkWrapper(block, n_classes=self.n_classes, halt_sequencewise=False)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        model.train()
        loss, intermediate_outputs = model(x, labels=self.labels)

        self.assertEqual(intermediate_outputs['halting_probs'].sum(dim=1).allclose(torch.ones(self.batch_size, self.seq_len)), True)
        self.assertEqual(intermediate_outputs['hiddens'].shape, (self.batch_size, model.train_max_steps, self.seq_len, self.d_model))
        self.assertEqual(intermediate_outputs['logits'].shape, (self.batch_size, model.train_max_steps, self.seq_len, self.n_classes))
        self.assertEqual(intermediate_outputs['halting_conditional_probs'].shape, (self.batch_size, model.train_max_steps, self.seq_len))
        self.assertEqual(intermediate_outputs['halting_probs'].shape, (self.batch_size, model.train_max_steps, self.seq_len))

        loss.backward()
        for param in model.parameters():
            self.assertIsNotNone(param.grad)

    def test_forward_train_sequencewise(self):
        block = EncoderBlock(d_model=self.d_model, n_heads=self.n_heads, causal=False)
        model = PonderNetworkWrapper(block, n_classes=self.n_classes, halt_sequencewise=True)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        model.train()
        loss, intermediate_outputs = model(x, labels=self.labels)

        self.assertEqual(intermediate_outputs['halting_probs'].sum(dim=1).allclose(torch.ones(self.batch_size)), True)
        self.assertEqual(intermediate_outputs['hiddens'].shape, (self.batch_size, model.train_max_steps, self.seq_len, self.d_model))
        self.assertEqual(intermediate_outputs['logits'].shape, (self.batch_size, model.train_max_steps, self.seq_len, self.n_classes))
        self.assertEqual(intermediate_outputs['halting_conditional_probs'].shape, (self.batch_size, model.train_max_steps))
        self.assertEqual(intermediate_outputs['halting_probs'].shape, (self.batch_size, model.train_max_steps, 1))

        loss.backward()
        for param in model.parameters():
            self.assertIsNotNone(param.grad)

    def test_forward_inference_tokenwise(self):
        block = EncoderBlock(d_model=self.d_model, n_heads=self.n_heads, causal=True)
        model = PonderNetworkWrapper(block, n_classes=self.n_classes, halt_sequencewise=False)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        model.eval()
        logits, outputs = model(x)

        n_steps = outputs['steps_halt'].shape[1]

        self.assertEqual(outputs['steps_hidden_states'].shape, (self.batch_size, n_steps, self.seq_len, self.d_model))
        self.assertEqual(outputs['steps_halt_logits'][0].shape, (self.batch_size, self.seq_len))
        self.assertEqual(outputs['steps_halt'].shape, (self.batch_size, n_steps, self.seq_len))
        self.assertEqual(outputs['halt_times'].shape, (self.batch_size, self.seq_len))
        self.assertEqual(outputs['halt_time_hidden_states'].shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertEqual(outputs['logits'].shape, (self.batch_size, self.seq_len, self.n_classes))

    def test_forward_inference_sequencewise(self):
        block = EncoderBlock(d_model=self.d_model, n_heads=self.n_heads, causal=False)
        model = PonderNetworkWrapper(block, n_classes=self.n_classes, halt_sequencewise=True)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        model.eval()
        logits, outputs = model(x)

        n_steps = outputs['steps_halt'].shape[1]

        self.assertEqual(outputs['steps_hidden_states'].shape, (self.batch_size, n_steps, self.seq_len, self.d_model))
        self.assertEqual(outputs['steps_halt_logits'][0].shape, (self.batch_size,))
        self.assertEqual(outputs['steps_halt'].shape, (self.batch_size, n_steps))
        self.assertEqual(outputs['halt_times'].shape, (self.batch_size,))
        self.assertEqual(outputs['halt_time_hidden_states'].shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertEqual(outputs['logits'].shape, (self.batch_size, self.seq_len, self.n_classes))

if __name__ == '__main__':
    unittest.main()
