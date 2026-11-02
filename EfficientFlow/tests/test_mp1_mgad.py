import unittest

import torch

from model.diffusion.meanflow_conditional_unet1d import ConditionalUnet1D_new
from model.meanflow_mgad import MGADConditionalUnet1D, MemoryGuidedActionDecoder


class MP1MGADTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.options = dict(input_dim=3, global_cond_dim=10,
                            diffusion_step_embed_dim=8, down_dims=(8, 16),
                            kernel_size=3, n_groups=2, var_fc_hidden_dim=8)

    def inputs(self, batch_size=2):
        return (torch.randn(batch_size, 4, 3), torch.full((batch_size,), 0.7),
                torch.full((batch_size,), 0.2), torch.randn(batch_size, 10))

    def test_disabled_matches_baseline_with_same_weights(self):
        baseline = ConditionalUnet1D_new(**self.options)
        adapted = MGADConditionalUnet1D(**self.options, mgad=False)
        adapted.load_state_dict(baseline.state_dict(), strict=True)
        self.assertEqual(set(baseline.state_dict()), set(adapted.state_dict()))
        z, t, r, observation = self.inputs()
        original = baseline(z, t, global_cond=observation, r=r)
        current = adapted(z, t, global_cond=observation, r=r)
        for expected, actual in zip(original, current):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(
            baseline(z, t, global_cond=observation, r=r, training=False),
            adapted(z, t, global_cond=observation, r=r, training=False),
            rtol=0, atol=0)

    def test_feature_api_and_multiple_batch_sizes(self):
        model = MGADConditionalUnet1D(**self.options, horizon=4)
        for batch_size in (1, 3):
            z, t, r, observation = self.inputs(batch_size)
            velocity, dispersive, features = model.forward_with_features(
                z, t, global_cond=observation, r=r)
            self.assertEqual(velocity.shape, (batch_size, 4, 3))
            self.assertEqual(dispersive.shape, (1, batch_size, 32))
            self.assertEqual(features.shape, (batch_size, 4, 8))
            torch.testing.assert_close(velocity, model(
                z, t, global_cond=observation, r=r, training=False))

    def test_mgad_jvp_and_meanflow_backward(self):
        model = MGADConditionalUnet1D(**self.options, horizon=4)
        z, t, r, observation = self.inputs()
        direction = torch.randn_like(z)
        prediction, tangent = torch.autograd.functional.jvp(
            lambda state, time, endpoint: model(
                state, time, global_cond=observation, r=endpoint),
            (z, t, r), (direction, torch.ones_like(t), torch.zeros_like(r)),
            create_graph=True)
        target = direction - (t - r)[:, None, None] * tangent[0]
        loss = (prediction[0] - target.detach()).square().mean()
        # Differentiate through the JVP as well, covering double backward of
        # attention for the MeanFlow training machinery and future objectives.
        loss = loss + 0.01 * tangent[0].square().mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            if name.startswith("mgad"):
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(float(model.mgad.cross_attn.key.weight.grad.abs().sum()), 0)

    def test_observation_memory_and_both_interval_endpoints_matter(self):
        model = MGADConditionalUnet1D(**self.options, horizon=4)
        z, t, r, observation = self.inputs()
        te = model.diffusion_step_encoder(t)
        re = model.diffusion_step_encoder_rs(r)
        memory, context = model._memory(observation, te, re)
        self.assertEqual(memory.shape, (2, 3, 8))  # two observations + time
        action_features = torch.randn(2, 4, 8)
        reference = model.mgad(action_features, memory, context)
        changed = memory.clone()
        changed[:, 0, 0] += 3
        self.assertFalse(torch.allclose(reference, model.mgad(
            action_features, changed, context)))
        for new_t, new_r in ((t + 0.1, r), (t, r + 0.1)):
            new_memory, new_context = model._memory(
                observation, model.diffusion_step_encoder(new_t),
                model.diffusion_step_encoder_rs(new_r))
            self.assertFalse(torch.allclose(reference, model.mgad(
                action_features, new_memory, new_context)))

    def test_invalid_memory_configuration_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "whole observation"):
            MGADConditionalUnet1D(**self.options, mgad_num_obs_steps=3)
        with self.assertRaisesRegex(ValueError, "divisible"):
            MemoryGuidedActionDecoder(8, 4, num_heads=3)
        model = MGADConditionalUnet1D(**self.options)
        z, t, r, observation = self.inputs()
        with self.assertRaisesRegex(ValueError, "flattened"):
            model._memory(observation.reshape(2, 2, 5),
                          model.diffusion_step_encoder(t),
                          model.diffusion_step_encoder_rs(r))


if __name__ == "__main__":
    unittest.main()
