import gc
import unittest
from importlib.metadata import version as distribution_version

import numpy as np
from snake_drl import AutoresetMode, ObsMode, VecConfig, VecSnake, __version__
from snake_drl.play.game import DOWN, LEFT, RIGHT, UP, GameSession


def make_config(obs=ObsMode.Bitboard, autoreset=AutoresetMode.SameStep, num_envs=2):
    cfg = VecConfig()
    cfg.num_envs = num_envs
    cfg.width = 8
    cfg.height = 8
    cfg.obs_mode = obs
    cfg.autoreset_mode = autoreset
    cfg.num_threads = 1
    cfg.seed = 42
    return cfg


class BindingTests(unittest.TestCase):
    def test_version(self):
        self.assertTrue(__version__)
        self.assertEqual(__version__, distribution_version("snake-drl"))

    def test_view_modes_and_final_policy(self):
        for obs in (ObsMode.NoObs, ObsMode.Grid, ObsMode.Bitboard):
            for autoreset in (
                AutoresetMode.SameStep,
                AutoresetMode.NextStep,
                AutoresetMode.Disabled,
            ):
                with self.subTest(obs=obs, autoreset=autoreset):
                    env = VecSnake(make_config(obs, autoreset))
                    self.assertEqual(env.bitboards().shape, (2, 8))

                    if obs == ObsMode.Grid:
                        self.assertEqual(env.observations().shape, (2, 3, 8, 8))
                    else:
                        with self.assertRaises(RuntimeError):
                            env.observations()

                    if obs == ObsMode.Grid and autoreset == AutoresetMode.SameStep:
                        self.assertEqual(env.final_observations().shape, (2, 3, 8, 8))
                    else:
                        self.assertIsNone(env.final_observations())

                    if autoreset == AutoresetMode.SameStep:
                        self.assertEqual(env.final_bitboards().shape, (2, 8))
                        self.assertEqual(env.final_head_cells().shape, (2,))
                        self.assertEqual(env.final_food_cells().shape, (2,))
                    else:
                        self.assertIsNone(env.final_bitboards())
                        self.assertIsNone(env.final_head_cells())
                        self.assertIsNone(env.final_food_cells())

                    # Terminal metadata exists in every mode.
                    self.assertEqual(env.final_scores().shape, (2,))
                    self.assertEqual(env.final_lengths().shape, (2,))
                    self.assertEqual(env.final_returns().shape, (2,))

    def test_views_are_read_only_zero_copy_and_lifetime_safe(self):
        env = VecSnake(make_config(obs=ObsMode.Grid))
        first = env.observations()
        second = env.observations()
        self.assertFalse(first.flags.writeable)
        self.assertTrue(np.shares_memory(first, second))
        with self.assertRaises(ValueError):
            first[0, 0, 0, 0] = 1

        del env
        gc.collect()
        self.assertEqual(first.shape, (2, 3, 8, 8))
        self.assertGreaterEqual(int(first.sum()), 0)

    def test_config_is_a_snapshot(self):
        env = VecSnake(make_config())
        config = env.config
        config.num_envs = 999
        self.assertEqual(env.num_envs, 2)

    def test_step_rejects_wrong_size(self):
        env = VecSnake(make_config(num_envs=3))
        with self.assertRaises(ValueError):
            env.step(np.zeros(2, dtype=np.int8))
        env.step(np.zeros(3, dtype=np.int8))

    def test_reseed_matches_a_fresh_instance(self):
        cfg = make_config(num_envs=4)
        cfg.obs_mode = ObsMode.Bitboard
        fresh = VecSnake(cfg)
        env = VecSnake(cfg)

        rng = np.random.default_rng(7)
        for _ in range(37):
            env.step(rng.integers(0, 3, size=4, dtype=np.int8))

        env.reseed_all(cfg.seed)
        np.testing.assert_array_equal(env.food_cells(), fresh.food_cells())
        np.testing.assert_array_equal(env.head_cells(), fresh.head_cells())
        np.testing.assert_array_equal(env.bitboards(), fresh.bitboards())

        env.reseed_all(cfg.seed)  # idempotent
        np.testing.assert_array_equal(env.food_cells(), fresh.food_cells())

        # A reseeded batch must not collapse onto a single food cell.
        batch = VecSnake(make_config(num_envs=16))
        batch.reseed_all(cfg.seed)
        self.assertGreater(len(set(batch.food_cells().tolist())), 8)


class PlayTests(unittest.TestCase):
    def test_turn_is_validated_against_engine_heading(self):
        session = GameSession(16, 16, seed=3)
        self.assertEqual(session.direction, RIGHT)

        self.assertTrue(session.turn(UP))
        # LEFT is a reversal relative to the engine's actual heading (RIGHT),
        # even though it is not opposite to the queued UP.
        self.assertFalse(session.turn(LEFT))
        session.step()
        self.assertEqual(session.direction, UP)
        self.assertEqual(int(session.env.directions()[0]), UP)

    def test_queued_turn_is_consumed_by_one_tick(self):
        session = GameSession(16, 16, seed=3)
        self.assertTrue(session.turn(DOWN))
        session.step()
        self.assertEqual(session.direction, DOWN)
        session.step()  # continues down, no stale pending direction
        self.assertEqual(session.direction, DOWN)

    def test_terminal_state_is_sticky_until_reset(self):
        session = GameSession(16, 16, seed=5)
        for _ in range(20):
            session.step()
            if session.game_over:
                break
        self.assertTrue(session.game_over)
        steps, length, score = session.steps, session.length, session.score
        session.step()
        self.assertEqual((session.steps, session.length, session.score), (steps, length, score))
        session.reset()
        self.assertFalse(session.game_over)
        self.assertEqual(session.steps, 0)


if __name__ == "__main__":
    unittest.main()
