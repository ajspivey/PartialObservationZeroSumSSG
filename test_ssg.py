import unittest
from unittest.mock import Mock, patch
from ssg import generateRewardsAndPenalties, createRandomGame, SequentialZeroSumSSG

import numpy as np
import numpy.testing as npt


class TestGenerateRewardsAndPenalties(unittest.TestCase):
    def setUp(self):
        self.rand = np.random.RandomState(1)

    def test_size(self):
        rewards, penalties = generateRewardsAndPenalties(7)
        self.assertAlmostEqual(7, len(rewards))
        self.assertAlmostEqual(7, len(penalties))
        rewards, penalties = generateRewardsAndPenalties(3)
        self.assertAlmostEqual(3, len(rewards))
        self.assertAlmostEqual(3, len(penalties))

    @patch('ssg.np.random.uniform')
    def test_value(self, mocked_uniform):
        mocked_uniform.side_effect = self.rand.uniform
        expectedRewards = np.array([21.43407823, 36.29590018, 1.00560437])
        expectedPenalties = np.array([15.81429606, 8.19103865, 5.52459114])
        rewards, penalties = generateRewardsAndPenalties(3)
        self.assertTrue(mocked_uniform.called)
        npt.assert_array_almost_equal(rewards,expectedRewards)
        npt.assert_array_almost_equal(penalties,expectedPenalties)


class TestCreateRandomGame(unittest.TestCase):
    def setUp(self):
        self.rand = np.random
        self.rand.seed(1)

    def test_gameDefaultAttributes(self):
        game, defenderRewards, defenderPenalties = createRandomGame(targets=5, resources=2, timesteps=3)
        self.assertAlmostEqual(5, game.numTargets)
        self.assertAlmostEqual(5, game.numTargets)
        self.assertAlmostEqual(5, game.numTargets)
        self.assertAlmostEqual(5, game.numTargets)
        self.assertAlmostEqual(5, game.numTargets)
        self.assertAlmostEqual(5, game.numTargets)
