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
        self.rand = np.random.RandomState(1)

    @patch('ssg.np.random.uniform')
    def test_gameDefaultAttributes(self, mocked_uniform):
        mocked_uniform.side_effect = self.rand.uniform
        game, defenderRewards, defenderPenalties = createRandomGame(targets=5, resources=2, timesteps=3)
        self.assertAlmostEqual(5, game.numTargets)
        self.assertAlmostEqual(2, game.numResources)
        self.assertAlmostEqual(3, game.timesteps)
        #
        expectedRewards = np.array([21.43407823, 36.29590018, 1.00560437, 15.814296, 8.191039])
        expectedPenalties = np.array([5.524591, 10.12675 , 17.932476, 20.441606, 27.40202])
        self.assertTrue(mocked_uniform.called)
        npt.assert_array_almost_equal(defenderRewards,expectedRewards)
        npt.assert_array_almost_equal(defenderPenalties,expectedPenalties)


class TestSequentialZeroSumSSG(unittest.TestCase):
    # @patch('ssg.np.random.uniform')
    def setUp(self):
        # self.rand = np.random.RandomState(1)
        # mocked_uniform.side_effect = self.rand.uniform
        self.expectedRewards = np.array([21.43407823, 36.29590018, 1.00560437, 15.814296, 8.191039])
        self.expectedPenalties = np.array([5.524591, 10.12675 , 17.932476, 20.441606, 27.40202])
        self.expectedResources = 2
        self.expectedTargets = 5
        self.expectedTimesteps = 3
        self.game = SequentialZeroSumSSG(self.expectedTargets, self.expectedResources, self.expectedRewards, self.expectedPenalties, self.expectedTimesteps)

    def test_init(self):
        self.assertAlmostEqual(self.game.numTargets, self.expectedTargets)
        self.assertAlmostEqual(self.game.numResources, self.expectedResources)
        npt.assert_array_almost_equal(self.game.defenderRewards, self.expectedRewards)
        npt.assert_array_almost_equal(self.game.defenderPenalties, self.expectedPenalties)
        self.assertAlmostEqual(self.game.timesteps, self.expectedTimesteps)
        self.assertAlmostEqual(self.game.defenderUtility, 0)

    def test_restartGame(self):
        # Do some stuff
        self.game.currentTimestep = 4000
        self.game.targets = [0] * self.game.numTargets
        self.game.availableResources = -120
        self.game.previousAttackerAction = np.array([43] * self.game.numTargets)
        self.game.previousDefenderAction = np.array([74] * self.game.numTargets)
        self.game.pastAttacks = [109090] * self.game.numTargets
        self.game.pastAttackStatuses = [123] * self.game.numTargets
        self.game.defenderUtility = -9999999
        # Reset
        self.game.restartGame()
        # Check that everything reset properly
        self.assertAlmostEqual(self.game.currentTimestep, 0)
        self.assertListEqual(self.game.targets, [1, 1, 1, 1, 1])
        self.assertAlmostEqual(self.game.availableResources, 2)
        npt.assert_array_almost_equal(self.game.previousAttackerAction, np.array([0, 0, 0, 0, 0]))
        npt.assert_array_almost_equal(self.game.previousDefenderAction, np.array([0, 0, 0, 0, 0]))
        npt.assert_array_almost_equal(self.game.previousAttackerObservation, np.array([0, 0, 0, 0, 0] * 5))
        npt.assert_array_almost_equal(self.game.previousDefenderObservation, np.array([0, 0, 0, 0, 0] * 5))
        self.assertListEqual(self.game.pastAttacks, [0, 0, 0, 0, 0])
        self.assertListEqual(self.game.pastAttackStatuses, [0, 0, 0, 0, 0])
        self.assertAlmostEqual(self.game.defenderUtility, 0)

    def test_place_ones(self):
        expectedOnes = [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0, 1], [1, 1, 1, 0, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 1, 0, 0, 1, 0, 0], [1, 1, 1, 0, 1, 0, 0, 0, 1, 0], [1, 1, 1, 0, 1, 0, 0, 0, 0, 1], [1, 1, 1, 0, 0, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 1, 0, 1, 0, 0], [1, 1, 1, 0, 0, 1, 0, 0, 1, 0], [1, 1, 1, 0, 0, 1, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 1, 0, 1, 0], [1, 1, 1, 0, 0, 0, 1, 0, 0, 1], [1, 1, 1, 0, 0, 0, 0, 1, 1, 0], [1, 1, 1, 0, 0, 0, 0, 1, 0, 1], [1, 1, 1, 0, 0, 0, 0, 0, 1, 1], [1, 1, 0, 1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0, 1, 0, 0], [1, 1, 0, 1, 1, 0, 0, 0, 1, 0], [1, 1, 0, 1, 1, 0, 0, 0, 0, 1], [1, 1, 0, 1, 0, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 1, 0, 1, 0, 0], [1, 1, 0, 1, 0, 1, 0, 0, 1, 0], [1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1, 0, 1, 0], [1, 1, 0, 1, 0, 0, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0, 0, 1, 1, 0], [1, 1, 0, 1, 0, 0, 0, 1, 0, 1], [1, 1, 0, 1, 0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0, 1, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 1, 0, 1, 1, 0, 0], [1, 1, 0, 0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0, 0, 1], [1, 1, 0, 0, 1, 0, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 0, 1], [1, 1, 0, 0, 1, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 0, 1, 0, 1, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1, 0, 1], [1, 1, 0, 0, 0, 1, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0, 1, 1, 0, 1], [1, 1, 0, 0, 0, 0, 1, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 1, 1, 1, 0, 0, 0, 0], [1, 0, 1, 1, 1, 0, 1, 0, 0, 0], [1, 0, 1, 1, 1, 0, 0, 1, 0, 0], [1, 0, 1, 1, 1, 0, 0, 0, 1, 0], [1, 0, 1, 1, 1, 0, 0, 0, 0, 1], [1, 0, 1, 1, 0, 1, 1, 0, 0, 0], [1, 0, 1, 1, 0, 1, 0, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 1, 0, 0, 0, 1], [1, 0, 1, 1, 0, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 0, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1, 0, 0, 1], [1, 0, 1, 1, 0, 0, 0, 1, 1, 0], [1, 0, 1, 1, 0, 0, 0, 1, 0, 1], [1, 0, 1, 1, 0, 0, 0, 0, 1, 1], [1, 0, 1, 0, 1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 1, 0, 1, 0, 0], [1, 0, 1, 0, 1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 1, 0, 0, 0, 1], [1, 0, 1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 0, 0, 1, 1], [1, 0, 1, 0, 0, 1, 1, 1, 0, 0], [1, 0, 1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 0, 1, 0, 0, 1, 1], [1, 0, 1, 0, 0, 0, 1, 1, 1, 0], [1, 0, 1, 0, 0, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 0, 1, 0, 1, 1], [1, 0, 1, 0, 0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 1, 1, 1, 0, 0, 0], [1, 0, 0, 1, 1, 1, 0, 1, 0, 0], [1, 0, 0, 1, 1, 1, 0, 0, 1, 0], [1, 0, 0, 1, 1, 1, 0, 0, 0, 1], [1, 0, 0, 1, 1, 0, 1, 1, 0, 0], [1, 0, 0, 1, 1, 0, 1, 0, 1, 0], [1, 0, 0, 1, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0, 0, 0, 1, 1], [1, 0, 0, 1, 0, 1, 1, 1, 0, 0], [1, 0, 0, 1, 0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0, 1, 1], [1, 0, 0, 1, 0, 0, 1, 1, 1, 0], [1, 0, 0, 1, 0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 0, 1, 0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 1, 1, 1, 1, 0, 0], [1, 0, 0, 0, 1, 1, 1, 0, 1, 0], [1, 0, 0, 0, 1, 1, 1, 0, 0, 1], [1, 0, 0, 0, 1, 1, 0, 1, 1, 0], [1, 0, 0, 0, 1, 1, 0, 1, 0, 1], [1, 0, 0, 0, 1, 1, 0, 0, 1, 1], [1, 0, 0, 0, 1, 0, 1, 1, 1, 0], [1, 0, 0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 0, 0, 1, 0, 1, 0, 1, 1], [1, 0, 0, 0, 1, 0, 0, 1, 1, 1], [1, 0, 0, 0, 0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 1, 1, 1, 0, 1], [1, 0, 0, 0, 0, 1, 1, 0, 1, 1], [1, 0, 0, 0, 0, 1, 0, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0, 1, 0, 0], [0, 1, 1, 1, 1, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 0, 0, 0, 0, 1], [0, 1, 1, 1, 0, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 1, 0, 1, 0, 0], [0, 1, 1, 1, 0, 1, 0, 0, 1, 0], [0, 1, 1, 1, 0, 1, 0, 0, 0, 1], [0, 1, 1, 1, 0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0, 0, 1, 1, 0], [0, 1, 1, 1, 0, 0, 0, 1, 0, 1], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1], [0, 1, 1, 0, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 1, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 1, 0, 0, 0, 1], [0, 1, 1, 0, 1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 1, 0, 0, 0, 1, 1], [0, 1, 1, 0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 0, 0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 0, 0, 1, 1], [0, 1, 1, 0, 0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 0, 0, 1, 1, 0, 1], [0, 1, 1, 0, 0, 0, 1, 0, 1, 1], [0, 1, 1, 0, 0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 1, 1, 1, 0, 0, 0], [0, 1, 0, 1, 1, 1, 0, 1, 0, 0], [0, 1, 0, 1, 1, 1, 0, 0, 1, 0], [0, 1, 0, 1, 1, 1, 0, 0, 0, 1], [0, 1, 0, 1, 1, 0, 1, 1, 0, 0], [0, 1, 0, 1, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 1, 0, 0, 1, 0, 1], [0, 1, 0, 1, 1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 1, 1, 0, 0], [0, 1, 0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 1, 0, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 0, 1, 1, 1, 0], [0, 1, 0, 1, 0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 0, 0, 0, 1, 1, 1], [0, 1, 0, 0, 1, 1, 1, 1, 0, 0], [0, 1, 0, 0, 1, 1, 1, 0, 1, 0], [0, 1, 0, 0, 1, 1, 1, 0, 0, 1], [0, 1, 0, 0, 1, 1, 0, 1, 1, 0], [0, 1, 0, 0, 1, 1, 0, 1, 0, 1], [0, 1, 0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 0, 1, 0, 1, 1, 1, 0], [0, 1, 0, 0, 1, 0, 1, 1, 0, 1], [0, 1, 0, 0, 1, 0, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0, 1, 1, 1], [0, 1, 0, 0, 0, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 1, 1, 1, 0, 1], [0, 1, 0, 0, 0, 1, 1, 0, 1, 1], [0, 1, 0, 0, 0, 1, 0, 1, 1, 1], [0, 1, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 1, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0, 1, 0], [0, 0, 1, 1, 1, 1, 0, 0, 0, 1], [0, 0, 1, 1, 1, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 1, 0, 1, 0], [0, 0, 1, 1, 1, 0, 1, 0, 0, 1], [0, 0, 1, 1, 1, 0, 0, 1, 1, 0], [0, 0, 1, 1, 1, 0, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 0, 1, 1, 0, 1, 0], [0, 0, 1, 1, 0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 0, 1, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0, 1, 0, 1, 1], [0, 0, 1, 1, 0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1, 1, 1, 0, 0], [0, 0, 1, 0, 1, 1, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1, 1, 0, 0, 1], [0, 0, 1, 0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 0, 1, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1, 0, 0, 1, 1], [0, 0, 1, 0, 1, 0, 1, 1, 1, 0], [0, 0, 1, 0, 1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 0, 0, 1, 1, 1], [0, 0, 1, 0, 0, 1, 1, 1, 1, 0], [0, 0, 1, 0, 0, 1, 1, 1, 0, 1], [0, 0, 1, 0, 0, 1, 1, 0, 1, 1], [0, 0, 1, 0, 0, 1, 0, 1, 1, 1], [0, 0, 1, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1, 1, 0, 0, 1], [0, 0, 0, 1, 1, 1, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1, 0, 1, 0, 1], [0, 0, 0, 1, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 0, 1, 1, 0, 1], [0, 0, 0, 1, 1, 0, 1, 0, 1, 1], [0, 0, 0, 1, 1, 0, 0, 1, 1, 1], [0, 0, 0, 1, 0, 1, 1, 1, 1, 0], [0, 0, 0, 1, 0, 1, 1, 1, 0, 1], [0, 0, 0, 1, 0, 1, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1, 0, 1, 1, 1], [0, 0, 0, 1, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0, 1], [0, 0, 0, 0, 1, 1, 1, 0, 1, 1], [0, 0, 0, 0, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 1, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
        expectedOnesSecond = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        ones = list(self.game.place_ones(10, 5))
        ones_second = list(self.game.place_ones(3,1))
        # print(ones_second)
        self.assertListEqual(ones, expectedOnes)
        self.assertListEqual(ones_second, expectedOnesSecond)

    def test_getValidActions(self):
        # Test defender actions
        # Test attacker actions
        pass

    def test_performActions(self):
        pass

    def test_getEmptyObservation(self):
        expectedDefenderObservation = np.concatenate(([0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], self.expectedRewards, self.expectedPenalties))
        expectedAttackerObservation = np.concatenate(([0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], self.expectedRewards, self.expectedPenalties))
        dOb, aOb = self.game.getEmptyObservations()
        npt.assert_array_almost_equal(dOb, expectedDefenderObservation)
        npt.assert_array_almost_equal(aOb, expectedAttackerObservation)

    def test_getActionScore(self):
        pass

    def test_getPayout(self):
        pass

    def test_getOracleScore(self):
        pass

    def test_getBestOracle(self):
        pass
