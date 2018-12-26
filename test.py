import unittest
import os
import shutil

import environment
import utils
import random_agent
import numpy as np

class EnvironmentTestCase(unittest.TestCase):

    def setUp(self):
        if not os.path.exists('testlogs'):
            os.mkdir('testlogs')
        self.agent = random_agent.SonicRandomAgent('GreenHillZone.Act1', 'testlogs/')

    def test_observation_space_is_frame_stack(self):
        self.agent.env.reset()
        obs, *_ = self.agent.env.step(self.agent.env.action_space.sample())
        self.assertIsInstance(obs, np.ndarray)
        self.assertTrue(obs.shape[0] == obs.shape[1])
        self.assertTrue(obs.ndim == 3)
    
    def test_run(self):
        self.agent.run(1, max_steps=10)
    
    def tearDown(self):
        if os.path.exists('testlogs'):
            shutil.rmtree('testlogs',ignore_errors=True)

class FrameStackTestCase(unittest.TestCase):

    def setUp(self):
        self.frames = utils.FrameStack(4, np.zeros((2,2)))

    def test_frame_stack(self):
        self.assertEqual(self.frames.shape, (2,2,4))
        self.frames.append( np.ones((2,2)) )
        self.assertEqual(self.frames.shape, ((2,2,4)))
        self.assertTrue( np.array_equal(self.frames[0], np.ones((2,2))) )
        self.frames.append( np.zeros((2,2)) )
        self.assertTrue( np.array_equal(self.frames[0], np.zeros((2,2))) )
        self.assertTrue( np.array_equal(self.frames[1], np.ones((2,2))) )
    
    def test_overflow(self):
        for i in range(5):
             self.frames.append( np.ones((2,2)) )
        self.assertTrue(np.array_equal(self.frames[0], np.ones((2,2))))
        self.assertEqual(self.frames.shape, (2,2,4))

    def test_get_set(self):
        y = np.zeros((2,2))
        self.frames[0] = y
        x = self.frames[0]
        self.assertTrue( np.array_equal(x, y))

class EpisodeLogTestCase(unittest.TestCase):
    def setUp(self):
        if not os.path.exists('testlogs/'):
            os.mkdir('testlogs/')
        self.agent = random_agent.SonicRandomAgent('GreenHillZone.Act1','testlogs/')
        self.agent.run(10)

    def test_logs(self):
        df = self.agent.log.make_dataframe()

    def tearDown(self):
        if os.path.exists('testlogs'):
            shutil.rmtree('testlogs', ignore_errors=True)



