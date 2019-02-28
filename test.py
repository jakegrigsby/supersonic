import unittest
import os
import shutil

import numpy as np

import environment
import utils
import random_agent
import camera
import agent

class EnvironmentTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists('testlogs'):
            os.mkdir('testlogs')
    
    def setUp(self):
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
        self.agent = None

    @classmethod 
    def tearDownClass(cls):
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
    @classmethod
    def setUpClass(cls):
        if not os.path.exists('testlogs'):
            os.mkdir('testlogs')

    def setUp(self):
        self.agent = random_agent.SonicRandomAgent('GreenHillZone.Act1','testlogs/')
        self.agent.run(10)

    def test_logs(self):
        df = self.agent.log.make_dataframe()
    
    def tearDown(self):
        del self.agent

    @classmethod
    def tearDownClass(cls):
        if os.path.exists('testlogs'):
            shutil.rmtree('testlogs', ignore_errors=True)

class CameraTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists('testvids'):
            os.mkdir('testvids')
        if not os.path.exists('testlogs'):
            os.mkdir('testlogs')

    def setUp(self):
        self.x = random_agent.SonicRandomAgent('GreenHillZone.Act1', 'testlogs')
        self.cam = camera.Camera(self.x, True, 5)
    
    def test_vanilla_record(self):
        self.cam.record('testvids/testvideo.mp4')
        self.x.run(1, max_steps=5)
        self.cam.stop_recording()
        self.assertTrue(os.path.exists('testvids/testvideo.mp4'))
    
    def test_record_that(self):
        self.x.run(1, 5)
        self.cam.record_that('testvids/testrecordthat.mp4')
        self.assertTrue(os.path.exists('testvids/testrecordthat.mp4'))

    def tearDown(self):
       del self.x
       del self.cam

    @classmethod
    def tearDownClass(cls):
        if os.path.exists('testvids'):
            shutil.rmtree('testvids', ignore_errors=True)
        if os.path.exists('testlogs'):
            shutil.rmtree('testlogs', ignore_errors=True)

class AgentTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        self.agent = agent.BaseAgent('GreenHillZone.Act1', logdir='logs', name='tester')
        if not os.path.exists('logs'):
            os.mkdir('logs')
    
    def test_trains(self):
        init_weights = self.agent.weights
        self.agent.train(1)
        final_weights = self.agent.weights
        self.assertNotEqual(init_weights, final_weights) 
    
    def test_plays(self):
        rew = self.agent.test(1)
        self.assertTrue(rew > 0)
    
    def test_logs_recorded(self):
        self.assertTrue(os.path.exists('logs/tester.txt'))
    
    @classmethod
    def tearDownClass(cls):
        os.remove('logs/tester.txt')
 
class TestJsonHypDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists('data/hyp_dicts'):
            os.mkdir('data/hyp_dicts')

    def test_hyp_dict_to_file(self):
        utils.save_hyp_dict_to_file('data/hyp_dicts/test_dict.json', {'test_param':1})
        self.assertTrue(os.path.exists('data/hyp_dicts/test_dict.json'))
        os.remove('data/hyp_dicts/test_dict.json')
    
    def test_hyp_dict_from_file(self):
        utils.save_hyp_dict_to_file('data/hyp_dicts/test_dict.json', {'test_param':1})
        test_dict = agent.load_hyp_dict_from_file('data/hyp_dicts/test_dict.json')
        self.assertTrue(test_dict == {'test_param':1})
    
    @classmethod
    def tearDownClass(cls):
        if os.path.exists('data/hyp_dicts/test_dict.json'):
            os.remove('data/hyp_dicts/test_dict.json')