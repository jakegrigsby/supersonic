import unittest
import os
import shutil

import numpy as np

from supersonic import environment, utils, random_agent, camera, agent, paramsearch

class EnvironmentTestCase(unittest.TestCase):
    
    def setUp(self):
        self.agent = random_agent.SonicRandomAgent('GreenHillZone.Act1')

    def test_observation_space_is_frame_stack(self):
        self.agent.env.reset()
        obs, rew, done, info = self.agent.env.step(self.agent.env.action_space.sample())
        self.assertIsInstance(obs, np.ndarray)
        self.assertTrue(obs.ndim == 4)
    
    def test_obs_is_float32(self):
        obs = self.agent.env.reset()
        self.assertEqual(obs.dtype, np.float32)
    
    def test_env_reset(self):
        self.assertTrue(len(self.agent.env.reset()), 84)

    def test_run(self):
        self.agent.run(1, max_steps=10)

    def tearDown(self):
        self.agent = None

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

class CameraTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists('logs/testvids'):
            os.mkdir('logs/testvids')

    def setUp(self):
        self.x = random_agent.SonicRandomAgent('GreenHillZone.Act1')
        self.cam = camera.Camera(self.x, True, 5)

    def test_vanilla_record(self):
        self.cam.record('logs/testvids/testvideo.mp4')
        self.x.run(1, max_steps=5)
        self.cam.stop_recording()
        self.assertTrue(os.path.exists('logs/testvids/testvideo.mp4'))

    def test_record_that(self):
        self.x.run(1, 5)
        self.cam.record_that('logs/testvids/testrecordthat.mp4')
        self.assertTrue(os.path.exists('logs/testvids/testrecordthat.mp4'))

    def tearDown(self):
       del self.x
       del self.cam

    @classmethod
    def tearDownClass(cls):
        if os.path.exists('logs/testvids'):
            shutil.rmtree('logs/testvids', ignore_errors=True)

class AgentTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_log_path = os.path.join('logs','tests')
        if not os.path.exists(test_log_path):
            os.makedirs(test_log_path)

    def test_trains(self):
        test_agent = agent.BaseAgent('GreenHillZone.Act1', log_dir='tests/testlog', rollout_length=32, exp_train_prop=1.)
        test_agent.train(1)

    def test_plays(self):
        test_agent = agent.BaseAgent('GreenHillZone.Act1', log_dir='tests/testlog')
        rew = test_agent.test(1, max_ep_steps=10)
    
    def test_save_load_weights(self):
        test_agent = agent.BaseAgent('GreenHillZone.Act1', log_dir='tests/testlog')
        test_agent.save_weights('model_zoo/testweights')
        test_agent.load_weights('model_zoo/testweights')
        shutil.rmtree('model_zoo/testweights')

    @classmethod
    def tearDownClass(cls):
        test_log_dir = 'logs/tests'
        if os.path.exists(test_log_dir):
            shutil.rmtree(test_log_dir)

class JsonHyptDictTestCase(unittest.TestCase):

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

class LvlMapsTestCase(unittest.TestCase):

    def test_load_all_maps(self):
        for lvl_id in utils.all_sonic_lvls().keys():
            self.assertIsInstance(utils.get_lvl_map(lvl_id), np.ndarray)

class SearchSpaceTestCase(unittest.TestCase):

    def test_DiscreteSpace(self):
        space = paramsearch.DiscreteSearchSpace(np.arange(3))
        self.assertEqual(len(space.space),3)
        self.assertTrue(space.probs[0] == space.probs[1])
        self.assertAlmostEqual(np.sum(space.probs), 1.)
        space.update(0, .17)
        self.assertAlmostEqual(space.probs[0], .5, 2)
        self.assertTrue(space.probs[1] == space.probs[2])
        self.assertAlmostEqual(np.sum(space.probs), 1.)
        point = space.sample(1)
        self.assertTrue(point in [0,1,2])
        #test to make sure discrete spaces initialized w lists do not nest that list
        space = paramsearch.DiscreteSearchSpace([0,1,2])
        self.assertEqual(space.space, [0,1,2])

    def test_PowerofNSearchSpace(self):
        space = paramsearch.PowerofNSearchSpace(2, 0, 3)
        self.assertEqual(len(space.space),3)
        self.assertTrue(space.probs[0] == space.probs[1])
        self.assertAlmostEqual(np.sum(space.probs), 1.)
        space.update(1, .17)
        self.assertAlmostEqual(space.probs[0], .5, 2)
        self.assertTrue(space.probs[1] == space.probs[2])
        self.assertAlmostEqual(np.sum(space.probs), 1.)
        point = space.sample(1)
        self.assertTrue(point in [2**0, 2**1, 2**2])
    
    def test_ContinuousSpace(self):
        space = paramsearch.ContinuousSearchSpace(0,1)
        point = space.sample(1)
        self.assertTrue(point >= 0 and point < 1)
        space.update(.25, .25)
 
    def test_BucketizedSpace(self):
        space = paramsearch.ContinuousSearchSpace(0,1)
        space = paramsearch.bucketize_space(space, 10)
        self.assertEqual(len(space.space),10)
        self.assertTrue(space.probs[0] == space.probs[1])
        self.assertAlmostEqual(np.sum(space.probs), 1.)
        space.update(0., .1)
        self.assertAlmostEqual(space.probs[0], .2, 2)
        self.assertTrue(space.probs[1] == space.probs[2])
        self.assertAlmostEqual(np.sum(space.probs), 1.)
        point = space.sample(1)
        self.assertTrue(np.all(np.isin(point, np.linspace(0,1,10))))
    
