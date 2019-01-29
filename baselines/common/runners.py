import numpy as np
from abc import ABC, abstractmethod

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + tuple(s if s is not None else 0 for s in
                                            env.observation_space.shape),
                            dtype=env.observation_space.dtype.name)
        obs = env.reset()
        if np.size(self.obs) == np.size(obs):
            self.obs[:] = obs
        else:
            self.obs = obs
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError
