from pathlib import Path

class Parameters:
    def __init__(self):
        self.cwd = Path(__file__).absolute().parents[1]
        self.envs = 'pong'
        self.ncolumns = 1
        self.lr = 1e-4
        self.gamma = 0.99
        self.tau = 1.0
        self.beta = 1e-2
        self.critic_loss_coef = 0.5
        self.clip = 40
        self.nlsteps = 20
        self.ngsteps = 4e7
        self.nprocesses = 16
        self.interval = 500
        self.max_actions = 100
        self.discount = 0.95
        self.log_path = self.cwd / "tensorboard" / "pnn"
        self.save_path = self.cwd / "trained_models"
        self.load = False
        self.render = False
        self.seed = 1
