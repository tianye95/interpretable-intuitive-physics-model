import os


class Paths(object):
    def __init__(self):
        self.data_root, self.save_root = '..', '..'
        self.data_path = os.path.join(self.data_root, 'data')
        self.info_path = os.path.join(self.data_root, 'info')
        self.model_path = os.path.join(self.save_root, 'checkpoints')
        self.visualization_path = os.path.join(self.save_root, 'visualization')
