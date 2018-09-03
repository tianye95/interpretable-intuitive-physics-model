import os


class Paths(object):
    def __init__(self):
        self.parent = os.path.dirname(os.path.abspath(__file__))
        self.data_root, self.save_root = os.path.join(self.parent, '..'), os.path.join(self.parent, '..')
        self.data_path = os.path.join(self.data_root, 'data')
        self.info_path = os.path.join(self.data_root, 'data', 'info')
        self.model_path = os.path.join(self.save_root, 'checkpoints')
        self.visualization_path = os.path.join(self.save_root, 'visualization')
