



class Trainable(object):

    def _train(self):
        pass
    def save_checkpoint(self, save_path = None):
        pass
    def load_checkpoint(self, load_path):
        pass
    def getMetric(self):
        pass
    def _stop(self):
        del self