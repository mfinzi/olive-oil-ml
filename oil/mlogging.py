import pandas as pd

#TODO: Add pandas datafram logging capability
class SummaryWriter:
    """ Thin wrapper around tensorboardX summarywriter,
        non tensorboard logging functionality to come
    """ 
    def __init__(self, save_dir, log=True, com=5):
        if not log or save_dir is None:
            self.logging = False
            self.tbWriter = None
        else:
            self.logging = True
            import tensorboardX
            self.tbWriter = tensorboardX.SummaryWriter(save_dir)
        self.loggingFrame = pd.DataFrame()
        self.com = com

    def state_dict(self):
        return self.loggingFrame

    def load_state_dict(self, data):
        self.loggingFrame = data

    def __repr__(self):
        return self.loggingFrame.__repr__()
    def __str__(self):
        return self.loggingFrame.__str__()

    def emas(self):
        return self.loggingFrame.ewm(com=self.com).mean().iloc[-1]

    def add_scalar(self, *args, **kwargs):
        if self.logging:
            self.tbWriter.add_scalar(*args, **kwargs)

    def add_scalars(self, name,dic,step, *args, **kwargs):
        if self.logging:
            self.tbWriter.add_scalars(name,dic,step,*args, **kwargs)
        newRow = pd.DataFrame(dic, index = [step])
        self.loggingFrame = self.loggingFrame.append(newRow)
    def export_scalars_to_json(self, *args, **kwargs):
        if self.logging:
            self.tbWriter.export_scalars_to_json(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        if self.logging:
            self.tbWriter.add_histogram(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        if self.logging:
            self.tbWriter.add_image(*args, **kwargs)

    def add_video(self, *args, **kwargs):
        if self.logging:
            self.tbWriter.add_video(*args, **kwargs)

    def add_audio(self, *args, **kwargs):
        if self.logging:
            self.tbWriter.add_audio(*args, **kwargs)

    def add_text(self, *args, **kwargs):
        if self.logging:
            self.tbWriter.add_text(*args, **kwargs)

    def add_graph_onnx(self, *args, **kwargs):
        if self.logging:
            self.tbWriter.add_graph_onnx(*args, **kwargs)

    def add_graph(self, *args, **kwargs):
        if self.logging:
            self.tbWriter.add_graph(*args, **kwargs)

    def add_embedding(self, *args, **kwargs):
        if self.logging:
            self.tbWriter.add_embedding(*args, **kwargs)

    def add_pr_curve(self, *args, **kwargs):
        if self.logging:
            self.tbWriter.add_pr_curve(*args, **kwargs)

    def close(self, *args, **kwargs):
        if self.logging:
            self.tbWriter.close(*args, **kwargs)