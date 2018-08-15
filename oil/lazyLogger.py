import pandas as pd
import time

class LogTimer(object):
        def __init__(self, minPeriod = 1, timeFrac = 1/10):
            self.avgLogTime = 0
            self.numLogs = 0
            self.lastLogTime = 0
            self.minPeriod = minPeriod #(measured in minutes)
            self.timeFrac = timeFrac
            self.performedLog = False
        def __enter__(self):
            """ returns yes iff the number of minutes have elapsed > minPeriod 
                and  > (1/timeFrac) * average time it takes to log """
            timeSinceLog = time.time() - self.lastLogTime
            self.performedLog = (timeSinceLog > 60*self.minPeriod) \
                                and (timeSinceLog > self.avgLogTime/self.timeFrac)
            if self.performedLog: self.lastLogTime = time.time()
            return self.performedLog
        def __exit__(self, *args):
            if self.performedLog:
                timeSpentLogging = time.time()-self.lastLogTime
                n = self.numLogs
                self.avgLogTime = timeSpentLogging/(n+1) + self.avgLogTime*n/(n+1)
                self.numLogs += 1
                self.lastLogTime = time.time()


class Silent(object):
    def __getattr__(self, name):
        if not hasattr(Silent, name):
            return (lambda *args, **kwargs:None)
try: 
    import tensorboardX
    maybeTbWriter = tensorboardX.SummaryWriter
except ModuleNotFoundError: 
    maybeTbWriter = Silent

class LazyLogger(maybeTbWriter):
    """ Thin wrapper around tensorboardX summarywriter,
        non tensorboard logging functionality to come
    """ 
    def __init__(self, log_dir = None, ema_com=5, min_period=1, time_frac = .1):
        self.text = {}
        self.constants = {}
        self.scalarFrame = pd.DataFrame()
        self.com = 20
        self.log_timer = LogTimer()
        super().__init__(log_dir)

    def maybe_do(self, log_func, *args, force=False, **kwargs):
        """ Executes log_func, passing in *args and **kwargs if
            sufficient time has passed since the previous log
            (as determined by LogTimer)"""
        if force: return log_func(*args, **kwargs)
        with self.log_timer as performLog:
            if performLog: return log_func(*args, **kwargs)

    def emas(self):
        return self.loggingFrame.ewm(com=self.com).mean().iloc[-1]

    def add_text(self, tag, text_string):
        try: self.text[tag].add(text_string)
        except KeyError: self.text[tag] = {text_string}
        super().add_text(tag, text_string)

    def _add_constants(self, tag, dic):
        try: self.constants[tag].update(dic)
        except KeyError: self.constants[tag] = dic
        for k,v in dic.items():
            super().add_text('Constants/'+tag, "{}:{}".format(k,v))

    def add_scalars(self, tag, dic, step=None, walltime=None):
        if step is None and walltime is None:
            self._add_constants(tag,dic)
        else:
            i = step if step is not None else walltime
            newRow = pd.DataFrame(dic, index = [i])
            self.scalarFrame = self.loggingFrame.append(newRow)
            super().add_scalars(tag, dic,step,walltime)

    def state_dict(self):
        # Will there be a problem with pickling the log_timer here?
        return {'text':self.text,'constants':self.constants,
                'scalarFrame':self.scalarFrame,'log_timer':self.log_timer}

    def load_state_dict(self, state):
        self.text = state['text']
        self.constants = state['constants']
        self.scalarFrame = state['scalarFrame']
        self.log_timer = state['log_timer']

    def __str__(self):
        return "{} object with text: {}, constants: {}, scalarFrame: {}".format(
            self.__class__,self.text,self.constants,self.scalarFrame)