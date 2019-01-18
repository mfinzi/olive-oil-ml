import pandas as pd
import torch
import time
import os
import dill

class LogTimer(object):
        """ Timer to automatically control time spent on expensive logs
                    by not logging when computations performed in __enter__
                    exceed the specified fraction of total time outside.
        """
        def __init__(self, minPeriod = 1, timeFrac = 1/10, **kwargs):
            """ minPeriod: minimum time between logs.
                timeFrac: max fraction of total time spent inside __enter__ block."""
            self.avgLogTime = 0
            self.numLogs = 0
            self.lastLogTime = 0
            self.minPeriod = minPeriod #(measured in minutes)
            self.timeFrac = timeFrac
            self.performedLog = False
            super().__init__(**kwargs)

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


# If tensorboardX fails to load, we replace it with a writer
# that does nothing
class NothingWriter(object):
    add_scalar = add_scalars = add_scalars_to_json = add_image \
    = add_image_with_boxes = add_figure = add_video = add_audio \
    = add_text = add_onnx_graph = add_graph = add_embedding \
    = add_pr_curve_raw = close = lambda *args,**kwargs:None
    def __init__(self, log_dir=None,comment='',**kwargs):
        return super().__init__(**kwargs)
try: 
    import tensorboardX
    MaybeTbWriter = tensorboardX.SummaryWriter
except ModuleNotFoundError: 
    MaybeTbWriter = NothingWriter

class MaybeTbWriterWSerial(MaybeTbWriter):
    """ Wraps summary writer but allows pickling with set and getstate """
    def __getstate__(self):
        return dict((k, v) for k, v in self.__dict__.items()
                       if not k in ['file_writer','all_writers'])
    def __setstate__(self,state):
        self.__init__(log_dir = state['_log_dir'])
        self.__dict__.update(state)


class LazyLogger(LogTimer, MaybeTbWriterWSerial):
    """ Tensorboard logging to log_dir, logged scalars are also stored to 
        a pandas dataframe called constants. Logged text is additionally
        store in a dictionary called text.
        Lazy context manager functionality allows controlling time spent on
        expensive logging operations to a fixed fraction. See LogTimer for
        more details.
    """ 
    def __init__(self, log_dir = None, no_print=False, ema_com=0, **kwargs):
        """ log_dir: Where tensorboardX logs are saved, tb default
            no_print: if True printing is disabled
            ema_com: if nonzero, emas and report show the exponential moving
                       average of tracked scalars
        """
        self.text = {}
        self.constants = {}
        self.scalar_frame = pd.DataFrame()
        self.no_print = no_print
        self._com = ema_com
        self._log_dir = log_dir
        self._unreported = {}
        super().__init__(log_dir=log_dir, **kwargs)

    def report(self):
        """ prints all unreported text and constants, prints scalar emas"""
        if self.no_print: return
        for unreported_info in self._unreported.values():
            print(unreported_info)#+'\n')
        self._unreported = {}
        emas = self.emas()
        print(emas)
        return emas

    @property # Needs to be read only
    def log_dir(self):
        return self._log_dir

    def emas(self):
        """ Returns the exponential moving average of the logged
            scalars (not consts) """
        return self.scalar_frame.ewm(com=self._com).mean().iloc[-1:]

    def add_text(self, tag, text_string):
        """ text_string is logged (into text and tensorboard)
            tag can be specified to allow overwrites so that
            a frequently logged text under a tag will only show
            the most recent after a report """
        try: self.text[tag].add(text_string)
        except KeyError: self.text[tag] = {text_string}
        self._unreported[tag] = text_string
        super().add_text(tag, text_string)

    def _add_constants(self, tag, dic):
        try: self.constants[tag].update(dic)
        except KeyError: self.constants[tag] = dic
        with pd.option_context('display.expand_frame_repr',False):
            self.add_text('Constants/{}'.format(tag),str(pd.Series(dic).to_frame(tag).T))

    def add_scalars(self, tag, dic, step=None, walltime=None):
        """ Like tensorboard add_scalars, but if step and walltime
             are not specified, the dic is assumed to hold constants
             which are logged as text using add_text"""
        if step is None and walltime is None:
            self._add_constants(tag,dic)
        else:
            i = step if step is not None else walltime
            self.scalar_frame.loc[i] = pd.Series(dic)
            super().add_scalars(tag, dic, step)#, walltime=walltime) #TODO: update tensorboardX?

    def save_object(self,obj,suffix):
        final_path = self.log_dir+suffix
        os.makedirs(os.path.dirname(final_path),exist_ok=True)
        torch.save(obj,final_path,pickle_module=dill)
        return os.path.abspath(final_path)

    def state_dict(self):
        # Will there be a problem with pickling the log_timer here?
        return {'text':self.text,'constants':self.constants,
                'scalar_frame':self.scalar_frame}

    def load_state_dict(self, state):
        self.text = state['text']
        self.constants = state['constants']
        self.scalar_frame = state['scalar_frame']
        #self.log_timer = state['log_timer']

    def __str__(self):
        return "{} object with text: {}, constants: {}, scalar_frame: {}.\n\
            logging in directory: {}".format(
            self.__class__,self.text,self.constants,self.scalar_frame,self.log_dir)