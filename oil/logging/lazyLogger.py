import pandas as pd
import time

class LogTimer(object):
        def __init__(self, minPeriod = 1, timeFrac = 1/10, **kwargs):
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


# class Silent(object):
#     def __init__(self,*args,**kwargs):
#         pass#super().__init__(*args,**kwargs)
#     def __getattr__(self, name):
#         if not hasattr(Silent, name):
#             return (lambda *args, **kwargs:None)
# def silent(*args,**kwargs):
#     pass

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
    """ Thin wrapper around tensorboardX summarywriter,
        non tensorboard logging functionality to come
    """ 
    def __init__(self, log_dir = None, no_print=False, ema_com=0, **kwargs):
        self.text = {}
        self.constants = {}
        self.scalar_frame = pd.DataFrame()
        self.no_print = no_print
        self._com = ema_com
        self._log_dir = log_dir
        self._unreported = {}
        super().__init__(log_dir=log_dir, **kwargs)

    def report(self):
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
        try: self.text[tag].add(text_string)
        except KeyError: self.text[tag] = {text_string}
        self._unreported[tag] = text_string
        #TB writer add_text doesn't work properly?
        #super().add_text(tag, text_string)

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
            self.scalar_frame = self.scalar_frame.combine_first(newRow)
            super().add_scalars(tag, dic, step)#, walltime=walltime) #TODO: update tensorboardX?

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