
import ray
import pandas as import pd

@ray.remote(num_gpus=1)
class Trainable(object):
    #metric = "a dictionary key"
    def train(self, numEpochs):
        pass
    def getMetric(self):
        pass
    def save_checkpoint(self, save_path):
        pass
    def load_checkpoint(self, load_path):
        pass

# Responsible for the trainable, trial metadata, keeping information on outcome of training
# Stores result of training after trainable has been destroyed
class Trial(object):
    PENDING = "PENDING"; RUNNING = "RUNNING"; PAUSED = "PAUSED"; COMPLETED = "COMPLETED"
    trialNum = 0 # class global counter for trial id number
    
    def __init__(self, trainable_constr, config):
        self.trainable_constr = trainable_constr
        self.config = config
        #self.trainable = trainable_constr(config)
        self.id = Trial.trialNum
        Trial.trialNum += 1
        self.checkpoint_loc = None
        self.log = None#self.trainable.writer.loggingFrame
        self.status = Trial.PENDING

    def run(self, numEpochs):
        """constructs trainable, loads if trial is not new, executes 1 unit of training, 
            saves checkpoint and finally deletes trainable to free up memory"""
        self.trainable = self.trainable_constr(self.config)
        if self.checkpoint_loc!=None:
            self.trainable.load(self.checkpoint_loc)
        self.trainable.train(numEpochs)
        self.log =self.trainable.log # self.trainable.writer.loggingFrame
        metricVal = self.trainable.getMetric()#self.trainable.writer.emas()[self.trainable.metric]
        self.checkpoint_loc = self.trainable.save(self.id)
        self.trainable = None
        #outcomes = self.log.emas()
        return metricVal

        

class Experiment(object):
    def __init__(self, trainable_constr, configs, trial_scheduler=FIFO):
        self._trials = [Trial(trainable_constr,config) for config in configs]
        self.trial_scheduler_alg = trial_scheduler

    def get_next_trial(self):
        self._update_avail_resources()
        trial = self.trial_scheduler_alg.choose_trial_to_run(self._trials)

    def add_trials(self,trainable_constr, configs):
        self._trials.extend([Trial(trainable_constr,config) for config in configs])

    def cancel_trials(self,trial_ids):
        pass

    def is_finished(self):
        """ returns whether all trials have finished running"""
        for t in self._trials:
            if t.status in [Trial.PENDING, Trial.RUNNING, Trial.PAUSED]:
                return False
        return True

    def get_best_config(self):
        pass
    
    
class ExperimentRunner(object):
    def __init__(self, experiments, weightBySize = False):
        self.weightBySize = weightBySize
        self._experiments = experiments

    def add_experiments(self, new_experiments):
        self._experiments.extend(new_experiments)

    def cancel_experiments(self, experiment_ids):
        pass

    def schedule_next(self):
        pass

def _update_avail_resources(self):
    clients = ray.global_state.client_table()
    local_schedulers = [
        entry for client in clients.values() for entry in client if
        (entry['ClientType'] == 'local_scheduler' and not entry['Deleted'])
    ]
    num_cpus = sum(ls['CPU'] for ls in local_schedulers)
    num_gpus = sum(ls.get('GPU', 0) for ls in local_schedulers)
    self._avail_resources = Resources(int(num_cpus), int(num_gpus))
    self._resources_initialized = True


remoteClass = ray.remote(num_cpus=self.cpus_required, num_gpus = self.gpus_required)(clas)
runner = remoteClass.remote(config=self.config)