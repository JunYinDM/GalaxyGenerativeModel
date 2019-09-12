import os
import json
import numpy as np
from tensorboardX import SummaryWriter


class Logger(object):
    def __init__(self, args, path, metrics=['loss']):
        super(Logger, self).__init__()
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(os.path.join(path, 'models')):
            os.makedirs(os.path.join(path, 'models'))
        with open(os.path.join(path, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        self.path = path
        self.metrics = metrics
        self.tf_writer = SummaryWriter(log_dir=self.path)
        self.csv_writer = open(os.path.join(self.path, 'log.csv'), 'w')

    def write(self, log, epoch, stage='train'):
        for key in log.keys():
            if key not in self.metrics:
                continue
            value = log[key]
            if isinstance(value, list):
                for idx, v in enumerate(value):
                    self.tf_writer.add_scalar('%s/%s/%i' % (stage, key, idx), v, epoch)
                    self.csv_writer.write('%d,%s,%s,%d,%f\n'%(epoch, stage, key, idx, v))
            elif isinstance(value, np.ndarray):
 #               self.tf_writer.add_embedding(value, global_step=epoch, tag='%s/%s'%(stage, key))
                 pass
            else:
                self.tf_writer.add_scalar('%s/%s'%(stage, key), log[key], epoch)
                self.csv_writer.write('%d,%s,%s,%f\n'%(epoch, stage, key, log[key]))
            self.csv_writer.flush()

    def close(self):
        self.tf_writer.close()
        self.csv_writer.close()
