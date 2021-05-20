import csv
import collections
from decimal import Decimal
import glob
import logging
import multiprocessing.dummy as mp
import os
import pandas as pd
import pickle
import random
import shutil
import subprocess
import sys
import time
import numpy as np

State = collections.namedtuple('State', ['init', 'perturb'])

class GaussianMult():

    def __init__(self, mu, sigma, min, max):
        self.mu = mu
        self.sigma = sigma
        self.min = min
        self.max = max

    def perturb(self, param):
        param = np.asarray([param]) if np.isscalar(param) else np.asarray(param)
        perturb = np.random.normal(self.mu, self.sigma, param.size)
        new = perturb * param
        if not self.min is None:
            new[new < self.min] = self.min
        if not self.max is None:
            new[new > self.max] = self.max
        return new

class GaussianAdd():

    def __init__(self, mu, sigma, min, max):
        self.mu = mu
        self.sigma = sigma
        self.min = min
        self.max = max

    def perturb(self, param):
        param = np.asarray([param]) if np.isscalar(param) else np.asarray(param)
        new = np.random.normal(self.mu, self.sigma, param.size) + param
        if not self.min is None:
            new[new < self.min] = self.min
        if not self.max is None:
            new[new > self.max] = self.max
        return new

class Uniform():

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def perturb(self, param):
        param = np.asarray([param]) if np.isscalar(param) else np.asarray(param)
        factor = np.random.uniform(min, max, size(param))
        return factor

    def initialize(self, n):
        sequence = np.linspace(self.min, self.max, n)
        random.shuffle(sequence)
        return sequence


class Particle():

    def __init__(self,
                 obs_error, input_error, state, state_perturbation,
                 weight, n,
                 data_dir, window_dir, state_dir, station_fn, log):
        self.obs_error = obs_error
        self.input_error = input_error
        self.state = state
        self.state_perturbation = state_perturbation
        self.weight = weight
        self.n = n

        self.ready = True
        self.start = None
        self.end = None

        # Establish directories and file paths
        self.root_dir = os.path.join(window_dir, 'particle{n}'.format(n=self.n))
        self.stdout_pth = os.path.join(self.root_dir, 'stdout.txt')
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        self.cfg_dir = os.path.join(self.root_dir, 'cfg')
        if not os.path.exists(self.cfg_dir):
            os.makedirs(self.cfg_dir)
        self.cfg_path = os.path.join(self.cfg_dir, 'dhsvm.cfg')

        self.data_dir = data_dir
        self.input_dir = os.path.join(self.data_dir, 'input')
        self.station_dir = os.path.join(self.input_dir, 'station')
        self.stations = glob.glob(os.path.join(self.station_dir, station_fn))
        logging.debug('STATIONS: %s', self.stations)

        self.adj_input_dir = os.path.join(self.root_dir, 'input')
        self.adj_station_dir = os.path.join(self.adj_input_dir, 'station')
        if not os.path.exists(self.adj_station_dir):
            os.makedirs(self.adj_station_dir)

        self.in_state_dir = state_dir

        self.log = log

        self.template_dir = os.path.join(self.data_dir, 'template')
        self.template_cfg = os.path.join(self.template_dir,
                                         'dhsvm.cfg.template')

    def __str__(self):
        return ', '.join([
            '{}={:.2}'.format(k, v)
            for k, v in zip(self.params, self.position)])

    def configure(self, start, end):
        """ Set up DHSVM with current parameters """
        if not self.ready:
            return

        self.ready = False
        self.score = None
        self.start = start
        self.end = end

        # Perturb the model state parameters
        self.state = {
            param: self.state_perturbation[param].perturb.perturb(state)
            for param, state in self.state.items()
        }

        # Perturb the precipitation observations
        for i, station in enumerate(self.stations, start=1):
            df = pd.read_csv(
                station,
                sep='\t',
                names = [
                    'datetime',
                    'air_temp',
                    'wind_speed',
                    'relative_humidity',
                    'incoming_shortwave',
                    'incoming_longwave',
                    'precipitation'],
                index_col='datetime')
            df.index = pd.to_datetime(df.index, format='%m/%d/%Y-%H')
            df = df[(df.index >= self.start) & (df.index <= self.end)]
            df.precipitation = self.input_error['P'].perturb(
                df.precipitation * self.state['P_adj'])

            logging.debug(station)
            adj_station_pth = os.path.join(self.adj_station_dir,
                                           'Station{}.tsv'.format(i))
            logging.debug('Precipitation adjusted and saved to %s',
                          adj_station_pth)
            df.to_csv(
                adj_station_pth,
                sep = '\t',
                float_format = '%.5f',
                header = False,
                index_label = 'datetime',
                date_format = '%m/%d/%Y-%H',
            )

         # Write configuration file
        with open(self.template_cfg) as template:
            cfg = template.read()

        logging.debug(self.state)
        with open(self.cfg_path, 'w') as cfg_file:
            cfg_file.write(
                cfg.format(
                    start=self.start,
                    end=self.end,
                    **{key: value[0] for key, value in self.state.items()}
                )
            )


    def run_next(self):
        """ Run DHSVM in a parallel process """

        logging.debug('RUNNING particle %s start %s',
                      self.n, self.start.isoformat())
        logging.debug('State: %s', self.state)

        # Start DHSVM in a new process
        with open(self.stdout_pth, 'w') as summary_file:
            logging.debug(mp.current_process())
            exit = subprocess.call(
                ['docker', 'run',
                 '-v', self.cfg_dir + ':/matilija/cfg',
                 '-v', self.data_dir + '/input:/matilija/input',
                 '-v', self.root_dir + '/input:/matilija/variable_input',
                 '-v', self.in_state_dir + ':/matilija/state',
                 '-v', self.root_dir + '/output:/matilija/output',
                 '-t', 'dhsvm-matilija:latest',
                 '/matilija/src/dhsvm/build/DHSVM/sourcecode/DHSVM',
                 '/matilija/cfg/dhsvm.cfg'],
                cwd=self.root_dir,
                stdout=summary_file,
                stderr=summary_file)
            if not exit==0:
                self._fitness = -float('inf')
                logging.error(
                    'DHSVM FAILED particle %d window %d', self.n, self.i)

        self.ready = exit==0
        return self

    def likelihood(self, results, obs):
        obs = [Decimal(o) for o in obs]
        results = [Decimal(r) for r in results]
        e = Decimal(np.log(1 + self.obs_error**2))
        return [(np.exp((r.log10() - o.log10())**2 / (Decimal(-2) * e**2)) /
                 (r * e * Decimal(np.sqrt(2 * np.pi))))
                for r, o in zip(results, obs)]


    def calc_rmse(self, results, obs):
        return np.sqrt(np.mean((obs - results)**2))

    def update_weight(self, obs):
        """ Calculate a new weight based on observations """
        results = pd.read_csv(
            os.path.join(self.root_dir, 'Streamflow.Only'),
            sep='\t',
            #skiprows=2,
            header = None,
            names = [
                'datetime',
                'modelled'],
            index_col = 'datetime'
        )

        results.index = pd.to_datetime(
            results.index, format='%m.%d.%Y-%H:%M:%S')
        results = results.join(obs, how='inner')
        results['likelihood'] = self.likelihood(results['modelled'].to_numpy(),
                                                results['observed'].to_numpy())
        logging.debug(results)

        self.weight = Decimal(1)
        for l in results['likelihood']:
            self.weight *= Decimal(l)
        logging.debug('UPDATED WEIGHT: %s', self.weight)

        self.rmse = self.calc_rmse(results['modelled'], results['observed'])

    def write_log(self):
        with open(self.log, 'a') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([self.n,
                             self.weight,
                             self.rmse] +
                             [val[0] for val in self.state.values()])


class Smoother():
    """Manage Particles according to particle batch smoother algorithm"""

    def __init__(self, nparticle,
                 obs_error, input_error, state_perturbation,
                 dates, obs,
                 jar,
                 data_dir, out_dir, station_fn):
        # Set smoother attributes
        self.nparticle = nparticle

        self.obs_error = obs_error
        self.input_error = input_error
        self.state_perturbation = state_perturbation

        self.start_dates = dates[:-1]
        self.end_dates = dates[1:]
        self.i = 0

        self.obs = obs
        self.jar = jar

        self.data_dir = data_dir
        self.out_dir = out_dir
        self.init_state_dir = os.path.join(self.data_dir, 'state')
        self.station_fn = station_fn
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Initialize logging file
        self.log = os.path.join(self.out_dir, 'particles.csv')
        self.all_params = list(self.state_perturbation.keys())
        with open(self.log, 'w') as log_file:
            writer = csv.writer(log_file)
            writer.writerow(['particle', 'weight', 'rmse'] + self.all_params)

        # Initialize particles
        self.initialize_particles()

    @property
    def start(self):
        return self.start_dates[self.i]

    @property
    def end(self):
        return self.end_dates[self.i]

    @property
    def window_dir(self):
        window_dir = os.path.join(self.out_dir,
                                  'window{i}'.format(i=self.i+1))
        if not os.path.exists(window_dir):
            os.makedirs(window_dir)
        return window_dir

    def initialize_particles(self):
        """ Initial sample of parameter space """
        self.new_particles = []

        # Select particle states from uniform distribution...
        initial_states = {
            key: st.init.initialize(self.nparticle)
            for key, st in self.state_perturbation.items()}
        logging.debug(self.state_perturbation)
        logging.debug(initial_states)
        # ... with uniform weights
        weight = 1 / self.nparticle
        for pidx in range(self.nparticle):
            state = collections.OrderedDict([
                (key, st[pidx]) for key, st in initial_states.items()])
            self.new_particles.append(
                Particle(self.obs_error, self.input_error,
                         state, self.state_perturbation,
                         weight, pidx,
                         self.data_dir, self.window_dir, self.init_state_dir,
                         self.station_fn, self.log))
            logging.debug('Initial state of particle %s: %s', pidx, state)


    def next(self):
        logging.info('Starting window %s', self.i + 1)
        logging.info('START DATE: %s', self.start.isoformat())
        logging.info('END DATE: %s', self.end.isoformat())

        self.particles = self.new_particles

        pool = mp.Pool(nparticle)
        self.particles = pool.map(self.run_particle_window, self.particles)
        pool.close()
        pool.join()

        self.update_weights()
        self.i += 1

        self.resample()
        self.pickle()


    def run_particle_window(self, particle):
        particle.configure(self.start, self.end)
        return particle.run_next()

    def update_weights(self):
        for p in self.particles:
            p.update_weight(self.obs)

        total_weights = Decimal(0)
        for p in self.particles:
            total_weights += p.weight
            logging.debug('TOTAL WEIGHT: %s', total_weights)

        for p in self.particles:
            p.weight = float(p.weight / total_weights)
            p.write_log()
            logging.debug('NORMALIZED WEIGHT: %s', p.weight)

    def resample(self):
        self.particles = list(filter(lambda x: x.weight>0.001, self.particles))
        if len(self.particles)==0:
            raise ValueError('No particles with non-negligible weights')
        self.new_particles = [
            Particle(self.obs_error, self.input_error,
                     p.state, self.state_perturbation,
                     None, idx,
                     self.data_dir, self.window_dir, p.root_dir,
                     self.station_fn, self.log)
            for idx, p in enumerate(random.choices(
                self.particles,
                [p.weight for p in self.particles],
                k=self.nparticle))
            ]

    def pickle(self):
        """ Save state """
        with open(self.jar, 'wb') as jarfile:
            pickle.dump(self, jarfile)


if __name__ == '__main__':
    # Set up logging
    mainlog = os.path.abspath('smoother.log')
    logging.basicConfig(
        #filename=mainlog,
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Starting assimilation')

    # Command line parameters
    nparticle = int(sys.argv[1])
    window = sys.argv[2]
    data_dir = sys.argv[3]
    discharge_fn = sys.argv[4]
    station_fn = sys.argv[5]
    out_dir = sys.argv[6]

    # Derived parameters
    jar = os.path.join(out_dir, 'swarm.jar')

    # Prepare observations
    obs = pd.read_csv(
        os.path.join(data_dir, discharge_fn),
        sep='\t',
        comment='#',
        skiprows = 34,
        header = None,
        usecols = [2, 4],
        names = ['datetime', 'observed'],
        index_col = 'datetime',
        parse_dates = ['datetime'],
        na_values = 'Eqp',
        engine='python'
    )
    logging.debug(obs)
    obs.observed = obs.observed / 35.31 * 3600 / 125199354 * 1000 # cfs to mm/h

    # Initialize perturbation distributions
    obs_error = 0.2
    input_error = {
        'P': GaussianMult(1, 0.01, 0, None)
        }
    state_perturbation = collections.OrderedDict([
        ('b',
         State(init=Uniform(.001, .1),
               perturb=GaussianMult(1, .3, 0, None))),
        ('P_adj',
         State(init=Uniform(0.8, 2.5),
               perturb=GaussianMult(1, .3, 0, None))),
        ('sandy_loam_K',
         State(init=Uniform(-5, -3),
               perturb=GaussianMult(1, .1, -10, -1))),
        ('sandy_loam_exp',
         State(init=Uniform(0.2, 4),
               perturb=GaussianMult(1, .1, 0, None))),
        ('loam_K',
         State(init=Uniform(-6, -4),
               perturb=GaussianMult(1, .1, -10, -1))),
        ('loam_exp',
         State(init=Uniform(.2, 4),
               perturb=GaussianMult(1, .1, 0, None)))
        ])


    # Compute window dates
    dates = pd.date_range('2003-10-01', '2017-10-01', freq=window)
    # Initialize smoother
    if os.path.exists(jar):
        with open(jar, 'rb') as jarfile:
            pbs = pickle.load(jarfile)
    else:
        pbs = Smoother(nparticle,
                       obs_error, input_error, state_perturbation,
                       dates, obs,
                       jar,
                       data_dir, out_dir, station_fn)


    # Run smoother for each window
    for window in dates[:-1]:
        logging.debug('Window: %s', window)
        pbs.next()
