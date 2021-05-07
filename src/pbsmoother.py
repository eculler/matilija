import csv
import collections
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
        new = np.random.normal(self.mu, self.sigma, param.size) * param
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
                 weight, n, err_sigma,
                 data_dir, window_dir, station_fn, log):
        self.obs_error = obs_error
        self.input_error = input_error
        self.state = state
        self.state_perturbation = state_perturbation
        self.weight = weight
        self.n = n
        self.err_sigma = err_sigma

        self.ready = True
        self.start = None
        self.end = None

        # Establish directories and file paths
        self.root_dir = os.path.join(window_dir, 'particle{n}'.format(n=self.n))
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        self.data_dir = data_dir
        self.station_dir = os.path.join(self.data_dir, 'input', 'station')
        self.stations = glob.glob(os.path.join(self.station_dir, station_fn))
        logging.debug('STATIONS: %s', self.stations)

        self.adj_station_dir = os.path.join(self.root_dir, 'station')
        if not os.path.exists(self.adj_station_dir):
            os.makedirs(self.adj_station_dir)

        self.log = log

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

        # Perturb the observations
        for station in self.stations:
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
            df.precipitation = self.input_error['P'].perturb(df.precipitation)

            logging.debug(station)
            adj_station_pth = os.path.join(self.adj_station_dir,
                                           os.path.basename(station))
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

        # Perturb the state
        self.state = {
            param: self.state_perturbation[param].perturb.perturb(state)
            for param, state in self.state.items()
        }

        # Write configuration file

    @property
    def cfg_path(self):
        return os.path.join(self.rootdir, self.windowdir, 'dhsvm.cfg')

    @property
    def summary_path(self):
        return os.path.join(self.rootdir, self.windowdir, 'summary.txt')

    def run_next(self):
        """ Run DHSVM in a parallel process """

        logging.debug('RUNNING particle %s start %s',
                      self.n, self.start.isoformat())
        logging.debug('State: %s', self.state)

        for station in self.stations:
            df = pd.read_csv(
                os.path.join(self.station_dir, station),
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

        df['streamflow'] = df.precipitation * self.state['m'] + self.state['b']
        df[['streamflow']].to_csv(
            os.path.join(self.root_dir, 'Streamflow.Only'),
            sep='\t',
            header = False,
            date_format = '%m.%d.%Y-%H:%M:%S'
        )
        return self

    def likelihood(self, results, obs):
        return (np.exp(-0.5 * ((obs - results) / self.err_sigma)**2) /
                (err_sigma * np.sqrt(2 * np.pi)))

    def rmse(self, results, obs):
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
        logging.debug(results)
        results.index = pd.to_datetime(
            results.index, format='%m.%d.%Y-%H:%M:%S')
        results = results.join(obs, how='inner')

        results['likelihood'] = self.likelihood(results['modelled'],
                                                results['observed'])
        logging.debug(results)

        self.weight = np.prod(results['likelihood'])
        logging.debug('UPDATED WEIGHT: %s', self.weight)

        with open(self.log, 'a') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([self.weight,
                             self.rmse(results['modelled'],
                                       results['observed']),
                             self.state['m'][0], self.state['b'][0]])


class Smoother():
    """Manage Particles according to particle batch smoother algorithm"""

    def __init__(self, nparticle,
                 obs_error, input_error, state_perturbation,
                 dates, obs, err_sigma,
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
        self.err_sigma = err_sigma
        self.jar = jar

        self.data_dir = data_dir
        self.out_dir = out_dir
        self.station_fn = station_fn
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Initialize logging file
        self.log = os.path.join(self.out_dir, 'particles.csv')
        self.all_params = ['m', 'b']
        with open(self.log, 'w') as log_file:
            writer = csv.writer(log_file)
            writer.writerow(['weight', 'rmse'] + self.all_params)

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
            state = {key: st[pidx] for key, st in initial_states.items()}
            self.new_particles.append(
                Particle(self.obs_error, self.input_error,
                         state, self.state_perturbation,
                         weight, pidx, self.err_sigma,
                         self.data_dir, self.window_dir,
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
        #result = []
        #for particle in self.particles:
        #    result.append(self.run_particle_window(particle))
        #self.particles = result

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

        total_weights = np.sum([p.weight for p in self.particles])
        for p in self.particles:
            p.weight = p.weight / total_weights

    def resample(self):
        self.particles = list(filter(lambda x: x.weight>0.001, self.particles))
        if len(self.particles)==0:
            raise ValueError('No particles with non-negligible weights')
        self.new_particles = [
            Particle(self.obs_error, self.input_error,
                     state, self.state_perturbation,
                     None, idx, self.err_sigma,
                     self.data_dir, self.window_dir, self.station_fn, self.log)
            for idx, state in enumerate(random.choices(
                [p.state for p in self.particles],
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
        na_values = 'Eqp'
    )
    logging.debug(obs)
    obs.observed = obs.observed / 35.31 * 3600 / 125199354 * 1000 # cfs to mm/h

    # Initialize perturbation distributions
    pct_error = 0.06
    err_sigma = pct_error * np.max(obs.observed)
    logging.debug('OBSERVATION ERROR STDDEV: %s', err_sigma)
    obs_error = {
        'streamflow':  GaussianMult(1, pct_error, None, None)
        }
    input_error = {
        'P': GaussianMult(1, 0.1, 0, None)
        }
    state_perturbation = {
        'm': State(init=Uniform(-5, 5),
                    perturb=GaussianAdd(0, .1, None, None)),
        'b': State(init=Uniform(.001, .1),
                    perturb=GaussianMult(1, .1, 0, None))
        }


    # Compute window dates
    dates = pd.date_range('2003-10-01', '2017-10-01', freq=window)
    # Initialize smoother
    if os.path.exists(jar):
        with open(jar, 'rb') as jarfile:
            pbs = pickle.load(jarfile)
    else:
        pbs = Smoother(nparticle,
                       obs_error, input_error, state_perturbation,
                       dates, obs, err_sigma,
                       jar,
                       data_dir, out_dir, station_fn)


    # Run smoother for each window
    for window in dates[:-1]:
        logging.debug('Window: %s', window)
        pbs.next()
