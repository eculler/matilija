import csv
:wq
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

class Particle():

    def __init__(self, bins, position, n, obs, start, end):
        self.params = [key for key in bins.keys()]
        self.min = np.array([min(b) for b in bins.values()])
        self.max = np.array([max(b) for b in bins.values()])

        self.position = position
        self.n = n
        self.obs = obs
        self.start = start
        self.end = end

        self.i = 0
        self.ready = True

        self.rootdir = os.path.abspath('particle{n}'.format(n=n))
        if not os.path.isdir(self.rootdir):
            shutil.copytree('template', self.rootdir)
        self.template = os.path.join(self.rootdir, 'dhsvm.cfg.template')
        self.log = os.path.join(self.rootdir, 'fitness.log')
        with open(self.log, 'a') as log_file:
            writer = csv.writer(log_file)
            writer.writerow(['nse'] + self.params)

        self.best_position = self.position
        self.best_value = -float('inf')
        self.velocity = np.ones(self.position.shape)
        self._fitness = None

        self.windowdir = os.path.join(self.rootdir, 'state')
        self.station_dir = os.path.join(
            self.rootdir, 'input', 'station')
        self.station_tmp = False

    def __str__(self):
        return ', '.join([
            '{}={:.2}'.format(k, v)
            for k, v in zip(self.params, self.position)])

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness

    def configure(self, start, end):
        """ Set up DHSVM with current parameters """
        if not self.ready:
            return

        self.ready = False
        self._fitness = None
        self.i += 1
        self.start = start
        self.end = end

        # New output directory
        state = self.windowdir
        windowdirname = 'particle{n}.window{i:02d}.WY{end:%Y}'.format(
            n=self.n, i=self.i, end=self.end)
        self.windowdir = os.path.join(self.rootdir, windowdirname)
        if os.path.isdir(self.windowdir):
            shutil.rmtree(self.windowdir)
        os.mkdir(self.windowdir)

        # New station file
        if self.station_tmp:
            shutil.rmtree(self.station_tmp)
        self.station_tmp = os.path.join(self.windowdir, 'station')
        os.mkdir(self.station_tmp)
        for station in os.listdir(self.station_dir):
            if not station.startswith('Station'):
                continue
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
                    'precipitation'
                ],
                index_col='datetime',
                )
            df.index = pd.to_datetime(df.index, format='%m/%d/%Y-%H')
            df = df[(df.index >= start) & (df.index <= end)]
            df.precipitation = df.precipitation * self.position[-1]
            df.to_csv(
                os.path.join(self.station_tmp, station),
                sep = '\t',
                float_format = '%.5f',
                header = False,
                index_label = 'datetime',
                date_format = '%m/%d/%Y-%H',
            )
        # Write configuration file
        with open(self.template) as template:
            cfg = template.read()

        with open(self.cfg_path, 'w') as cfgfile:
            cfgfile.write(
                cfg.format(
                    start=self.start, end=self.end,
                    output=self.windowdir + '/', state=state + '/',
                    station_dir = self.station_tmp,
                    **{param: pos
                       for param, pos in zip(self.params, self.position)}
                )
            )

    @property
    def cfg_path(self):
        return os.path.join(self.rootdir, self.windowdir, 'dhsvm.cfg')

    @property
    def summary_path(self):
        return os.path.join(self.rootdir, self.windowdir, 'summary.txt')

    def move(self):
        """ Update position """
        self.position = self.position + self.velocity

        # Keep within bounds
        self.position = np.maximum(self.position, self.min)
        self.position = np.minimum(self.position, self.max)

    def run_next(self):
        """ Run DHSVM in a parallel process """

        logging.debug('RUNNING particle %d window %d start %s',
                      self.n, self.i, self.start.isoformat())

        # Start DHSVM in a new process
        with open(self.summary_path, 'w') as summary_file:
            logging.debug(mp.current_process())
            exit = subprocess.call([
                os.path.join(self.rootdir, 'DHSVM3.1.2'),
                self.cfg_path],
                cwd=self.rootdir,
                stdout=summary_file,
                stderr=summary_file)
            if not exit==0:
                self._fitness = -float('inf')
                logging.error(
                    'DHSVM FAILED particle %d window %d', self.n, self.i)

        self.ready = exit==0
        return self

    @property
    def fitness(self):
        """ Calculate calibration metrics """
        if self._fitness is None:
            results = pd.read_csv(
                os.path.join(self.rootdir, self.windowdir, 'Streamflow.Only'),
                delim_whitespace=True,
                skiprows=2,
                header = None,
                names = ['datetime', 'dhsvm'],
                index_col = 'datetime'
            )
            results.index = pd.to_datetime(
                results.index, format='%m.%d.%Y-%H:%M:%S')
            results = results.join(self.obs, how='left', on='datetime')
            self._fitness = 1 - (
                np.sum((results.dhsvm - results.observed)**2) /
                np.sum((results.observed - np.mean(results.observed))**2)
            )
        return self._fitness

    def update_best(self):
        if self.fitness > self.best_value:
            self.best_value = self.fitness
            self.best_position = self.position
        logging.debug('PBEST VALUE: %s', self.best_value)
        logging.debug('PBEST: %s', self)

        with open(self.log, 'a') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([self.best_value] + list(self.position))


class Swarm():

    W = 0.5
    c1 = 0.8
    c2 = 0.9

    def __init__(self, bins, dates, obs, jar):
        self.bins = bins
        for param, lst in self.bins.items():
            lst.sort()

        self.start_dates = dates[:-1]
        self.end_dates = dates[1:]
        self.i = 0

        self.obs = obs
        self.jar = jar

        self.initialize_particles()
        self.best_value = -float('inf')
        self.best_position = None

    def initialize_particles(self):
        """ Latin Hypercube sample of parameter space """
        self.particles = []

        ranges = {
            k: [ (b[i], b[i+1]) for i, _ in enumerate(b[:-1]) ]
            for k, b in self.bins.items()
        }
        for r in ranges.values():
            random.shuffle(r)

        n = 0
        while any(ranges.values()):
            position = np.array(
                [random.uniform(*r.pop()) for r in ranges.values()])
            particle = Particle(
                self.bins, position, n, self.obs, self.start, self.end)
            logging.debug('Created particle: %s', particle)
            self.particles.append(particle)
            n += 1

    def next(self):
        logging.debug('Starting window %d', self.i + 1)
        logging.debug('START DATE: %s', self.start.isoformat())
        logging.debug('END DATE: %s', self.end.isoformat())
        for p in self.particles:
            p.configure(self.start, self.end)
            p.ready = False

        pool = mp.Pool()
        self.particles = pool.map(self.run_particle_window, self.particles)
        pool.close()
        pool.join()

        self.update_best()
        self.move()
        self.pickle()
        self.i += 1

    def run_particle_window(self, particle):
        return particle.run_next()

    @property
    def start(self):
        return self.start_dates[self.i % len(self.start_dates)]

    @property
    def end(self):
        return self.end_dates[self.i % len(self.end_dates)]

    def update_best(self):
        """ Update global best value """
        for p in self.particles:
            p.update_best()
        best_particle = max(self.particles)
        if best_particle.fitness > self.best_value:
            self.best_value = best_particle.fitness
            self.best_position = best_particle.position
        logging.info('GBEST VALUE: %s', self.best_value)
        logging.info('GBEST: %s', self.best_position)

    def move(self):
        """ Move all the particles """
        for p in self.particles:
            p.velocity = (
                self.W * p.velocity +
                self.c1 * random.random() * (p.best_position - p.position) +
                self.c2 * random.random() * (self.best_position - p.position)
            )
            p.move()

    def pickle(self):
        """ Save state """
        with open(self.jar, 'wb') as jarfile:
            pickle.dump(self, jarfile)

if __name__ == '__main__':
    nbins = int(sys.argv[1]) + 1
    max_time = int(sys.argv[2])
    window = sys.argv[3]

    start_time = time.time()
    jar = 'swarm.jar'
    obs = pd.read_csv(
        'discharge.tsv',
        sep='\t',
        comment='#',
        skiprows = 34,
        header = None,
        usecols = [2, 4],
        names = ['datetime', 'observed'],
        index_col = 'datetime',
        parse_dates = ['datetime']
    )
    obs.observed = obs.observed / 35.31 * 3600 # cfs to m^3/h
    mainlog = '/Users/elsaculler/Documents/calibratetest/smoother.log'
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Starting calibration')
    bins = {
        'sandy_loam_K': list(np.logspace(-7, -5, num=nbins)),
        'sandy_loam_exp': list(np.linspace(0.2, 4, num=nbins)),
        'loam_K': list(np.geomspace(1e-6, 1e-4, num=nbins)),
        'loam_exp': list(np.linspace(0.2, 4, num=nbins)),
        'P_adj': list(np.linspace(0.8, 1.2, num=nbins))
    }
    dates = pd.date_range('2003-10-01', '2017-10-01', freq=window)
    if os.path.exists(jar):
        with open(jar, 'rb') as jarfile:
            swarm = pickle.load(jarfile)
    else:
        swarm = Swarm(bins, dates, obs, jar)

    round = 0
    wall = time.time() - start_time
    while wall < max_time:
        logging.debug('Wall time: %s', wall)
        swarm.next()
        wall = time.time() - start_time
