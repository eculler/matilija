import csv
import collections
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

    def __init__(self, bins, position, n, obs, start, end, dhsvm):
        self.params = [key for key, _ in bins.items()]
        self.min = np.array([min(b) for b in bins.values()])
        self.max = np.array([max(b) for b in bins.values()])

        self.position = position
        self.n = n
        self.obs = obs
        self.start = start
        self.end = end
        self.dhsvm = dhsvm

        self.i = 0
        self.ready = True
        self.weight = None
        self._fitness = None

        self.rootdir = os.path.abspath(os.curdir)
        self.workroot = os.path.join(self.rootdir, '_work')
        os.makedirs(self.workroot, exist_ok=True)
        self.workdir = os.path.join(
            self.workroot, 'particle{n:03d}'.format(n=n))
        self._prepare_workdir()
        self.template = os.path.join(self.workdir, 'dhsvm.cfg.template')

        self.state_dir = os.path.join(self.workdir, 'state')
        self.windowdir = None
        self.output_dir = None
        self.station_dir = os.path.join(self.workdir, 'input', 'station')
        self.station_tmp = False

    def __str__(self):
        return ', '.join([
            '{}={:.2}'.format(k, v)
            for k, v in zip(self.params, self.position)])

    def configure(self, start, end):
        """ Set up DHSVM with current parameters """
        if not self.ready:
            return
        self._ensure_paths()

        self.ready = False
        self._fitness = None
        self.weight = None
        self.i += 1
        self.start = start
        self.end = end

        # New output directory: <run root>/windowNNN/particleNNN/
        state = self._initial_state_dir()
        self.windowdir = os.path.join(
            self.rootdir, 'window{i:03d}'.format(i=self.i))
        self.output_dir = os.path.join(
            self.windowdir, 'particle{n:03d}'.format(n=self.n))
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

        # New station file
        if self.station_tmp and os.path.isdir(self.station_tmp):
            shutil.rmtree(self.station_tmp)
        self.station_tmp = os.path.join(self.output_dir, 'station')
        os.mkdir(self.station_tmp)
        station_files = sorted(
            f for f in os.listdir(self.station_dir)
            if f.startswith('Station'))
        for i, station in enumerate(station_files, start=1):
            df = pd.read_csv(
                os.path.join(self.station_dir, station),
                sep='\t',
                names=[
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
            first_record = df.index.min()
            last_record = df.index.max()
            df = df[(df.index >= start) & (df.index <= end)]
            if df.empty:
                raise ValueError(
                    '{station} has no records for {start} to {end}; '
                    'available range is {first} to {last}'.format(
                        station=station,
                        start=start.isoformat(),
                        end=end.isoformat(),
                        first=first_record.isoformat(),
                        last=last_record.isoformat()))
            df.precipitation = df.precipitation * self.position[-1]
            logging.debug('Precipitation adjusted by: %s', self.position[-1])
            df.to_csv(
                os.path.join(self.station_tmp, 'Station{}.tsv'.format(i)),
                sep='\t',
                float_format='%.5f',
                header=False,
                index_label='datetime',
                date_format='%m/%d/%Y-%H',
            )
        # Write configuration file
        with open(self.template) as template:
            cfg = template.read()

        output = self._relative_to_workdir(self.output_dir)
        state = self._relative_to_workdir(state)
        station_dir = self._relative_to_workdir(self.station_tmp)

        with open(self.cfg_path, 'w') as cfgfile:
            cfgfile.write(
                cfg.format(
                    start=self.start, end=self.end,
                    output=output + '/', state=state + '/',
                    station_dir=station_dir,
                    **{param: pos
                       for param, pos in zip(self.params, self.position)}
                )
            )

    @property
    def cfg_path(self):
        self._ensure_paths()
        return os.path.join(self.output_dir, 'dhsvm.cfg')

    @property
    def cfg_arg_path(self):
        self._ensure_paths()
        return self._relative_to_workdir(self.cfg_path)

    @property
    def summary_path(self):
        self._ensure_paths()
        return os.path.join(self.output_dir, 'summary.txt')

    @property
    def streamflow_path(self):
        self._ensure_paths()
        return os.path.join(self.output_dir, 'Streamflow.Only')

    def _ensure_paths(self):
        """Populate path attributes for particles loaded from older jars."""
        if not hasattr(self, 'workdir'):
            old_rootdir = self.rootdir
            if os.path.basename(old_rootdir).startswith('particle'):
                self.workdir = old_rootdir
                self.rootdir = os.path.dirname(old_rootdir)
            else:
                self.workroot = os.path.join(self.rootdir, '_work')
                self.workdir = os.path.join(
                    self.workroot, 'particle{n:03d}'.format(n=self.n))

        if not hasattr(self, 'workroot'):
            self.workroot = os.path.dirname(self.workdir)

        if not hasattr(self, 'state_dir'):
            self.state_dir = getattr(
                self, 'windowdir', os.path.join(self.workdir, 'state'))

        self._prepare_workdir()
        if not getattr(self, 'windowdir', None):
            self.windowdir = None
        if not hasattr(self, 'output_dir') or self.output_dir is None:
            current = getattr(self, 'windowdir', None)
            if current and os.path.basename(current).startswith('particle'):
                self.output_dir = current
                self.windowdir = os.path.dirname(current)

        self.template = os.path.join(self.workdir, 'dhsvm.cfg.template')
        self.station_dir = os.path.join(self.workdir, 'input', 'station')

    def _prepare_workdir(self, refresh=False):
        """Create particle workdir; optionally refresh non-state inputs."""
        if not os.path.isdir(self.workdir):
            shutil.copytree('template', self.workdir, symlinks=True)
            return

        if not refresh:
            return

        root_template = os.path.join(self.rootdir, 'template')
        for relpath in (
                'dhsvm.cfg.template',
                os.path.join('input', 'stream', 'stream.network.csv'),
                os.path.join('input', 'stream', 'stream.network.saveall.csv')):
            src = os.path.join(root_template, relpath)
            dst = os.path.join(self.workdir, relpath)
            if os.path.isfile(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

    def _initial_state_dir(self):
        """Return the directory containing the state files for self.start."""
        state_file = 'Channel.State.{:%m.%d.%Y.%H.%M.%S}'.format(self.start)
        if os.path.isfile(os.path.join(self.state_dir, state_file)):
            return self.state_dir

        dated_state_dir = os.path.join(
            self.state_dir, 'state.{:%Y.%m.%d.%H.%M.%S}'.format(self.start))
        if os.path.isfile(os.path.join(dated_state_dir, state_file)):
            return dated_state_dir

        return self.state_dir

    def _relative_to_workdir(self, path):
        return os.path.relpath(path, self.workdir)

    def run_next(self):
        """ Run DHSVM in a parallel process """

        logging.debug('RUNNING particle %d window %d start %s',
                      self.n, self.i, self.start.isoformat())

        with open(self.summary_path, 'w') as summary_file:
            logging.debug(mp.current_process())
            exit_code = subprocess.call(
                [self.dhsvm, self.cfg_arg_path],
                cwd=self.workdir,
                stdout=summary_file,
                stderr=summary_file)
            if exit_code != 0:
                self._fitness = -float('inf')
                logging.error(
                    'DHSVM FAILED particle %d window %d', self.n, self.i)

        self.ready = exit_code == 0
        if self.ready:
            self.state_dir = self.output_dir
        return self

    @property
    def fitness(self):
        """ Nash-Sutcliffe efficiency vs observed streamflow """
        if self._fitness is None:
            results = pd.read_csv(
                self.streamflow_path,
                delim_whitespace=True,
                skiprows=2,
                header=None,
                names=['datetime', 'dhsvm'],
                index_col='datetime'
            )
            results.index = pd.to_datetime(
                results.index, format='%m.%d.%Y-%H:%M:%S')
            results = results.join(self.obs, how='inner')
            self._fitness = 1 - (
                np.sum((results.dhsvm - results.observed) ** 2) /
                np.sum((results.observed - np.mean(results.observed)) ** 2)
            )
        return self._fitness


class Swarm():

    def __init__(self, nparticle, bins, dates, obs, jar, dhsvm, c):
        self.nparticle = nparticle
        self.bins = bins
        for param, lst in self.bins.items():
            lst.sort()

        self.start_dates = dates[:-1]
        self.end_dates = dates[1:]
        self.i = 0

        self.obs = obs
        self.jar = jar
        self.dhsvm = dhsvm
        self.c = c

        self._build_clim_sigma()
        self.initialize_particles()

        self.log = os.path.abspath('particles.csv')
        with open(self.log, 'a') as log_file:
            writer = csv.writer(log_file)
            writer.writerow(
                ['window', 'particle', 'weight', 'nse'] +
                [param for param in bins.keys()])

    def _build_clim_sigma(self):
        """ 
        Climatological sigma = 
            c * mean(obs within ±15 days of day-of-year)
        """
        doy = self.obs.index.day_of_year
        self.clim_sigma = {}
        for d in range(1, 367):
            diff = np.abs(doy - d)
            diff = np.minimum(diff, 366 - diff)
            mask = diff <= 15
            mean_q = self.obs.loc[mask, 'observed'].mean()
            self.clim_sigma[d] = self.c * mean_q if mean_q > 0 else self.c

    def initialize_particles(self):
        """ Latin Hypercube sample of parameter space """
        self.particles = []

        ranges = collections.OrderedDict()
        for k, b in self.bins.items():
            r = [(b[i], b[i + 1]) for i, _ in enumerate(b[:-1])]
            random.shuffle(r)
            ranges[k] = r

        n = 0
        while any(ranges.values()):
            position = np.array(
                [random.uniform(*r.pop()) for r in ranges.values()])
            particle = Particle(
                self.bins, position, n, self.obs, self.start, self.end,
                self.dhsvm)
            logging.debug('Created particle: %s', particle)
            self.particles.append(particle)
            n += 1

    def next(self):
        if self.complete:
            logging.info('Calibration complete after %d windows', self.i)
            return False

        logging.debug('Starting window %d', self.i + 1)
        logging.debug('START DATE: %s', self.start.isoformat())
        logging.debug('END DATE: %s', self.end.isoformat())

        pool = mp.Pool(self.nparticle)
        self.particles = pool.map(self.run_particle_window, self.particles)
        pool.close()
        pool.join()

        self.compute_weights()
        self.resample()
        self.perturb()
        self.i += 1
        self.pickle()
        return True

    def run_particle_window(self, particle):
        try:
            particle.configure(self.start, self.end)
        except Exception as e:
            particle.ready = False
            particle._fitness = -float('inf')
            logging.exception('Configure failed for particle %d', particle.n)
            if getattr(particle, 'output_dir', None):
                with open(particle.summary_path, 'w') as summary_file:
                    summary_file.write('Configuration failed: {}\n'.format(e))
            return particle
        return particle.run_next()

    @property
    def nwindow(self):
        return len(self.start_dates)

    @property
    def complete(self):
        return self.i >= self.nwindow

    @property
    def start(self):
        return self.start_dates[self.i]

    @property
    def end(self):
        return self.end_dates[self.i]

    def _log_likelihood(self, particle):
        """ Gaussian log-likelihood with climatological sigma """
        if not particle.ready:
            return -np.inf
        try:
            results = pd.read_csv(
                particle.streamflow_path,
                delim_whitespace=True,
                skiprows=2,
                header=None,
                names=['datetime', 'dhsvm'],
                index_col='datetime'
            )
            results.index = pd.to_datetime(
                results.index, format='%m.%d.%Y-%H:%M:%S')
            results = results.join(self.obs, how='inner')
            if results.empty:
                return -np.inf
            sigma = np.array([
                self.clim_sigma.get(d, self.c)
                for d in results.index.day_of_year
            ])
            return -0.5 * np.sum(
                ((results.dhsvm.values - results.observed.values) / sigma) ** 2
            )
        except Exception as e:
            logging.error('Log-likelihood failed for particle %d: %s',
                          particle.n, e)
            return -np.inf

    def compute_weights(self):
        """ Compute normalized importance weights from log-likelihoods """
        log_w = np.array([self._log_likelihood(p) for p in self.particles])
        logging.debug('Log-weights: %s', log_w)

        finite_mask = np.isfinite(log_w)
        if not finite_mask.any():
            logging.error('All particles have -inf log-likelihood')
            w = np.ones(len(self.particles)) / len(self.particles)
        else:
            log_w_shifted = log_w - log_w[finite_mask].max()
            w = np.exp(log_w_shifted)
            w[~finite_mask] = 0.0
            w /= w.sum()

        for p, wi in zip(self.particles, w):
            p.weight = wi

        with open(self.log, 'a') as log_file:
            writer = csv.writer(log_file)
            for p in self.particles:
                writer.writerow(
                    [self.i, p.n, p.weight, p.fitness] + list(p.position))

        eff_n = 1.0 / np.sum(w ** 2) if np.sum(w ** 2) > 0 else 0
        logging.info('WINDOW %d effective N: %.1f / %d',
                     self.i, eff_n, len(self.particles))

    def resample(self):
        """ Systematic resampling proportional to weights """
        N = len(self.particles)
        weights = np.array([p.weight for p in self.particles])
        cumsum = np.cumsum(weights)
        u = (np.arange(N) + random.random()) / N
        indices = np.clip(np.searchsorted(cumsum, u), 0, N - 1)

        new_positions = [self.particles[i].position.copy() for i in indices]
        for p, pos in zip(self.particles, new_positions):
            p.position = pos
            p.ready = True

    def perturb(self):
        """ Add Gaussian noise scaled to 10% of each parameter's range """
        pmin = np.array([min(b) for b in self.bins.values()])
        pmax = np.array([max(b) for b in self.bins.values()])
        sigma = 0.1 * (pmax - pmin)
        for p in self.particles:
            p.position = np.clip(
                p.position + np.random.normal(0, sigma), pmin, pmax)

    def pickle(self):
        """ Save state """
        with open(self.jar, 'wb') as jarfile:
            pickle.dump(self, jarfile)

    def sync_progress(self):
        """Recover progress from particles in jars saved before i was updated."""
        completed = max(
            [getattr(p, 'i', 0) for p in self.particles] + [self.i])
        if completed > self.i:
            logging.warning(
                'Advancing swarm progress from %d to %d completed windows',
                self.i, completed)
            self.i = completed
            self.pickle()


if __name__ == '__main__':
    nparticle = int(sys.argv[1])
    nbins = nparticle + 1
    max_time = int(sys.argv[2])
    window = sys.argv[3]
    start_date = sys.argv[4]
    end_date = sys.argv[5]
    bins_csv = sys.argv[6]
    dhsvm = sys.argv[7]
    c = float(sys.argv[8])

    start_time = time.time()
    jar = 'swarm.jar'
    obs = pd.read_csv(
        'discharge.tsv',
        sep='\t',
        comment='#',
        skiprows=34,
        header=None,
        usecols=[2, 4],
        names=['datetime', 'observed'],
        index_col='datetime',
        parse_dates=['datetime']
    )
    obs.observed = obs.observed / 35.31 * 3600  # cfs to m^3/h
    mainlog = os.path.abspath('smoother.log')
    logging.basicConfig(
        filename=mainlog,
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info('Starting calibration c=%.3f', c)
    bins_df = pd.read_csv(bins_csv)
    bins = collections.OrderedDict()
    for _, row in bins_df.iterrows():
        if row['scale'] == 'log':
            bins[row['name']] = list(np.logspace(
                np.log10(row['min']), np.log10(row['max']), num=nbins))
        else:
            bins[row['name']] = list(np.linspace(
                row['min'], row['max'], num=nbins))
    dates = pd.date_range(start_date, end_date, freq=window)
    end_timestamp = pd.Timestamp(end_date)
    if dates[-1] < end_timestamp:
        dates = dates.append(pd.DatetimeIndex([end_timestamp]))
    if os.path.exists(jar):
        with open(jar, 'rb') as jarfile:
            swarm = pickle.load(jarfile)
        swarm.sync_progress()
    else:
        swarm = Swarm(nparticle, bins, dates, obs, jar, dhsvm, c)

    wall = time.time() - start_time
    while wall < max_time and not swarm.complete:
        logging.debug('Wall time: %s', wall)
        swarm.next()
        wall = time.time() - start_time

    if swarm.complete:
        logging.info('Calibration stopped: reached final window ending %s',
                     swarm.end_dates[-1].isoformat())
