import copy
import time

import numpy as np  # engine for numerical computing
from scipy.stats import norm  # normal continuous random variable

from HCC.NDAs.MMES.es import ES  # abstract class of all Evolution Strategies (ES) classes
from HCC.NDAs.MMES.state import MMESBlockResult, MMESState


class MMES(ES):
    """Mixture Model-based Evolution Strategy (MMES).

    Parameters
    ----------
    problem : dict
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : dict
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'sigma'         - initial global step-size, aka mutation strength (`float`),
                * 'mean'          - initial (starting) point, aka mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'm'             - number of candidate direction vectors (`int`, default:
                  `2*int(np.ceil(np.sqrt(problem['ndim_problem'])))`),
                * 'c_c'           - learning rate of evolution path update (`float`, default:
                  `0.4/np.sqrt(problem['ndim_problem'])`),
                * 'ms'            - mixing strength (`int`, default: `4`),
                * 'c_s'           - learning rate of global step-size adaptation (`float`, default: `0.3`),
                * 'a_z'           - target significance level (`float`, default: `0.05`),
                * 'distance'      - minimal distance of updating evolution paths (`int`, default:
                  `int(np.ceil(1.0/options['c_c']))`),
                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default:
                  `4 + int(3*np.log(problem['ndim_problem']))`),
                * 'n_parents'     - number of parents, aka parental population size (`int`, default:
                  `int(options['n_individuals']/2)`).

    Examples
    --------
    Use the black-box optimizer `MMES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.mmes import MMES
       >>> problem = {'fitness_function': rosenbrock,  # to define problem arguments
       ...            'ndim_problem': 200,
       ...            'lower_boundary': -5.0*numpy.ones((200,)),
       ...            'upper_boundary': 5.0*numpy.ones((200,))}
       >>> options = {'max_function_evaluations': 500000,  # to set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3.0*numpy.ones((200,)),
       ...            'sigma': 3.0}  # global step-size may need to be tuned for optimality
       >>> mmes = MMES(problem, options)  # to initialize the optimizer class
       >>> results = mmes.optimize()  # to run the optimization/evolution process
       >>> print(f"MMES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       MMES: 500000, 2.6018

    For its correctness checking of Python coding, please refer to `this code-based repeatability report
    <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_mmes.py>`_
    for all details. For *pytest*-based automatic testing, please see `test_mmes.py
    <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/test_mmes.py>`_.

    Attributes
    ----------
    a_z           : `float`
                    target significance level.
    c_c           : `float`
                    learning rate of evolution path update.
    c_s           : `float`
                    learning rate of global step-size adaptation.
    distance      : `int`
                    minimal distance of updating evolution paths.
    m             : `int`
                    number of candidate direction vectors.
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    ms            : `int`
                    mixing strength.
    n_individuals : `int`
                    number of offspring, aka offspring population size.
    n_parents     : `int`
                    number of parents, aka parental population size.
    sigma         : `float`
                    final global step-size, aka mutation strength.

    References
    ----------
    He, X., Zheng, Z. and Zhou, Y., 2021.
    `MMES: Mixture model-based evolution strategy for large-scale optimization.
    <https://ieeexplore.ieee.org/abstract/document/9244595>`_
    IEEE Transactions on Evolutionary Computation, 25(2), pp.320-333.

    Please refer to the *official* Matlab version from Prof. He:
    https://github.com/hxyokokok/MMES
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        # set number of candidate direction vectors
        self.m = options.get('m', 2*int(np.ceil(np.sqrt(self.ndim_problem))))
        assert self.m > 0
        # set learning rate of evolution path
        self.c_c = options.get('c_c', 0.4/np.sqrt(self.ndim_problem))
        self.ms = options.get('ms', 4)  # mixing strength (l)
        assert self.ms > 0
        # set for paired test adaptation (PTA)
        self.c_s = options.get('c_s', 0.3)  # learning rate of global step-size adaptation
        self.a_z = options.get('a_z', 0.05)  # target significance level
        # set minimal distance of updating evolution paths (T)
        self.distance = options.get('distance', int(np.ceil(1.0/self.c_c)))
        # set success probability of geometric distribution (different from 4/n in the original paper)
        self.c_a = options.get('c_a', 3.8/self.ndim_problem)  # same as the official Matlab code
        self.gamma = options.get('gamma', 1.0 - np.power(1.0 - self.c_a, self.m))
        self._n_mirror_sampling = None
        self._z_1 = np.sqrt(1.0 - self.gamma)
        self._z_2 = np.sqrt(self.gamma/self.ms)
        self._p_1 = 1.0 - self.c_c
        self._p_2 = np.sqrt(self.c_c*(2.0 - self.c_c))
        self._w_1 = 1.0 - self.c_s
        self._w_2 = np.sqrt(self.c_s*(2.0 - self.c_s))

    def initialize(self, args=None, is_restart=False):
        self._n_mirror_sampling = int(np.ceil(self.n_individuals/2))
        x = np.zeros((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        p = np.zeros((self.ndim_problem,))  # evolution path
        w = 0.0
        q = np.zeros((self.m, self.ndim_problem))  # candidate direction vectors
        t = np.zeros((self.m,))  # recorded generations
        v = np.arange(self.m)  # indexes to evolution paths
        y = np.tile(self._evaluate_fitness(mean, args), (self.n_individuals,))  # fitness
        return x, mean, p, w, q, t, v, y

    def iterate(self, x=None, mean=None, q=None, v=None, args=None):
        for k in range(self._n_mirror_sampling):
            zq = np.zeros((self.ndim_problem,))
            for _ in range(self.ms):
                j_k = v[(self.m - self.rng_optimization.geometric(self.c_a) % self.m) - 1]
                zq += self.rng_optimization.standard_normal() * q[j_k]
            z = self._z_1 * self.rng_optimization.standard_normal((self.ndim_problem,))
            z += self._z_2 * zq
            x[k] = mean + self.sigma * z
            if (self._n_mirror_sampling + k) < self.n_individuals:
                x[self._n_mirror_sampling + k] = mean - self.sigma * z

        if self._check_terminations():
            return x, y

        y = self._evaluate_fitness(x, args)

        return x, y

    def _update_distribution(self, x=None, mean=None, p=None, w=None, q=None,
                             t=None, v=None, y=None, y_bak=None):
        order = np.argsort(y)[:self.n_parents]
        y.sort()
        mean_w = np.dot(self._w[:self.n_parents], x[order])
        p = self._p_1*p + self._p_2*np.sqrt(self._mu_eff)*(mean_w - mean)/self.sigma
        mean = mean_w
        if self._n_generations < self.m:
            q[self._n_generations] = p
        else:
            k_star = np.argmin(t[v[1:]] - t[v[:(self.m - 1)]])
            k_star += 1
            if t[v[k_star]] - t[v[k_star - 1]] > self.distance:
                k_star = 0
            v = np.append(np.append(v[:k_star], v[(k_star + 1):]), v[k_star])
            t[v[-1]], q[v[-1]] = self._n_generations, p
        # conduct success-based mutation strength adaptation
        l_w = np.dot(self._w, y_bak[:self.n_parents] > y[:self.n_parents])
        w = self._w_1*w + self._w_2*np.sqrt(self._mu_eff)*(2*l_w - 1)
        self.sigma *= np.exp(norm.cdf(w) - 1.0 + self.a_z)
        return mean, p, w, q, t, v

    def restart_reinitialize(self, args=None, x=None, mean=None, p=None, w=None, q=None,
                             t=None, v=None, y=None, fitness=None):
        if self.is_restart and ES.restart_reinitialize(self, y):
            x, mean, p, w, q, t, v, y = self.initialize(args, True)
            self._print_verbose_info(fitness, y[0])
        return x, mean, p, w, q, t, v, y

    def _append_recent_best(self, recent_best):
        checkpoint = (
            int(self.n_function_evaluations),
            float(self.best_so_far_y),
        )
        if recent_best and recent_best[-1][0] == checkpoint[0]:
            recent_best[-1] = checkpoint
        else:
            recent_best.append(checkpoint)
        del recent_best[:-3]

    def _capture_state(
        self,
        x,
        mean,
        p,
        w,
        q,
        t,
        v,
        y,
        fitness,
        recent_best,
        pending_distribution_update=False,
        pending_y_bak=None,
    ):
        runtime = float(self.runtime)
        if self.start_time is not None:
            runtime = max(runtime, time.time() - self.start_time)
        state = MMESState(
            x=np.copy(x),
            mean=np.copy(mean),
            p=np.copy(p),
            w=float(w),
            q=np.copy(q),
            t=np.copy(t),
            v=np.copy(v),
            y=np.copy(y),
            sigma=float(self.sigma),
            n_individuals=int(self.n_individuals),
            n_parents=int(self.n_parents),
            n_mirror_sampling=int(self._n_mirror_sampling),
            n_generations=int(self._n_generations),
            n_restart=int(self._n_restart),
            list_generations=list(self._list_generations),
            list_fitness=[float(value) for value in self._list_fitness],
            list_initial_mean=[np.copy(value) for value in self._list_initial_mean],
            best_so_far_x=np.copy(self.best_so_far_x),
            best_so_far_y=float(self.best_so_far_y),
            n_function_evaluations=int(self.n_function_evaluations),
            termination_signal=int(self.termination_signal),
            fitness=[float(value) for value in fitness],
            recent_best=[(int(fe), float(best)) for fe, best in recent_best[-3:]],
            rng_initialization_state=copy.deepcopy(
                self.rng_initialization.bit_generator.state
            ),
            rng_optimization_state=copy.deepcopy(
                self.rng_optimization.bit_generator.state
            ),
            sigma_bak=float(self._sigma_bak),
            initial_mean=None if self.mean is None else np.copy(self.mean),
            counter_early_stopping=int(self._counter_early_stopping),
            base_early_stopping=float(self._base_early_stopping),
            printed_evaluations=int(self._printed_evaluations),
            time_function_evaluations=float(self.time_function_evaluations),
            runtime=runtime,
            pending_distribution_update=bool(pending_distribution_update),
            pending_y_bak=(
                None if pending_y_bak is None else np.copy(pending_y_bak)
            ),
        )
        state.validate()
        return state

    def _validate_state_compatibility(self, state):
        state.validate()
        if np.asarray(state.best_so_far_x).size != self.ndim_problem:
            raise ValueError("state dimension does not match optimizer")
        if np.asarray(state.q).shape[0] != self.m:
            raise ValueError("state direction count does not match optimizer")

    def _restore_state(self, state):
        self._validate_state_compatibility(state)
        self.sigma = float(state.sigma)
        self._sigma_bak = float(
            state.sigma if state.sigma_bak is None else state.sigma_bak
        )
        self.mean = np.copy(
            state.mean if state.initial_mean is None else state.initial_mean
        )
        self.n_individuals = int(state.n_individuals)
        self.n_parents = int(state.n_parents)
        self._n_mirror_sampling = int(state.n_mirror_sampling)
        self._w, self._mu_eff = self._compute_weights()
        self._n_generations = int(state.n_generations)
        self._n_restart = int(state.n_restart)
        self._list_generations = list(state.list_generations)
        self._list_fitness = [float(value) for value in state.list_fitness]
        self._list_initial_mean = [np.copy(value) for value in state.list_initial_mean]
        self.best_so_far_x = np.copy(state.best_so_far_x)
        self.best_so_far_y = float(state.best_so_far_y)
        self.n_function_evaluations = int(state.n_function_evaluations)
        self.termination_signal = int(state.termination_signal)
        self._counter_early_stopping = int(state.counter_early_stopping)
        self._base_early_stopping = float(state.base_early_stopping)
        self._printed_evaluations = int(state.printed_evaluations)
        self.time_function_evaluations = float(state.time_function_evaluations)
        self.runtime = float(state.runtime)
        self.start_time = time.time() - self.runtime
        self.rng_initialization.bit_generator.state = copy.deepcopy(
            state.rng_initialization_state
        )
        self.rng_optimization.bit_generator.state = copy.deepcopy(
            state.rng_optimization_state
        )
        return (
            np.copy(state.x),
            np.copy(state.mean),
            np.copy(state.p),
            float(state.w),
            np.copy(state.q),
            np.copy(state.t),
            np.copy(state.v),
            np.copy(state.y),
            list(state.fitness),
            list(state.recent_best),
            bool(state.pending_distribution_update),
            None if state.pending_y_bak is None else np.copy(state.pending_y_bak),
        )

    def initialize_state(self, fitness_function=None, args=None):
        fitness = ES.optimize(self, fitness_function)
        x, mean, p, w, q, t, v, y = self.initialize(args)
        self._print_verbose_info(fitness, y[0])
        recent_best = []
        self._append_recent_best(recent_best)
        return self._capture_state(
            x,
            mean,
            p,
            w,
            q,
            t,
            v,
            y,
            fitness,
            recent_best,
        )

    def optimize_with_state(self, fitness_function=None, args=None):
        fitness = ES.optimize(self, fitness_function)
        x, mean, p, w, q, t, v, y = self.initialize(args)
        self._print_verbose_info(fitness, y[0])
        recent_best = []
        self._append_recent_best(recent_best)
        pending_distribution_update = False
        pending_y_bak = None
        while not self.termination_signal:
            y_bak = np.copy(y)
            # sample and evaluate offspring population
            evaluations_before = self.n_function_evaluations
            x, y = self.iterate(x, mean, q, v, args)
            evaluated = self.n_function_evaluations > evaluations_before
            if evaluated:
                self._append_recent_best(recent_best)
            if self._check_terminations():
                pending_distribution_update = evaluated
                pending_y_bak = np.copy(y_bak) if evaluated else None
                break
            mean, p, w, q, t, v = self._update_distribution(x, mean, p, w, q, t, v, y, y_bak)
            self._n_generations += 1
            self._print_verbose_info(fitness, y)
            evaluations_before = self.n_function_evaluations
            x, mean, p, w, q, t, v, y = self.restart_reinitialize(
                args, x, mean, p, w, q, t, v, y, fitness)
            if self.n_function_evaluations > evaluations_before:
                self._append_recent_best(recent_best)
        state = self._capture_state(
            x,
            mean,
            p,
            w,
            q,
            t,
            v,
            y,
            fitness,
            recent_best,
            pending_distribution_update=pending_distribution_update,
            pending_y_bak=pending_y_bak,
        )
        results = self._collect(list(fitness), y, mean)
        results['p'] = p
        results['w'] = w
        return results, state

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        results, _state = self.optimize_with_state(fitness_function, args)
        return results

    def _complete_pending_distribution_update(
        self,
        x,
        mean,
        p,
        w,
        q,
        t,
        v,
        y,
        y_bak,
        fitness,
        recent_best,
        args=None,
    ):
        mean, p, w, q, t, v = self._update_distribution(
            x,
            mean,
            p,
            w,
            q,
            t,
            v,
            y,
            y_bak,
        )
        self._n_generations += 1
        self._print_verbose_info(fitness, y)
        evaluations_before = self.n_function_evaluations
        x, mean, p, w, q, t, v, y = self.restart_reinitialize(
            args,
            x,
            mean,
            p,
            w,
            q,
            t,
            v,
            y,
            fitness,
        )
        if self.n_function_evaluations > evaluations_before:
            self._append_recent_best(recent_best)
        return x, mean, p, w, q, t, v, y

    def run_block(self, state, additional_function_evaluations, args=None):
        self._validate_state_compatibility(state)
        fingerprint_before = state.fingerprint()
        requested_fes = max(0, int(additional_function_evaluations))
        best_before = float(state.best_so_far_y)
        if requested_fes < int(state.n_individuals):
            return MMESBlockResult(
                state=state.clone(),
                best_before=best_before,
                best_after=best_before,
                actual_fes=0,
                requested_fes=requested_fes,
                unused_fes=requested_fes,
                normalized_utility=0.0,
                termination_reason="insufficient_population_budget",
                state_fingerprint_before=fingerprint_before,
                state_fingerprint_after=fingerprint_before,
            )

        (
            x,
            mean,
            p,
            w,
            q,
            t,
            v,
            y,
            fitness,
            recent_best,
            pending_distribution_update,
            pending_y_bak,
        ) = self._restore_state(state)
        previous_limit = self.max_function_evaluations
        block_start = int(self.n_function_evaluations)
        block_limit = block_start + requested_fes
        self.max_function_evaluations = block_limit
        self.termination_signal = self.Terminations.NO_TERMINATION
        try:
            if pending_distribution_update:
                x, mean, p, w, q, t, v, y = (
                    self._complete_pending_distribution_update(
                        x,
                        mean,
                        p,
                        w,
                        q,
                        t,
                        v,
                        y,
                        pending_y_bak,
                        fitness,
                        recent_best,
                        args,
                    )
                )
                pending_distribution_update = False
                pending_y_bak = None
            while self.n_function_evaluations < block_limit:
                restart_reserve = 1 if self.is_restart else 0
                required = self.n_individuals + restart_reserve
                if self.n_function_evaluations + required > block_limit:
                    break
                y_bak = np.copy(y)
                evaluations_before = self.n_function_evaluations
                x, y = self.iterate(x, mean, q, v, args)
                evaluated = self.n_function_evaluations > evaluations_before
                if evaluated:
                    self._append_recent_best(recent_best)
                if self._check_terminations():
                    pending_distribution_update = evaluated
                    pending_y_bak = np.copy(y_bak) if evaluated else None
                    break
                mean, p, w, q, t, v = self._update_distribution(
                    x,
                    mean,
                    p,
                    w,
                    q,
                    t,
                    v,
                    y,
                    y_bak,
                )
                self._n_generations += 1
                self._print_verbose_info(fitness, y)
                evaluations_before = self.n_function_evaluations
                x, mean, p, w, q, t, v, y = self.restart_reinitialize(
                    args,
                    x,
                    mean,
                    p,
                    w,
                    q,
                    t,
                    v,
                    y,
                    fitness,
                )
                if self.n_function_evaluations > evaluations_before:
                    self._append_recent_best(recent_best)
            next_state = self._capture_state(
                x,
                mean,
                p,
                w,
                q,
                t,
                v,
                y,
                fitness,
                recent_best,
                pending_distribution_update=pending_distribution_update,
                pending_y_bak=pending_y_bak,
            )
        finally:
            self.max_function_evaluations = previous_limit

        actual_fes = int(next_state.n_function_evaluations) - block_start
        best_after = float(next_state.best_so_far_y)
        normalized_utility = max(0.0, best_before - best_after) / (
            max(abs(best_before), 1.0) * max(actual_fes, 1)
        )
        termination_reason = (
            "block_complete"
            if actual_fes > 0
            else "insufficient_population_budget"
        )
        return MMESBlockResult(
            state=next_state,
            best_before=best_before,
            best_after=best_after,
            actual_fes=actual_fes,
            requested_fes=requested_fes,
            unused_fes=max(0, requested_fes - actual_fes),
            normalized_utility=normalized_utility,
            termination_reason=termination_reason,
            state_fingerprint_before=fingerprint_before,
            state_fingerprint_after=next_state.fingerprint(),
        )

    def state_to_result(self, state):
        (
            _x,
            mean,
            p,
            w,
            _q,
            _t,
            _v,
            y,
            fitness,
            _recent_best,
            _pending_distribution_update,
            _pending_y_bak,
        ) = self._restore_state(state)
        results = self._collect(list(fitness), y, mean)
        results['p'] = p
        results['w'] = w
        return results
