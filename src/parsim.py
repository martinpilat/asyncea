import heapq

from collections import namedtuple, Counter

RunningTask = namedtuple('RunningTask', ['end_time', 'cpu', 'value', 'args', 'start_time'])
TaskResult = namedtuple('TaskResult', ['value', 'args'])
PriorityTask = namedtuple('Priority', ['priority', 'args'])

class ParallelSimulator:

    def __init__(self, n_cpus, eval_func, time_func):
        self.submitted = 0
        self.n_cpus = n_cpus
        self.eval_func = eval_func
        self.time_func = time_func
        self.free_cpus = set(range(n_cpus))
        self.queue = []
        heapq.heapify(self.queue)
        self.current_time = 0
        self.computing = []
        heapq.heapify(self.computing)
        self.evals = 0

        self.total_eval_time = 0

        self.log = []
        self.best_fitness = 1e6

    def _create_log_info(self):
        gens = Counter(map(lambda x: x.args[0].gen if hasattr(x.args[0], 'gen') else 0, self.computing))
        return dict(time=self.current_time, free_cpus=len(self.free_cpus),
                    computing_gens=gens, best_fitness=self.best_fitness,
                    evals=self.evals, total_eval_time=self.total_eval_time)

    def _update_log(self):
        log_info = self._create_log_info()
        if self.log and self.log[-1]['time'] == log_info['time']:
            self.log[-1] = log_info             # multiple things happened now, record last state
        else:
            self.log.append(log_info)

    def _start_computing(self, *args):
        cpu = self.free_cpus.pop()
        value = self.eval_func(*args)
        time = self.time_func(value, *args)
        heapq.heappush(self.computing, RunningTask(self.current_time + time, cpu, value, args, self.current_time))
        self._update_log()

    def submit(self, *args, order=None):
        if order is None:
            order = ()
        order += (self.submitted,)
        # have a free CPU, start computing immediately
        task = PriorityTask(order, args)
        if self.free_cpus:
            self._start_computing(*task.args)
        else:
            heapq.heappush(self.queue, task)
        self.submitted += 1
        self._update_log()

    def next_finished(self):
        finished = heapq.heappop(self.computing)
        self.evals += 1
        self.current_time = finished.end_time
        self.total_eval_time += finished.end_time - finished.start_time
        self.free_cpus.add(finished.cpu)
        if self.queue:
            task = heapq.heappop(self.queue)
            self._start_computing(*task.args)
        self.best_fitness = min(self.best_fitness, finished.value[0])
        self._update_log()
        return TaskResult(finished.value, finished.args)

    def all_evaluations_finished(self):
        return len(self.free_cpus) == self.n_cpus and not self.queue

    def print_stats(self):
        print(f'function evaluations: {self.evals}')
        print(f'total evaluation time: {self.total_eval_time}')
        print(f'current time: {self.current_time}')
        print('average evaluation time:', self.total_eval_time/self.evals)
        print('cpu utilization:', self.total_eval_time/(self.n_cpus*self.current_time))

