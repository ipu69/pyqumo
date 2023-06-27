from typing import List

from tabulate import tabulate

from pyqumo import str_array
from pyqumo.randoms import CountableDistribution
from pyqumo.stats import Statistics


class GG1TandemResults:
    """
    Results returned from G/G/1/N model simulation.

    Discrete stochastic properties like system size, queue size and busy
    periods are represented with `CountableDistribution`. Continuous properties
    are not fitted into any kind of distribution, they are represented with
    `Statistics` tuples.

    Utilization coefficient, as well as loss probability, are just floating
    point numbers.

    To pretty print the results one can make use of `tabulate()` method.
    """
    def __init__(self, num_stations: int):
        """
        Create results.

        Parameters
        ----------
        num_stations: int, optional
        """
        self._num_stations = num_stations
        self.real_time = 0.0
        self.system_size: List[CountableDistribution] = []
        self.queue_size: List[CountableDistribution] = []
        self.busy: List[CountableDistribution] = []
        self.drop_prob: List[float] = []
        self.delivery_prob: List[float] = []
        self.departures: List[Statistics] = []
        self.arrivals: List[Statistics] = []
        self.wait_time: List[Statistics] = []
        self.response_time: List[Statistics] = []
        self.delivery_delays: List[Statistics] = []


    def get_utilization(self, node: int) -> float:
        """
        Get utilization coefficient, that is `Busy = 1` probability.
        """
        return self.busy[node].pmf(1)

    def tabulate(self) -> str:
        """
        Build a pretty formatted table with all key properties.
        """
        items = [
            ('Number of stations', self._num_stations),
            ('Loss probability', self.drop_prob),
        ]

        for node in range(self._num_stations):
            items.append((f'[[ STATION #{node} ]]', ''))

            ssize = self.system_size[node]
            qsize = self.queue_size[node]
            busy = self.busy[node]

            ssize_pmf = [ssize.pmf(x) for x in range(ssize.truncated_at + 1)]
            qsize_pmf = [qsize.pmf(x) for x in range(qsize.truncated_at + 1)]
            busy_pmf = [busy.pmf(x) for x in range(busy.truncated_at + 1)]

            items.extend([
                ('System size PMF', str_array(ssize_pmf)),
                ('System size average', ssize.mean),
                ('System size std.dev.', ssize.std),
                ('Queue size PMF', str_array(qsize_pmf)),
                ('Queue size average', qsize.mean),
                ('Queue size std.dev.', qsize.std),
                ('Busy PMF', str_array(busy_pmf)),
                ('Utilization', self.get_utilization(node)),
                ('Drop probability', self.drop_prob[node]),
                ('Delivery probability', self.delivery_prob[node]),
                ('Departures, average', self.departures[node].avg),
                ('Departures, std.dev.', self.departures[node].std),
                ('Response time, average', self.response_time[node].avg),
                ('Response time, std.dev.', self.response_time[node].std),
                ('Wait time, average', self.wait_time[node].avg),
                ('Wait time, std.dev.', self.wait_time[node].std),
                ('End-to-end delays, average', self.delivery_delays[node].avg),
                ('End-to-end delays, std.dev.', self.delivery_delays[node].std),
            ])
        return tabulate(items, headers=('Param', 'Value'))
