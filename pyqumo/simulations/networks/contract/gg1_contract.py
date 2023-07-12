from dataclasses import dataclass
from typing import Optional

from tabulate import tabulate

from pyqumo import CountableDistribution, str_array
from pyqumo.stats import Statistics


@dataclass
class GG1Results:
    """
    Results returned from G/G/1/N model simulation.
    """
    system_size: Optional[CountableDistribution] = None
    queue_size: Optional[CountableDistribution] = None
    busy: Optional[CountableDistribution] = None
    loss_prob: float = 0.0
    departures: Optional[Statistics] = None
    response_time: Optional[Statistics] = None
    wait_time: Optional[Statistics] = None
    real_time: Optional[float] = 0.0

    @property
    def utilization(self) -> float:
        """
        Get utilization coefficient, that is `Busy = 1` probability.
        """
        return self.busy.pmf(1)

    def tabulate(self) -> str:
        """
        Build a pretty formatted table with all key properties.
        """
        system_size_pmf = [
            self.system_size.pmf(x)
            for x in range(self.system_size.truncated_at + 1)]
        queue_size_pmf = [
            self.queue_size.pmf(x)
            for x in range(self.queue_size.truncated_at + 1)]

        items = [
            ('System size PMF', str_array(system_size_pmf)),
            ('System size average', self.system_size.mean),
            ('System size std.dev.', self.system_size.std),
            ('Queue size PMF', str_array(queue_size_pmf)),
            ('Queue size average', self.queue_size.mean),
            ('Queue size std.dev.', self.queue_size.std),
            ('Utilization', self.utilization),
            ('Loss probability', self.loss_prob),
            ('Departures, average', self.departures.avg),
            ('Departures, std.dev.', self.departures.std),
            ('Response time, average', self.response_time.avg),
            ('Response time, std.dev.', self.response_time.std),
            ('Wait time, average', self.wait_time.avg),
            ('Wait time, std.dev.', self.wait_time.std),
            ("Execution time, ms.", self.real_time),
        ]
        return tabulate(items, headers=('Param', 'Value'))
