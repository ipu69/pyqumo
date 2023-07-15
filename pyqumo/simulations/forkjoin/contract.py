from typing import List, Optional, Callable, Sequence

from tabulate import tabulate

from pyqumo import CountableDistribution, str_array, Distribution
from pyqumo.stats import Statistics


ForkJoinSimulate = Callable[
    [Distribution, Sequence[Distribution], Sequence[Optional[int]], float, int],
    "ForkJoinResults"]


class ForkJoinResults:
    """
    Results returned from a fork-join model.

    System-wide properties:

    - ``num_servers``: number of servers and queues
    - ``packet_drop_prob``: probability that a packet will be dropped
    - ``response_time``: time from packet arrival till the end of its service
    - ``model_time``: time in a simulation model when it finished execution
    - ``real_time``: elapsed time in seconds

    Per-queue properties:

    - ``system_sizes[i]``: countable distribution of the system size
    - ``queue_sizes[i]``: countable distributions of the queue size
    - ``busy[i]``: busy time distribution of the server
    - ``queue_drop_probs[i]``: probability that this queue drops the packet
    - ``queue_response_time[i]``: time from the packet arrives at this queue,
            till the sub-packet is served by this particular queue
    """
    def __init__(self, num_servers: int):
        self.num_servers = num_servers
        self.system_sizes: List[CountableDistribution] = []
        self.queue_sizes: List[CountableDistribution] = []
        self.busy: List[CountableDistribution] = []
        self.queue_drop_probs: List[float] = []
        self.packet_drop_prob: float = 0.0
        self.queue_response_times: List[Statistics] = []
        self.response_time: Optional[Statistics] = None
        self.model_time: float = 0.0
        self.real_time: float = 0.0

    def get_utilization(self, si: int) -> float:
        """
        Get utilization coefficient, that is `Busy = 1` probability.
        """
        return self.busy[si].pmf(1)

    def tabulate(self) -> str:
        """
        Build a pretty formatted table with all key properties.
        """
        items = [
            ('Time elapsed', self.real_time),
            ('Model time', self.model_time),
            ('Number of stations', self.num_servers),
            ('Packet loss prob.', self.packet_drop_prob),
            ('Response time, average', self.response_time.avg),
            ('Response time, std.dev.', self.response_time.std),
        ]

        for si in range(self.num_servers):
            # noinspection DuplicatedCode
            items.append((f'[[ QUEUE #{si} ]]', ''))

            ssize = self.system_sizes[si]
            qsize = self.queue_sizes[si]
            busy = self.busy[si]

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
                ('Utilization', self.get_utilization(si)),
                ('Queue drop prob.', self.queue_drop_probs[si]),
                ('Response time, average', self.queue_response_times[si].avg),
                ('Response time, std.dev.', self.queue_response_times[si].std)
            ])
        return tabulate(items, headers=('Param', 'Value'))
