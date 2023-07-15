"""
Simulation model prototype for fork-join queueing system (:mod:`pyqumo.simulations.forkjoin.sandbox.model`)
===========================================================================================================

Simple simulation model for a fork-join queueing system written in pure python.

Summary
-------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    simulate_forkjoin

Details
-------



"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from heapq import heappush, heappop
from typing import Optional, List, Sequence, Dict, Tuple

import numpy as np

from pyqumo import Distribution, CountableDistribution
from pyqumo.stats import TimeSizeRecords, build_statistics
from ..contract import ForkJoinResults


class _Event(Enum):
    ARRIVAL = 0
    SERVICE_END = 1


class _State:
    def __init__(self, n: int):
        self.served_map: Dict[int, int] = {}  # packet_id -> (num served parts)
        self.arrived_at: Dict[int, float] = {}  # packet_id -> (arrival time)

        # queues[i] stores a list of packet_ids currently in the queue
        self.queues: List[List[int]] = [list() for _ in range(n)]

        # servers[i] stores packet_id that is served at node #i.
        # When server is free, store None.
        self.servers: List[Optional[int]] = [None] * n


@dataclass
class _Params:
    num_servers: int
    services: List[Distribution]
    capacities: List[Optional[int]]


class _Statistics:
    def __init__(self, n: int):
        # system_size[i] is the distribution of the i-th system size
        self.system_size = [TimeSizeRecords() for _ in range(n)]

        self.num_generated = 0  # total number of generated packets
        self.drops = 0  # total number of dropped packets
        self.num_served = 0  # total number of packets served completely

        # queue_drops[i] is the number of packets dropped because this
        # i-th queue was full
        self.queue_drops = [0 for _ in range(n)]

        # queue_num_served[i] is the number of sub-packets served by the queue
        self.queue_num_served = [0 for _ in range(n)]

        # Stores time from packet arrival to its complete serve
        self.responses: List[float] = []

        # queue_responses[i] stores time from packet arrival to i-th queue
        # till the i-th part of packet was served
        self.queue_responses: List[List[float]] = [list() for _ in range(n)]


_EventQueue = List[Tuple[float, _Event, Optional[int]]]


def simulate_forkjoin(
        arrival: Distribution,
        services: Sequence[Distribution],
        capacities: Sequence[Optional[int]],
        max_time: float = np.inf,
        max_packets: int = 1000000
) -> ForkJoinResults:
    """
    Run simulation model of fork-join model.

    Simulation can be stopped in two ways: by reaching maximum simulation time,
    or by reaching the maximum number of generated packets. By default,
    simulation is limited with the maximum number of packets only (1 million).

    Queues may have finite or infinite capacities. To indicate the queue
    is infinite, pass np.inf or None.

    Arrival and service time processes can be of any kind, including Poisson
    or MAP. To use a PH or normal distribution, a GenericIndependentProcess
    model with the corresponding distribution may be used.

    **Example 1:** model M/M/1 system

    >>> from pyqumo.randoms import Exponential as Exp
    >>> from pyqumo.simulations.forkjoin.sandbox.model import simulate_forkjoin
    >>> ret = simulate_forkjoin(Exp(1), [Exp(2)], [None],
    >>>                         max_packets=1000000)
    >>> print(ret.tabulate())

    **Example 2:** system with two queues, one infinite and another - finite:

    >>> ret = simulate_forkjoin(
    >>>             Exp(1), [Exp(2), Exp(2)], [None, 2], max_packets=1000000)
    >>> print(ret.tabulate())

    **Example 3:** complex system with different service times:

    >>> ret = simulate_forkjoin(
    >>>             Exp(1), [Exp(2), Exp(3), Exp(1.5), Exp(5)],
    >>>             [None, 2, 4, None], max_packets=1000000)
    >>> print(ret.tabulate())

    Parameters
    ----------
    arrival : RandomProcess
        Arrival random process.
    services : Sequence[RandomProcess]
        Service times distribution, must have the same length as ``capacities``
    capacities : Sequence[int | None]
        Capacities of the servers' queues. For infinite queue, pass None.
    max_time : float, optional
        Maximum simulation time (default: infinity).
    max_packets
        Maximum number of simulated packets (default: 1'000'000)

    Returns
    -------
    results : Results
        Simulation results.
    """
    started_simulation_at = datetime.now()
    n = len(services)
    if len(capacities) != n:
        raise ValueError("size of services and capacities must be equal")

    evq: _EventQueue = []  # (time, event, [server]), use heappush|heappop

    state = _State(n)
    params = _Params(num_servers=n, services=list(services),
                     capacities=list(capacities))
    statistics = _Statistics(n)
    next_packet_id = 1
    time = 0

    # Initialize the model:
    heappush(evq, (arrival(), _Event.ARRIVAL, -1))

    # Run main simulation loop:
    while evq:
        # Extract next event, update current time and stop simulation,
        # if time reached the boundary
        event_time, event_type, server_index = heappop(evq)
        if event_time > max_time:
            time = max_time
            break
        else:
            time = event_time

        if event_type == _Event.ARRIVAL:
            # If generated too many packets, stop
            if statistics.num_generated >= max_packets:
                break

            # Generate a new packet
            packet_id = next_packet_id
            next_packet_id += 1
            statistics.num_generated += 1

            # Check there is place for a packet.
            can_serve = True
            for i in range(params.num_servers):
                if ((state.servers[i] is not None) and
                        (params.capacities[i] is not None)
                        and (len(state.queues[i]) >= params.capacities[i])):
                    can_serve = False
                    statistics.queue_drops[i] += 1

            if can_serve:
                # If there is place for a packet, either start serving it,
                # or add to a queue. Also record the packet arrival time.
                state.served_map[packet_id] = 0
                state.arrived_at[packet_id] = time
                for si in range(n):
                    if state.servers[si] is None:
                        end_at = time + params.services[si]()
                        state.servers[si] = packet_id
                        heappush(evq, (end_at, _Event.SERVICE_END, si))
                        statistics.system_size[si].add(time, 1)
                    else:
                        queue = state.queues[si]
                        queue.append(packet_id)
                        statistics.system_size[si].add(time, len(queue) + 1)
            else:
                # If any queue is full, drop the packet.
                statistics.drops += 1

            # Finally, schedule next arrival
            heappush(evq, (time + arrival(), _Event.ARRIVAL, -1))

        elif event_type == _Event.SERVICE_END:
            # Extract the packet from the server:
            packet_id = state.servers[server_index]

            # Record the number of served packets by this queue:
            statistics.queue_num_served[server_index] += 1

            # Mark the packet part being served and record response time.
            # If the packet was completely served, record total response time
            # and remove the packet from indexes:
            response_time = time - state.arrived_at[packet_id]
            statistics.queue_responses[server_index].append(response_time)
            state.served_map[packet_id] += 1
            if state.served_map[packet_id] >= n:
                statistics.responses.append(response_time)
                statistics.num_served += 1
                del state.arrived_at[packet_id]
                del state.served_map[packet_id]

            if len(state.queues[server_index]) > 0:
                # Queue is not empty, so take the next packet and start serving
                # it. Also update system size statistics.
                packet_id = state.queues[server_index][0]
                state.servers[server_index] = packet_id
                state.queues[server_index] = state.queues[server_index][1:]

                # Schedule next service end:
                end_at = time + params.services[server_index]()
                heappush(evq, (end_at, _Event.SERVICE_END, server_index))

                # Record statistics:
                statistics.system_size[server_index].add(
                    time, len(state.queues[server_index]) + 1)
            else:
                # Queue is empty - mark server as empty and record statistics.
                state.servers[server_index] = None
                statistics.system_size[server_index].add(time, 0)

    # Build results from the collected statistics:
    elapsed = datetime.now() - started_simulation_at
    return _build_results(n, statistics, time, elapsed)


def _build_results(
        num_servers: int,
        statistics: _Statistics,
        model_time: int,
        elapsed: timedelta
) -> ForkJoinResults:
    results = ForkJoinResults(num_servers)
    results.real_time = elapsed.seconds + 1e-6 * elapsed.microseconds
    results.model_time = model_time
    results.response_time = build_statistics(statistics.responses)
    results.packet_drop_prob = \
        statistics.drops / (statistics.num_served + statistics.drops)

    for i in range(num_servers):
        # Take system size probability mass function (PMF) and extract from it:
        # - busy rate (server is free <=> system is empty)
        # - queue sizes (system size - 1 when system is not empty, o0 otherwise)
        # - system sizes
        pmf = statistics.system_size[i].pmf
        p0 = pmf[0]
        p1 = pmf[1] if len(pmf) > 0 else 0

        busy_pmf = [p0, sum(pmf[1:])]
        queue_size_pmf = [p0+p1, *pmf[2:]]

        results.busy.append(CountableDistribution(busy_pmf))
        results.queue_sizes.append(CountableDistribution(queue_size_pmf))
        results.system_sizes.append(CountableDistribution(pmf))

        # Record drop ratio and response time for each queue:
        results.queue_response_times.append(
            build_statistics(statistics.queue_responses[i]))
        results.queue_drop_probs.append(
            statistics.queue_drops[i] /
            (statistics.queue_drops[i] + statistics.queue_num_served[i]))

    return results
