from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyqumo import Distribution, Exponential as Exp, Poisson as Po
from pyqumo.simulations.forkjoin.contract import ForkJoinSimulate
from pyqumo.simulations.forkjoin.sandbox.model import simulate_forkjoin


@dataclass
class FjProps:
    arrival: Distribution
    services: List[Distribution]
    capacities: List[Optional[int]]

    # System and queue sizes:
    system_size_avg: float
    system_size_std: float
    queue_size_avg: float
    queue_size_std: float

    # Loss probability and utilization:
    loss_prob: float
    utilization: float

    # Response and wait time:
    response_time_avg: float

    # Test parameters:
    tol: float = 1e-1
    max_packets: int = int(1e6)


@pytest.mark.parametrize('props', [
    FjProps(
        arrival=Po(2), services=[Po(5)], capacities=[4],
        system_size_avg=0.642, system_size_std=0.981,
        queue_size_avg=0.2444, queue_size_std=0.6545,
        loss_prob=0.0062, utilization=0.3975, response_time_avg=0.323,
        max_packets=100_000),
    FjProps(
        arrival=Exp(42), services=[Exp(34)], capacities=[7],
        system_size_avg=5.3295, system_size_std=5.6015**0.5,
        queue_size_avg=4.3708, queue_size_std=5.2010**0.5,
        loss_prob=0.2239, utilization=0.9587,
        response_time_avg=0.163, max_packets=100_000),
    FjProps(
        arrival=Po(1), services=[Exp(2)], capacities=[None],
        system_size_avg=1, system_size_std=2.0**0.5,
        queue_size_avg=0.5, queue_size_std=1.25**0.5,
        loss_prob=0, utilization=0.5, response_time_avg=1.0,
        max_packets=100_000)
    ])
@pytest.mark.parametrize('simulate', [simulate_forkjoin])
def test_gg1(props: FjProps, simulate: ForkJoinSimulate):
    tol = props.tol
    results = simulate(
        props.arrival, props.services, props.capacities,
        np.inf, props.max_packets)
    desc = f"arrival: {props.arrival}, " \
           f"service: {props.services}, " \
           f"queue capacity: {props.capacities}"

    # Check system and queue sizes:
    assert_allclose(
        results.system_sizes[0].mean, props.system_size_avg, rtol=tol,
        err_msg=f"system size average mismatch ({desc})")
    assert_allclose(
        results.system_sizes[0].std, props.system_size_std, rtol=tol,
        err_msg=f"system size std.dev. mismatch ({desc})")
    assert_allclose(
        results.queue_sizes[0].mean, props.queue_size_avg, rtol=tol,
        err_msg=f"queue size average mismatch ({desc})")
    assert_allclose(
        results.queue_sizes[0].std, props.queue_size_std, rtol=tol,
        err_msg=f"queue size std.dev. mismatch ({desc})")

    # Loss probability and utilization:
    assert_allclose(
        results.packet_drop_prob, props.loss_prob, rtol=tol,
        err_msg=f"loss probability mismatch ({desc})")
    assert_allclose(
        results.queue_drop_probs[0], props.loss_prob, rtol=tol,
        err_msg=f"queue loss probability mismatch ({desc})")
    assert_allclose(
        results.get_utilization(0), props.utilization, rtol=tol,
        err_msg=f"utilization mismatch ({desc})")

    # Response time:
    assert_allclose(
        results.response_time.avg, props.response_time_avg, rtol=tol,
        err_msg=f"response time mismatch ({desc})")
    assert_allclose(
        results.queue_response_times[0].avg, props.response_time_avg, rtol=tol,
        err_msg=f"waiting time mismatch ({desc})")
