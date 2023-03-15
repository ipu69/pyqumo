/**
 * A top-level module with classes and routines for launching simulations.
 *
 * > This module is intended for integrating in Cython or calling from main().
 *
 * Two data types for results representation are defined here:
 *
 * - NodeData: node-specific performance metrics;
 * - SimData: collection of NodeData and overall performance metrics. Objects
 *      of this type are returned by all simulation routines.
 *
 * In contrast to journals (@see Journals.h) these types contain only estimated
 * results. They also don't operate with pointers and behave more like plain
 * C-structs with several additional methods. That's done to make it more simple
 * to integrate these types with Cython and don't bother with free()/delete.
 *
 * These data types are not intended for intermediate usage, so they are not
 * derived from Object base class.
 *
 * Simulation routines for different experiments:
 *
 * - simMM1(): simulate a basic queueing system M/M/1/N or M/M/1
 * - simGG1(): simulate a general queueing system G/G/1/N or G/G/1
 *
 * @author Andrey Larionov
 */
#ifndef CQUMO_TANDEM_SIMULATION_H
#define CQUMO_TANDEM_SIMULATION_H

#include <vector>

#include "Statistics.h"
#include "Components.h"
#include "System.h"


#define MAX_PACKETS 10000


namespace cqumo {

/**
 * Struct with performance metric of a network node. Defines the following
 * metrics:
 *
 * - statistics and PMF of the system size
 * - statistics and PMF of the queue size
 * - statistics and PMF of the server size (e.g. mean value is busy rate)
 * - statistics of the delivery delays of the packets generated by this node
 * - statistics of the intervals between consequent departures
 * - statistics of the waiting times (time spent by the packets in the queue)
 * - statistics of the response times (time spent in the queue and the server)
 * - number of packets, generated at this node
 * - number of packets, generated at this node and delivered
 * - number of packets, generated at this node and lost somewhere
 * - number of packets, arrived at this node
 * - number of packets, served by this node
 * - number of packets, dropped by this node
 * - estimated probability that the generated packet will be lost
 * - estimated probability that the arrived packet will be dropped
 * - estimated probability that the generated packet will be delivered
 *
 * All size statistics are represented with SizeDist objects (@see SizeDist).
 * These objects contain probability mass function and routines for
 * mean, standard deviation, variance and moments estimation.
 *
 * Intervals statistics are carried in VarData objects (@see VarData).
 * They contain only information about the first N moments, mean value,
 * standard deviation, variance and the number of samples used in estimation.
 *
 * The structure can be built from NodeJournal instance. It doesn't contain
 * any dynamically allocated parts, it can be safely copied.
 */
struct NodeData {
    SizeDist systemSize;
    SizeDist queueSize;
    SizeDist serverSize;
    VarData delays;
    VarData departures;
    VarData waitTime;
    VarData responseTime;
    unsigned numPacketsGenerated = 0;
    unsigned numPacketsDelivered = 0;
    unsigned numPacketsLost = 0;
    unsigned numPacketsArrived = 0;
    unsigned numPacketsServed = 0;
    unsigned numPacketsDropped = 0;
    double lossProb = 0.0;
    double dropProb = 0.0;
    double deliveryProb = 0.0;

    NodeData() = default;
    NodeData(const NodeData &other) = default;

    explicit NodeData(const NodeJournal &records);

    NodeData &operator=(const NodeData &other) = default;
};


/**
 * Struct with simulation results. Contains a mapping of node address to
 * NodeData structure with node-specific performance metrics. Also provides
 * general simulation metrics:
 *
 * - total number of generated packets
 * - model (simulation) clock time, that was reached when simulation stopped
 * - real time from simulation start till end in milliseconds
 */
struct SimData {
    std::map<int, NodeData> nodeData;   ///< Mapping of Node address to NodeData
    unsigned numPacketsGenerated = 0;   ///< Total number of generated packets
    double simTime = 0.0;               ///< Model time when simulation finished
    double realTimeMs = 0.0;            ///< Real time in milliseconds

    SimData() = default;
    SimData(const SimData &other) = default;

    /**
     * Create SimData from NetworkJournal.
     * @param journal NetworkJournal instance
     * @param simTime time on model clock
     * @param realTimeMs real time spent on simulation
     */
    SimData(const NetworkJournal &journal, double simTime, double realTimeMs);

    SimData &operator=(const SimData &other) = default;
};


/**
 * Simulate M/M/1 or M/M/1/N queuing system.
 *
 * @param arrivalRate inverse of mean arrival interval ('lambda')
 * @param serviceRate inverse of mean service duration ('mu')
 * @param queueCapacity if non-negative, queue capacity, otherwise queue
 *          is supposed to be infinite (default)
 * @param maxPackets simulation will finish when this number of packets
 *          will be generated (default: MAX_PACKETS macro)
 * @return SimData
 */
SimData simMM1(
        double arrivalRate,
        double serviceRate,
        int queueCapacity = -1,
        int maxPackets = MAX_PACKETS);

/**
 * Simulate G/G/1 or G/G/1/N queueing system.
 *
 * @param arrival function with signature `() -> double` for arrival intervals
 * @param service function with signature `() -> double` for service durations
 * @param queueCapacity if non-negative, queue capacity, otherwise queue
 *          is supposed to be infinite (default)
 * @param maxPackets simulation will finish when this number of packets
 *          will be generated (default: MAX_PACKETS macro)
 * @return SimData
 */
SimData simGG1(
        const DblFn &arrival,
        const DblFn &service,
        int queueCapacity = -1,
        int maxPackets = MAX_PACKETS);

SimData simTandem(
    const std::map<int,DblFn>& arrivals,
    const std::vector<DblFn>& services,
    int queueCapacity = -1,
    bool fixedService = false,
    int maxPackets = MAX_PACKETS);

}

#endif //CQUMO_TANDEM_SIMULATION_H
