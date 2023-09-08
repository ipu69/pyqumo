/**
 * Module provides journals for the whole network, as well as for
 * separate nodes. Journals store series and counters regarding various
 * performance metrics - queue and system sizes, end-to-end delays, number
 * of generated packets, etc. See NodeJournal for reference.
 *
 * @author Andrey Larionov
 */
#ifndef CQUMO_TANDEM_JOURNALS_H
#define CQUMO_TANDEM_JOURNALS_H

#include <map>

#include "../../statistics/statistics.h"


namespace cqumo {
namespace oqnet {

class Node;
class NetworkJournal;
class NodeJournal;


/**
 * Class that is used to store performance metrics about the network as a whole
 * and separate nodes journals.
 *
 * Right now, the only performance metric this journal provides is the
 * number of totally generated packets.
 *
 * Also, network journal stores settings for Series and TimeSizeSeries:
 *
 * - window size
 * - number of moments estimated
 * - initial model time
 *
 * These settings are used by NodeJournal objects when storing their performance
 * metrics using Series and TimeSizeSeries.
 */
class NetworkJournal {
  public:
    /**
     * Create network journal.
     * @param windowSize size of the window used in Series (default: 100)
     * @param numMoments number of moments estimated by Series (default: 4)
     * @param time initial model time used in TimeSizeSeries (default: 0.0)
     */
    explicit NetworkJournal(
            unsigned windowSize = 100,
            unsigned numMoments = 4,
            double time = 0.0);

    /** Destroy the network journal and all child node journals. */
    ~NetworkJournal();

    /** Create and store a NodeJournal for a given node. */
    void addNodeJournal(Node *node);

    /** Get the number of moments estimated by Series. */
    inline unsigned numMoments() const { return numMoments_; }

    /** Get the size of the sliding window used by Series. */
    inline unsigned windowSize() const { return windowSize_; }

    /** Get NodeJournal for a given node address, or nullptr if not found. */
    inline NodeJournal *nodeJournal(int address) const {
        auto iter = nodeRecordsMap_.find(address);
        return iter == nodeRecordsMap_.end() ? nullptr : iter->second;
    }

    /** Get a counter of the total number of generated packets. */
    inline Counter *numPacketsGenerated() const { return numPacketsGenerated_; }

    /** Get a mapping of node addresses to NodeJournal objects. */
    inline const std::map<int, NodeJournal*>&
    nodeJournals() const { return nodeRecordsMap_; }

    /**
     * Reset the journal. After this call all Series and TimeSizeSeries will
     * become empty, counters reset to 0. However, topology is not changed,
     * all NodeJournal objects will be kept, but also rest.
     * @param time initial model time used to initialize TimeSizeSeries
     */
    void reset(double time = 0.0);

    /**
     * Estimate all statistical properties of Series and TimeSizeSeries.
     * @see NodeJournal::commit() for reference.
     */
    void commit();

    /** Get string representation of the network journal. */
    std::string toString() const;

  private:
    unsigned windowSize_ = 100;
    unsigned numMoments_ = 4;
    double initTime_ = 0.0;
    std::map<int, NodeJournal*> nodeRecordsMap_;
    Counter *numPacketsGenerated_ = nullptr;
};


/**
 * Journal of various performance metrics of a single node. These metrics
 * are recorded:
 *
 * - system size (queue size + server size)
 * - queue size
 * - server size
 * - end-to-end delays of packets generated by this node till delivery
 * - departures intervals, e.g. intervals between consequent service ends
 * - waiting times, e.g. times spent by packets before start of the service
 * - response times, e.g. times spent by packets in the node (queue and server)
 * - number of packets generated by this node source
 * - number of packets generated by this node and delivered to their target
 * - number of packets generated by this node and being lost
 * - number of packets arrived - generated at this node, or come from neighbours
 * - number of packets served by this node
 * - number of packets dropped by this node
 *
 * Settings for all series (e.g. window size or number of recorded moments)
 * are obtained from the parent network journal.
 *
 * Size metrics (queue, server and system sizes) are stored using
 * TimeSizeSeries objects. They allow to record data about size changes and
 * time when these changes appeared. Afterwards, they allow to estimate
 * probability mass functions of size distributions.
 *
 * Interval metrics (delays, departures, waiting and response times) are stored
 * using Series objects. They use sliding window approach to moments estimation
 * and can be effectively used with a limited amount of memory.
 *
 * Quantitative metrics are stored using Counter objects, those provide
 * API for increment and evaluation only.
 *
 * All Series, TimeSizeSeries and Counters are created and deleted along with
 * the journal.
 */
class NodeJournal {
  public:
    /**
     * Create a node journal.
     * @param journal journal of the network node is part of
     * @param node a node for which the journal is created
     * @param time initial model time (default: 0)
     */
    NodeJournal(NetworkJournal *journal, Node *node, double time = 0.0);

    /** Delete the journal and all internals series and counters. */
    ~NodeJournal();

    /** Get network journal. */
    inline NetworkJournal *getNetworkJournal() const { return networkJournal_; }

    /** Get the node. */
    inline Node *getNode() const { return node_; }

    /** Get system size series. */
    inline TimeSizeSeries *systemSize() const { return systemSize_; }

    /** Get queue size series. */
    inline TimeSizeSeries *queueSize() const { return queueSize_; }

    /** Get server size series. */
    inline TimeSizeSeries *serverSize() const { return serverSize_; }

    /**
     * Get delays series - time between packet generation at this node and
     * its delivery at the destination.
     */
    inline Series *delays() const { return delays_; }

    /** Get departures series - time between consequent service ends. */
    inline Series *departures() const { return departures_; }

    /** Get waiting times series - time spent by the packet in the queue. */
    inline Series *waitTimes() const { return waitTimes_; }

    /** Get response time series - time between packet arrival and departure. */
    inline Series *responseTimes() const { return responseTimes_; }

    /** Get number of generated packets counter. */
    inline Counter *numPacketsGenerated() const { return numPacketsGenerated_; }

    /** Get a counter of packets, originated at this node and delivered. */
    inline Counter *numPacketsDelivered() const { return numPacketsDelivered_; }

    /** Get a counter of packets, originated at this node and lost somewhere. */
    inline Counter *numPacketsLost() const { return numPacketsLost_; }

    /** Get a counter of packets arrived at this node. */
    inline Counter *numPacketsArrived() const { return numPacketsArrived_; }

    /** Get a counter of packets served at this node. */
    inline Counter *numPacketsServed() const { return numPacketsServed_; }

    /** Get a counter of packets dropped by this node. */
    inline Counter *numPacketsDropped() const { return numPacketsDropped_; }

    /**
     * Reset the journal - clean all series, reset counters to 0.
     * @param time initial time for time size series after reset
     */
    void reset(double time = 0.0);

    /**
     * Estimate all statistics properties of all series stored in this journal.
     * Series are recorded using sliding windows, and statistical properties
     * are estimated only when windows are filled completely. Calling this
     * method cause re-estimation of series properties using even partially
     * filled windows. This method should be called when preparing simulation
     * results.
     */
    void commit();

    /** Get string representation of the journal. */
    std::string toString() const;

  private:
    Node *node_;
    NetworkJournal *networkJournal_;

    TimeSizeSeries *systemSize_ = nullptr;
    TimeSizeSeries *queueSize_ = nullptr;
    TimeSizeSeries *serverSize_ = nullptr;
    Series *delays_ = nullptr;
    Series *departures_ = nullptr;
    Series *waitTimes_ = nullptr;
    Series *responseTimes_ = nullptr;
    Counter *numPacketsGenerated_ = nullptr;
    Counter *numPacketsDelivered_ = nullptr;
    Counter *numPacketsLost_ = nullptr;
    Counter *numPacketsArrived_ = nullptr;
    Counter *numPacketsServed_ = nullptr;
    Counter *numPacketsDropped_ = nullptr;

    /** Helper that deletes all series and sets the pointers to nullptr. */
    void clean();

    /** Create all series, time size series and counters. */
    void build(double time);
};

}}

#endif //CQUMO_TANDEM_JOURNALS_H
