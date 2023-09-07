#include "series.h"
#include <iterator>
#include <sstream>

#include "statistics.h"

namespace cqumo {

// Class Series
// --------------------------------------------------------------------------

Series::Series(unsigned nMoments, unsigned windowSize) {
    moments_.resize(nMoments, 0.0);
    window_.resize(windowSize, 0.0);
    wPos_ = 0;
    nRecords_ = 0;
    nCommittedRecords_ = 0;
}

void Series::record(double x) {
    window_[wPos_++] = x;
    nRecords_++;
    if (wPos_ >= window_.size()) {
        commit();
    }
}

void Series::commit() {
    int numMoments = static_cast<int>(moments_.size());
    for (int i = 0; i < numMoments; ++i) {
        moments_[i] = estimateMoment(
                i+1, moments_[i], window_, wPos_, nRecords_);
    }
    nCommittedRecords_ = nRecords_;
    wPos_ = 0;
}

double Series::var() const {
    return getUnbiasedVariance(
            moments_[0],
            moments_[1],
            nCommittedRecords_);
}

std::string Series::toString() const {
    std::stringstream ss;
    ss << "(Series: moments=[";
    std::copy(moments_.begin(), moments_.end(),
              std::ostream_iterator<double>(ss, " "));
    ss << "], nRecords=" << nRecords_ << ")";
    return ss.str();
}

double Series::estimateMoment(
        int order,
        double value,
        const std::vector<double> &window,
        unsigned windowSize,
        unsigned nRecords) {
    if (nRecords <= 0) {
        return value;
    }
    double accum = 0.0;
    windowSize = std::min(static_cast<unsigned>(window.size()), windowSize);
    for (unsigned i = 0; i < windowSize; ++i) {
        accum += std::pow(window[i], order);
    }
    return value * (1.0 - static_cast<double>(windowSize) / nRecords) +
           accum / nRecords;
}


// Class TimeSizeSeries
// --------------------------------------------------------------------------
TimeSizeSeries::TimeSizeSeries(double time, unsigned value)
        : initTime_(time), currValue_(value), prevRecordTime_(0.0) {
    durations_.resize(1, 0.0);
}

TimeSizeSeries::~TimeSizeSeries() = default;

void TimeSizeSeries::record(double time, unsigned value) {
    if (durations_.size() <= currValue_) {
        durations_.resize(currValue_ + 1, 0.0);
    }
    durations_[currValue_] += time - prevRecordTime_;
    prevRecordTime_ = time;
    currValue_ = value;
}

std::vector<double> TimeSizeSeries::pmf() const {
    std::vector<double> pmf(durations_);
    double dt = prevRecordTime_ - initTime_;
    for (double & i : pmf) {
        i /= dt;
    }
    return pmf;
}

std::string TimeSizeSeries::toString() const {
    std::stringstream ss;
    ss << "(TimeSizeSeries: durations=[";
    std::copy(durations_.begin(), durations_.end(),
              std::ostream_iterator<double>(ss, " "));
    ss << "])";
    return ss.str();
}


}
