#ifndef CQUMO_STATISTICS_SERIES_H
#define CQUMO_STATISTICS_SERIES_H

#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>

namespace cqumo {

/**
 * Class representing samples series moments estimation using
 */
class Series {
public:
    Series(unsigned nMoments, unsigned windowSize);

    ~Series() = default;

    /**
     * Estimate new k-th moment value from the previous estimation and
     * new samples.
     * @param order moment order (greater or equal then 1)
     * @param value previous estimation
     * @param window array of new samples
     * @param windowSize number of samples to be taken from window
     * @param nRecords total number of samples, incl. those in the window
     * @return new moment estimation
     */
    static double estimateMoment(
            int order,
            double value,
            const std::vector<double> &window,
            unsigned windowSize,
            unsigned nRecords
    );

    /**
     * Record new sample. The sample will be written into the window.
     * If the window is full, then new moments values will be estimated
     * using `commit()`.
     * @param x
     */
    void record(double x);

    /** Estimate new moments values and reset sliding window.
     */
    void commit();

    /** Get estimated moments values. */
    inline const std::vector<double> &moments() const {
        return moments_;
    }

    /** Get moment of the given order. */
    inline double moment(int order) const {
        if (order <= 0 || order > static_cast<int>(moments_.size())) {
            throw std::out_of_range("illegal order");
        }
        return moments_[order - 1];
    }

    /** Get mean value. */
    inline double mean() const { return moments_[0]; }

    /** Get unbiased variance. */
    double var() const;

    /** Get standard deviation. */
    inline double std() const { return std::pow(var(), 0.5); }

    /** Get number of recorded samples. */
    inline unsigned count() const { return nRecords_; }

    /** Get string representation of the Series object. */
    std::string toString() const;

private:
    std::vector<double> moments_;
    std::vector<double> window_;
    unsigned wPos_;
    unsigned nRecords_;
    unsigned nCommittedRecords_;
};


/**
 * Class for recording time-size series, e.g. system or queue size.
 *
 * Size varies in time, so here we store how long each size value
 * was kept. When estimating moments, we just divide all the time
 * on the total time and so value the probability mass function.
 */
class TimeSizeSeries {
public:
    explicit TimeSizeSeries(double time = 0.0, unsigned value = 0);

    ~TimeSizeSeries();

    /**
     * Record new value update.
     *
     * Here we record information about _previous_ value, and that
     * it was kept for `(time - prevRecordTime)` interval.
     * We also store the new value as `currValue`, so the next
     * time this method is called, information about this value
     * will be recorded.
     *
     * @param time current time
     * @param value new value
     */
    void record(double time, unsigned value);

    /** Estimate probability mass function. */
    std::vector<double> pmf() const;

    /** Get string representation. */
    std::string toString() const;

private:
    double initTime_;
    unsigned currValue_;
    double prevRecordTime_;
    std::vector<double> durations_;
};



}

#endif //CQUMO_STATISTICS_SERIES_H
