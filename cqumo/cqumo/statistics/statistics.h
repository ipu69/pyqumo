/**
 * @author Andrey Larionov
 */
#ifndef CQUMO_STATISTICS_STATISTICS_H
#define CQUMO_STATISTICS_STATISTICS_H

#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>

#include "series.h"

namespace cqumo {

/**
 * Get unbiased variance estimation.
 * If no number of samples provided, returns biased estimation.
 * Otherwise, multiplies biased estimation `m2 - m1*m1` by `n/(n-1)`
 * to value unbiased estimation.
 *
 * @param m1 sample mean
 * @param m2 sample moment of order 2
 * @param n number of samples
 * @return `(m2 - m1^2) * (n/(n-1))`
 */
double getUnbiasedVariance(double m1, double m2, unsigned n = 0);


/**
 * Size distribution given with a probability mass function of
 * values 0, 1, ..., N-1.
 */
class SizeDist {
public:
    /**
     * Create size distribution from a given PMF.
     * @param pmf a vector with sum of elements equal 1.0,
     *      all elements should be non-negative.
     */
    SizeDist();
    explicit SizeDist(std::vector<double> pmf);
    SizeDist(const SizeDist &other) = default;
    ~SizeDist() = default;

    /**
     * Get k-th moment of the distribution.
     * @param order - number of moment (e.g. 1 - mean value)
     * @return sum of i^k * pmf[i] over all i
     */
    double moment(int order) const;

    /** Get mean value. */
    double mean() const;

    /** Get variance. */
    double var() const;

    /** Get standard deviation. */
    double std() const;

    /** Get probability mass function. */
    inline const std::vector<double> &pmf() const {
        return pmf_;
    }

    /** Get string representation. */
    std::string toString() const;

private:
    std::vector<double> pmf_;
};


class Series;

/**
 * A plain structure-like class representing samples statistics:
 *
 * - average value
 * - standard deviation
 * - variance
 * - number (count) of samples
 * - estimated moments (first N moments)
 *
 * This class doesn't contain any dynamically allocated objects those need
 * manually freeing/deletion.
 */
class VarData {
public:
    double mean = 0.0;    ///< Estimated average value
    double std = 0.0;     ///< Estimated standard deviation
    double var = 0.0;     ///< Estimated variance
    unsigned count = 0;   ///< Number of samples used in estimation
    std::vector<double> moments;  ///< First N moments

    VarData() = default;
    VarData(const VarData &other) = default;

    /**
     * Construct VarData from Series object.
     * @param series
     */
    explicit VarData(const Series &series);

    /** Get string representation. */
    std::string toString() const;
};


/**
 * Simple integer counter that can be evaluated, incremented or reset.
 */
class Counter {
public:
    /**
     * Convert from integer constructor.
     * @param initValue initial counter value (default: 0)
     */
    Counter(int initValue = 0); // NOLINT(google-explicit-constructor)

    Counter(const Counter &counter) = default;
    ~Counter() = default;

    Counter &operator=(const Counter &rside);

    /** Get counter value. */
    inline int value() const { return value_; }

    /** Increment counter value. */
    inline void inc() { value_++; }

    /** Reset counter. */
    inline void reset(int initValue = 0) { value_ = initValue; }

    /** Get string representation. */
    std::string toString() const;

private:
    int value_ = 0;
};

}

#endif //CQUMO_STATISTICS_STATISTICS_H
