/**
 * @author Andrey Larionov
 */
#include "statistics.h"
#include "../utils/strings.h"

#include <cmath>
#include <sstream>
#include <ostream>
#include <iterator>


namespace cqumo {

double getUnbiasedVariance(double m1, double m2, unsigned n) {
    if (n > 1) {
        auto _n = static_cast<double>(n);
        return (m2 - m1 * m1) * (_n / (_n - 1));
    }
    return m2 - m1 * m1;
}





// Class SizeDist
// --------------------------------------------------------------------------

SizeDist::SizeDist() : pmf_(std::vector<double>(1, 1.0)) {}

SizeDist::SizeDist(std::vector<double> pmf) : pmf_(std::move(pmf)) {}

double SizeDist::moment(int order) const {
    double accum = 0.0;
    for (unsigned i = 0; i < pmf_.size(); ++i) {
        accum += std::pow(i, order) * pmf_[i];
    }
    return accum;
}

double SizeDist::mean() const {
    return moment(1);
}

double SizeDist::var() const {
    return moment(2) - std::pow(moment(1), 2);
}

double SizeDist::std() const {
    return std::pow(var(), 0.5);
}

std::string SizeDist::toString() const {
    std::stringstream ss;
    ss << "(SizeDist: mean=" << mean() << ", std=" << std()
       << ", pmf=" << cqumo::toString(pmf_) << ")";
    return ss.str();
}



// Class VarData
// --------------------------------------------------------------------------

VarData::VarData(const Series &series)
        : mean(series.mean()),
          std(series.std()),
          var(series.var()),
          count(series.count()),
          moments(series.moments()) {}

std::string VarData::toString() const {
    std::stringstream ss;
    ss << "(VarData: mean=" << mean
       << ", var=" << var
       << ", std=" << std
       << ", count=" << count
       << ", moments=[" << cqumo::toString(moments) << "])";
    return ss.str();
}


// Class Counter
// --------------------------------------------------------------------------

Counter::Counter(int initValue) : value_(initValue) {}

Counter &Counter::operator=(const Counter &rside) {
    value_ = rside.value();
    return *this;
}

std::string Counter::toString() const {
    std::stringstream ss;
    ss << "(Counter: value=" << value_ << ")";
    return ss.str();
}


}
