#ifndef CQUMO_CORE_STRINGS_H
#define CQUMO_CORE_STRINGS_H

#include <string>
#include <vector>
#include <sstream>

namespace cqumo {

#ifdef DEBUG
#define debug(...) printf(__VA_ARGS__)
#else
#define debug(...) /* nop */
#endif

/**
 * Convert an array into a string with a separator between elements.
 * @tparam T elements type
 * @param array a vector
 * @param delim delimiter string (default: ", ")
 * @return string
 */
template<typename T>
std::string toString(
        const std::vector<T> &array,
        const std::string &delim = ", ") {
    std::stringstream ss;
    ss << "[";
    if (array.size() > 0) {
        ss << array[0];
        for (unsigned i = 1; i < array.size(); i++) {
            ss << delim << array[i];
        }
    }
    ss << "]";
    return ss.str();
}

}

#endif //CQUMO_CORE_STRINGS_H
