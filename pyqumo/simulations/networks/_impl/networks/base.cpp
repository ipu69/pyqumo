/**
 * @author Andrey Larionov
 */
#include "base.h"
#include <sstream>

namespace cqumo {

std::string Object::toString() const {
    std::stringstream ss;
    ss << "(Object: addr=" << this << ")";
    return ss.str();
}

}
