#include <pmt/pmtf_wrap.hpp>

namespace pmtf {

std::ostream& operator<<(std::ostream& os, const pmt_wrap& x) {
    // We need a virtual member function so that we can use polymorphism.
    x.ptr()->print(os);
    return os;
}

}
