## DSP

A simple header only DSP library.

```c++
#include <iostream>

#include "dsp.hpp"

int main() {
    using namespace dsp;

    std::vector<double> data(1024);
    std::generate(data.begin(), data.end(), [n = 0]() mutable { return std::sin(n++); });

    auto w = firwin(100, 0.5);
    auto r = lfilter(w, 1.0, data);

    for (const auto& x : r) {
        std::cout << x << " ";
    }
}
```
