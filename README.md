## DSP

A simple header only DSP library.

```c++
#include <iostream>

#include "dsp.hpp"

int main() {
    using namespace dsp;

    std::vector<double> data(1024);
    std::generate(data.begin(), data.end(), [n = 0]() mutable { return std::sin(n++); });

    auto p = [](const auto& v) {
        for (const auto& x : v) {
            std::cout << x << " ";
        }
        std::cout << endl;
    };

    {
        auto w = firwin(100, 0.5);
        auto r = lfilter(w, 1.0, data);
        p(r);
    }

    {
        auto r = paa(data, 10);
        p(r);
    }

    {
        auto r = mtf(data, 16);
        p(r);
    }
}
```
