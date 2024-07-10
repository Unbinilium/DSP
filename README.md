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
        std::cout << std::endl;
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

    {
        auto r = resize(data, std::vector<int>{4, 4}, std::vector<int>{2, 2});
        p(r);
    }

    {
        auto c = std::vector<double>(data.size(), 0.1);
        auto r = psnr(data, c);
        std::cout << r << std::endl;
    }

    {
        auto [psi, x] = integrate_wavelet(cwt_wavelet_t::MORLET, 10);
        std::vector<int> scales(65);
        std::iota(scales.begin(), scales.end(), 1);
        auto r = cwt(data, scales, psi, x);
        p(r);
    }
}
```
