#pragma once

#define ENABLE_ASSERT
#define ENABLE_THROW
#define BUILD_TESTS

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef ENABLE_ASSERT
    #include <cassert>
#endif

#ifdef BUILD_TESTS
    #include <unordered_map>
#endif

namespace dsp {

namespace constants {

static constexpr double EPS = 1.0e-20;
static constexpr double PI  = 3.141592653589793238462643383279502884197169399375105820974944;

}  // namespace constants

using namespace constants;

namespace math {

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
decltype(auto) sinc(T x, T eps = static_cast<T>(EPS)) {
    if (x < eps && x > -eps) {
        x = eps;
    }
    x *= static_cast<T>(PI);
    return static_cast<T>(std::sin(x) / x);
}

}  // namespace math

using namespace math;

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
decltype(auto) hamming_window(size_t M, T alpha = static_cast<T>(0.54), bool sym = true) {
    const std::vector<T> a = {alpha, static_cast<T>(1.0) - alpha};

    std::vector<T> fac(sym ? M : M + 1);
    {
        const T    start = -PI;
        const T    end   = PI;
        const auto step  = std::abs(end - start);
        const auto size  = fac.size() - 1;
        std::generate(fac.begin(), fac.end(), [&start, &step, size, n = 0]() mutable {
            using S = std::decay_t<decltype(start)>;
            return start + (step * (static_cast<S>(n++) / static_cast<S>(size)));
        });
    }

    std::vector<T> w(fac.size(), static_cast<T>(0.0));
    {
        const auto size = w.size();
        assert(size == fac.size());
        size_t k = 0;
        for (const auto& ai : a) {
            for (size_t i = 0; i < size; ++i) {
                w[i] += ai * std::cos(static_cast<T>(k) * fac[i]);
            }
            ++k;
        }
    }

    if (!sym) {
        w.pop_back();
    }

    return w;
}

namespace types {

enum class window_t {
    HAMMING,
};

}  // namespace types

using namespace types;

namespace traits {

template <typename T, typename = std::void_t<>> struct has_index_access_operator : std::false_type {};

template <typename T>
struct has_index_access_operator<T, std::void_t<decltype(std::declval<T>()[std::declval<size_t>()])>> : std::true_type {
};

template <typename T> constexpr bool has_index_access_operator_v = has_index_access_operator<T>::value;

template <typename T, typename = std::void_t<>> struct has_iterator_support : std::false_type {};

template <typename T>
struct has_iterator_support<T, std::void_t<typename T::iterator, typename T::const_iterator>> : std::true_type {};

template <typename T> constexpr bool has_iterator_support_v = has_iterator_support<T>::value;

template <typename T, typename = std::void_t<>> struct has_random_access_iterator : std::false_type {};

template <typename T>
struct has_random_access_iterator<
  T,
  std::void_t<typename std::iterator_traits<T>::iterator_category, decltype(std::declval<T>().operator+(1))>>
    : std::conditional_t<
        std::is_base_of_v<std::random_access_iterator_tag, typename std::iterator_traits<T>::iterator_category>,
        std::true_type,
        std::false_type> {};

template <typename T> constexpr bool has_random_access_iterator_v = has_random_access_iterator<T>::value;

template <typename T, typename = std::void_t<>> struct has_size_method_with_size_t : std::false_type {};

template <typename T>
struct has_size_method_with_size_t<T, std::void_t<decltype(std::declval<T>().size())>>
    : std::is_same<size_t, decltype(std::declval<T>().size())> {};

template <typename T> constexpr bool has_size_method_with_size_t_v = has_size_method_with_size_t<T>::value;

template <typename Container, typename T, typename = std::void_t<>>
struct has_contained_type_nothrow_convertible_to : std::false_type {};

template <typename Container, typename T>
struct has_contained_type_nothrow_convertible_to<
  Container,
  T,
  std::void_t<decltype(std::declval<typename Container::value_type>()),
              std::enable_if_t<std::is_nothrow_convertible_v<typename Container::value_type, T>>>> : std::true_type {};

template <typename Container, typename T>
constexpr bool has_contained_type_nothrow_convertible_to_v =
  has_contained_type_nothrow_convertible_to<Container, T>::value;

}  // namespace traits

using namespace traits;

template <typename T = double,
          typename Conatiner,
          std::enable_if_t<has_size_method_with_size_t_v<Conatiner> && has_iterator_support_v<Conatiner> &&
                             has_contained_type_nothrow_convertible_to_v<Conatiner, T>,
                           bool> = true>
decltype(auto) firwin(size_t           numtaps,
                      const Conatiner& cutoff,
                      bool             pass_zero = true,
                      bool             scale     = true,
                      window_t         window    = window_t::HAMMING) {
    if (cutoff.size() == 0) {
#ifdef ENABLE_THROW
        throw std::invalid_argument("At least one cutoff frequency must be given.");
#else
        return std::vector<T>();
#endif
    }

    const bool pass_nyquist = 1 ^ pass_zero;
    if (pass_nyquist && numtaps % 2 == 0) {
#ifdef ENABLE_THROW
        throw std::invalid_argument("The numtaps must be odd when pass_nyquist is True.");
#else
        return std::vector<T>();
#endif
    }

    std::vector<T> bands;
    {
        if (pass_zero) {
            bands.push_back(static_cast<T>(0.0));
        }
        std::copy(cutoff.begin(), cutoff.end(), std::back_inserter(bands));
        if (pass_nyquist) {
            bands.push_back(static_cast<T>(1.0));
        }
    }

    std::vector<T> m(numtaps);
    {
        const T    alpha = 0.5 * static_cast<T>(numtaps - 1);
        const auto size  = m.size();
#ifdef ENABLE_ASSERT
        assert(size == numtaps);
#endif
        for (size_t i = 0; i < size; ++i) {
            m[i] = static_cast<T>(i) - alpha;
        }
    }

    std::vector<T> h(numtaps, static_cast<T>(0.0));
    {
        const auto size   = bands.size();
        const auto size_h = h.size();
#ifdef ENABLE_ASSERT
        assert(size_h == m.size());
#endif
        for (size_t i = 1; i < size; i += 2) {
            const auto left  = bands[i - 1];
            const auto right = bands[i];

            for (size_t j = 0; j < size_h; ++j) {
                const auto& mj = m[j];
                h[j]           = (right * sinc(right * mj)) - (left * sinc(left * mj));
            }
        }
    }

    {
        std::vector<T> win;
        switch (window) {
        case window_t::HAMMING:
            win = hamming_window<T>(numtaps);
            break;
        default:
#ifdef ENABLE_THROW
            throw std::invalid_argument("Unsupported window type.");
#else
            return std::vector<T>();
#endif
        }
        const auto size = h.size();
#ifdef ENABLE_ASSERT
        assert(size == win.size());
#endif
        for (size_t i = 0; i < size; ++i) {
            h[i] *= win[i];
        }
    }

    if (scale) {
#ifdef ENABLE_ASSERT
        assert(bands.size() >= 2);
#endif
        const auto left  = bands[0];
        const auto right = bands[1];

        T scale_frequency;
        if (left == 0.0) {
            scale_frequency = 0.0;
        } else if (right == 1.0) {
            scale_frequency = 1.0;
        } else {
            scale_frequency = 0.5 * (left + right);
        }

        const auto size = h.size();
#ifdef ENABLE_ASSERT
        assert(size == m.size());
#endif
        T s = 0.0;
        for (size_t i = 0; i < size; ++i) {
            s += h[i] * std::cos(PI * m[i] * scale_frequency);
        }
        for (auto& hi : h) {
            hi /= s;
        }
    }

    return h;
}

template <typename T = double, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
decltype(auto) firwin(
  size_t numtaps, T cutoff, bool pass_zero = true, bool scale = true, window_t window = window_t::HAMMING) {
    std::vector<T> c = {cutoff};
    return firwin<T>(numtaps, c, pass_zero, scale, window);
}

namespace traits {

template <typename Container, typename T, typename = std::void_t<>>
struct is_lfilter_container_fine : std::false_type {};

template <typename Container, typename T>
struct is_lfilter_container_fine<
  Container,
  T,
  std::void_t<std::enable_if_t<has_size_method_with_size_t_v<Container> && has_index_access_operator_v<Container> &&
                               has_contained_type_nothrow_convertible_to_v<Container, T>>>> : std::true_type {};

template <typename Container, typename T>
constexpr bool is_lfilter_container_fine_v = is_lfilter_container_fine<Container, T>::value;

}  // namespace traits

using namespace traits;

template <typename T = double> struct lfilter_ctx_t { std::vector<T> result; };

template <typename T,
          typename Container_1,
          typename Container_2,
          typename Container_3,
          std::enable_if_t<std::is_floating_point_v<T> && is_lfilter_container_fine_v<Container_1, T> &&
                             is_lfilter_container_fine_v<Container_2, T> && is_lfilter_container_fine_v<Container_3, T>,
                           bool> = true>
void lfilter(lfilter_ctx_t<T>& ctx, const Container_1& b, const Container_2& a, const Container_3& x) {
#ifdef ENABLE_ASSERT
    assert(b.size() > 0);
    assert(a.size() > 0);
    assert(x.size() > 0);
#endif

    const auto nx = x.size();
    auto&      y  = ctx.result;
    if (y.size() != nx) {
        y.resize(nx);
        std::fill(y.begin(), y.end(), static_cast<T>(0.0));
    }

    {
        const auto nb = b.size();
        const auto na = a.size();
#ifdef ENABLE_ASSERT
        assert(y.size() == nx);
#endif
        for (size_t i = 0; i < nx; ++i) {
            const auto in = i + 1;
            const auto be = std::min(nb, in);
            const auto ae = std::min(na, in);

            auto& yi = y[i];

            for (size_t j = 0; j < be; ++j) {
                yi += b[j] * x[i - j];
            }

            for (size_t j = 1; j < ae; ++j) {
                yi -= a[j] * y[i - j];
            }
        }
    }
}

template <typename T = double,
          typename Container_1,
          typename Container_2,
          typename Container_3,
          std::enable_if_t<std::is_floating_point_v<T> && is_lfilter_container_fine_v<Container_1, T> &&
                             is_lfilter_container_fine_v<Container_2, T> && is_lfilter_container_fine_v<Container_3, T>,
                           bool> = true>
decltype(auto) lfilter(const Container_1& b, const Container_2& a, const Container_3& x) {
    lfilter_ctx_t<T> ctx{};
    lfilter(ctx, b, a, x);
    const auto r = std::move(ctx.result);
    return r;
}

template <typename T = double,
          typename Container_1,
          typename Container_2,
          std::enable_if_t<std::is_floating_point_v<T> && is_lfilter_container_fine_v<Container_1, T> &&
                             is_lfilter_container_fine_v<Container_2, T>,
                           bool> = true>
decltype(auto) lfilter(const Container_1& b, T a, const Container_2& x) {
    std::vector<T> a_ = {a};
    return lfilter<T>(b, a_, x);
}

template <typename T = double> struct paa_ctx_t { std::vector<T> result; };

template <typename T = double,
          typename Container,
          std::enable_if_t<std::is_floating_point_v<T> && has_size_method_with_size_t_v<Container> &&
                             has_index_access_operator_v<Container> &&
                             has_contained_type_nothrow_convertible_to_v<Container, T>,
                           bool> = true>
void paa(paa_ctx_t<T>& ctx, const Container& ts, size_t segments) {
    const auto n = ts.size();
    auto&      y = ctx.result;
    if (y.size() != segments) {
        y.resize(segments);
        std::fill(y.begin(), y.end(), static_cast<T>(0.0));
    }

    {
        for (size_t i = 0; i < segments; ++i) {
            const size_t start = (n * i) / segments;
            const size_t end   = (n * (i + 1)) / segments;
            T            sum   = 0.0;
            for (size_t j = start; (j < end) & (j < n); ++j) {
                sum += ts[j];
            }
            y[i] = sum / static_cast<T>(end - start);
        }
    }
}

template <typename T = double,
          typename Container,
          std::enable_if_t<std::is_floating_point_v<T> && has_size_method_with_size_t_v<Container> &&
                             has_index_access_operator_v<Container> &&
                             has_contained_type_nothrow_convertible_to_v<Container, T>,
                           bool> = true>
decltype(auto) paa(const Container& ts, size_t segments) {
    paa_ctx_t<T> ctx;
    paa(ctx, ts, segments);
    const auto r = std::move(ctx.result);
    return r;
}

template <
  typename T = double,
  typename Container,
  std::enable_if_t<has_iterator_support_v<Container> && has_contained_type_nothrow_convertible_to_v<Container, T>,
                   bool> = true>
void minmax_scale(Container& v, T lower, T upper) {
    const auto [min, max] = std::minmax_element(v.begin(), v.end());
    if (min == v.end() || max == v.end()) {
        return;
    }

    const auto min_v = *min;
    const auto diff  = *max - min_v;
    const auto scale = upper - lower;
    for (auto& vi : v) {
        vi = ((vi - min_v) * scale / diff) + lower;
    }
}

template <typename T = double> struct minmax_scale_ctx_t { std::vector<T> result; };

template <
  typename T,
  typename Container,
  std::enable_if_t<has_iterator_support_v<Container> && has_contained_type_nothrow_convertible_to_v<Container, T>,
                   bool> = true>
void minmax_scale(minmax_scale_ctx_t<T>& ctx, const Container& v, T lower, T upper) {
    const auto n = v.size();
    auto&      y = ctx.result;
    if (y.size() != n) {
        y.resize(n);
    }
    if constexpr (std::is_same_v<decltype(ctx.result), Container>) {
        y = v;
    } else {
        std::copy(v.begin(), v.end(), y.begin());
    }
    minmax_scale(y, lower, upper);
}

template <typename T = double> struct mtf_ctx_t {
    minmax_scale_ctx_t<T> minmax_ctx;
    std::vector<T>        bins;
    std::vector<size_t>   digitize;
    std::vector<T>        transition_matrix;

    std::vector<T>      result;
    std::vector<size_t> shape;
};

template <typename T = double,
          typename Container,
          std::enable_if_t<has_size_method_with_size_t_v<Container> && has_index_access_operator_v<Container> &&
                             has_contained_type_nothrow_convertible_to_v<Container, T>,
                           bool> = true>
void mtf(mtf_ctx_t<T>& ctx, const Container& ts, size_t n_bins = 16) {
    auto& ctx_minmax_ctx = ctx.minmax_ctx;
    minmax_scale(ctx_minmax_ctx, ts, static_cast<T>(0.0), static_cast<T>(1.0));
    const auto& normalized = ctx_minmax_ctx.result;

    auto& bins = ctx.bins;
    if (bins.size() != n_bins) {
        bins.resize(n_bins);
        std::generate(bins.begin(), bins.end(), [space = n_bins - 1, n = 0]() mutable {
            return static_cast<T>(n++) * static_cast<T>(1.0) / static_cast<T>(space);
        });
    }
    auto&      digitize = ctx.digitize;
    const auto ns       = normalized.size();
    if (digitize.size() != ns) {
        digitize.resize(ns);
    }
    {
#ifdef ENABLE_ASSERT
        assert(digitize.size() == ns);
        assert(bins.size() == n_bins);
#endif
        for (size_t i = 0; i < ns; ++i) {
            const auto ni  = normalized[i];
            auto       idx = n_bins - 1;
            for (size_t j = 0; j < n_bins; ++j) {
                if (ni < bins[j]) {
                    idx = j;
                    break;
                }
            }
            digitize[i] = idx >= 1 ? idx - 1 : idx;
        }
    }

    const auto tms               = n_bins * n_bins;
    auto&      transition_matrix = ctx.transition_matrix;
    if (transition_matrix.size() != n_bins) {
        transition_matrix.resize(tms);
        std::fill(transition_matrix.begin(), transition_matrix.end(), static_cast<T>(0.0));
    }
    {
        const size_t start = 1;
        const size_t end   = digitize.size();
        for (size_t p = start; p < end; ++p) {
            const size_t i = digitize[p - 1];
            const size_t j = digitize[p];
            transition_matrix[(i * n_bins) + j] += static_cast<T>(1.0);
        }
    }
    {
        const auto stride = n_bins;
        const auto size   = transition_matrix.size();
#ifdef ENABLE_ASSERT
        assert(size == tms);
#endif
        for (size_t i = 0; i < size; i += stride) {
            T sum = 0.0;
            for (size_t j = 0; j < stride; ++j) {
                sum += transition_matrix[i + j];
            }
            for (size_t j = 0; j < stride; ++j) {
                transition_matrix[i + j] /= sum;
            }
        }
    }

    auto&      y     = ctx.result;
    auto&      shape = ctx.shape;
    const auto cols  = ns;
    const auto rows  = ns;
    const auto ys    = cols * rows;
    if (y.size() != ys) {
        y.resize(ys);
        shape = {cols, rows};
    }
    {
#ifdef ENABLE_ASSERT
        assert(y.size() == ys);
#endif
        for (size_t i = 0; i < cols; ++i) {
            const auto i_mul_cols = i * cols;
            for (size_t j = 0; j < rows; ++j) {
                const auto idx = i_mul_cols + j;
                y[idx]         = transition_matrix[(digitize[i] * n_bins) + digitize[j]];
            }
        }
    }
}

template <typename T = double,
          typename Container,
          std::enable_if_t<has_size_method_with_size_t_v<Container> && has_index_access_operator_v<Container> &&
                             has_contained_type_nothrow_convertible_to_v<Container, T>,
                           bool> = true>
decltype(auto) mtf(const Container& ts, size_t n_bins = 16) {
    mtf_ctx_t<T> ctx;
    mtf(ctx, ts, n_bins);
    const auto r = std::move(ctx.result);
    return r;
}

namespace types {

enum class resize_interpolation_t {
    BILINEAR,
};

}

using namespace types;

namespace traits {

template <typename Container, typename T, typename = std::void_t<>>
struct is_resize_container_fine : std::false_type {};

template <typename Container, typename T>
struct is_resize_container_fine<
  Container,
  T,
  std::void_t<std::enable_if_t<has_size_method_with_size_t_v<Container> && has_index_access_operator_v<Container> &&
                               has_contained_type_nothrow_convertible_to_v<Container, T>>>> : std::true_type {};

template <typename Container, typename T>
constexpr bool is_resize_container_fine_v = is_resize_container_fine<Container, T>::value;

template <typename Container, typename T, typename = std::void_t<>>
struct is_resize_shape_container_fine : std::false_type {};

template <typename Container, typename T>
struct is_resize_shape_container_fine<
  Container,
  T,
  std::void_t<std::enable_if_t<has_size_method_with_size_t_v<Container> && has_iterator_support_v<Container> &&
                               has_contained_type_nothrow_convertible_to_v<Container, T>>>> : std::true_type {};

template <typename Container, typename T>
constexpr bool is_resize_shape_container_fine_v = is_resize_shape_container_fine<Container, T>::value;

}  // namespace traits

using namespace traits;

template <typename T = double> struct resize_ctx_t {
    std::vector<T>      result;
    std::vector<size_t> shape;
};

template <
  typename T = double,
  typename P = int16_t,
  typename Q = double,
  typename Container_1,
  typename Container_2,
  typename Container_3,
  std::enable_if_t<std::is_nothrow_convertible_v<P, size_t> && std::is_floating_point_v<Q> &&
                     is_resize_container_fine_v<Container_1, T> && is_resize_shape_container_fine_v<Container_2, P> &&
                     is_resize_shape_container_fine_v<Container_3, P>,
                   bool> = true>
void resize(resize_ctx_t<T>&       ctx,
            const Container_1&     m,
            const Container_2&     shape,
            const Container_3&     new_shape,
            resize_interpolation_t interpolation = resize_interpolation_t::BILINEAR) {
    if (shape.size() != 2 || new_shape.size() != 2) {
#ifdef ENABLE_THROW
        throw std::invalid_argument("Unsupported dimensions of shape or new_shape.");
#else
        return;
#endif
    }
    for (const auto& s : shape) {
        if (s < 2) {
#ifdef ENABLE_THROW
            throw std::invalid_argument("The shape must be greater than or equal to 2.");
#else
            return;
#endif
        }
    }
    const auto s0        = *shape.begin();
    const auto s1        = *(std::next(shape.begin(), 1));
    const auto s0_mul_s1 = s0 * s1;
    if (m.size() < s0_mul_s1) {
#ifdef ENABLE_THROW
        throw std::invalid_argument("The size of m must be greater than or equal to the product of shape.");
#else
        return;
#endif
    }
    if (s0_mul_s1 > std::numeric_limits<P>::max()) {
#ifdef ENABLE_THROW
        throw std::invalid_argument("The product of shape must be less than or equal to the maximum value of type P.");
#else
        return;
#endif
    }

    switch (interpolation) {
    case resize_interpolation_t::BILINEAR: {
        const P w       = static_cast<P>(s0);
        const P h       = static_cast<P>(s1);
        const P w_c     = w - static_cast<P>(2);
        const P h_c     = h - static_cast<P>(2);
        const P new_w   = static_cast<P>(*new_shape.begin());
        const P new_h   = static_cast<P>(*(std::next(new_shape.begin(), 1)));
        const Q scale_x = static_cast<Q>(w) / static_cast<Q>(new_w);
        const Q scale_y = static_cast<Q>(h) / static_cast<Q>(new_h);
        const Q _0_5    = static_cast<Q>(0.5);
        const T _1      = static_cast<T>(1.0);

        auto&   t     = ctx.result;
        const P new_n = new_w * new_h;
        if (t.size() != new_n) {
            t.resize(new_n);
            ctx.shape = {static_cast<size_t>(new_h), static_cast<size_t>(new_w)};
        }

        for (P i = 0; i < new_h; ++i) {
            const P i_mul_new_w = i * new_w;

            for (P j = 0; j < new_w; ++j) {
                const Q x = (static_cast<Q>(static_cast<Q>(j) + _0_5) * scale_x) - _0_5;
                const Q y = (static_cast<Q>(static_cast<Q>(i) + _0_5) * scale_y) - _0_5;

                const P x0 = static_cast<P>(std::floor(x));
                const P y0 = static_cast<P>(std::floor(y));

                const P x1 = std::min(x0, w_c);
                const P y1 = std::min(y0, h_c);

                const P y1_mul_w    = y1 * w;
                const P y1_p1_mul_w = (y1 + static_cast<P>(1)) * w;
                const P x1_p1       = x1 + static_cast<P>(1);

                const T a = m[y1_mul_w + x1];
                const T b = m[y1_mul_w + x1_p1];
                const T c = m[y1_p1_mul_w + x1];
                const T d = m[y1_p1_mul_w + x1_p1];

                const T xd      = x - x1;
                const T yd      = y - y1;
                const T _1_s_xd = _1 - xd;
                const T _1_s_yd = _1 - yd;

                const T p = (a * _1_s_xd * _1_s_yd) + (b * xd * _1_s_yd) + (c * _1_s_xd * yd) + (d * xd * yd);

                t[i_mul_new_w + j] = p;
            }
        }
    } break;

    default:
#ifdef ENABLE_THROW
        throw std::invalid_argument("Unsupported interpolation type.");
#else
        return;
#endif
    }
}

template <
  typename T = double,
  typename P = int16_t,
  typename Q = double,
  typename Container_1,
  typename Container_2,
  typename Container_3,
  std::enable_if_t<std::is_nothrow_convertible_v<P, size_t> && std::is_floating_point_v<Q> &&
                     is_resize_container_fine_v<Container_1, T> && is_resize_shape_container_fine_v<Container_2, P> &&
                     is_resize_shape_container_fine_v<Container_3, P>,
                   bool> = true>
decltype(auto) resize(const Container_1&     m,
                      const Container_2&     shape,
                      const Container_3&     new_shape,
                      resize_interpolation_t interpolation = resize_interpolation_t::BILINEAR) {
    resize_ctx_t<T> ctx;
    resize<T, P, Q>(ctx, m, shape, new_shape, interpolation);
    const auto r = std::move(ctx.result);
    return r;
}

namespace traits {

template <typename Container, typename T, typename = std::void_t<>> struct is_psnr_container_fine : std::false_type {};

template <typename Container, typename T>
struct is_psnr_container_fine<
  Container,
  T,
  std::void_t<std::enable_if_t<has_size_method_with_size_t_v<Container> && has_index_access_operator_v<Container> &&
                               has_contained_type_nothrow_convertible_to_v<Container, T>>>> : std::true_type {};

template <typename Container, typename T>
constexpr bool is_psnr_container_fine_v = is_psnr_container_fine<Container, T>::value;

}  // namespace traits

using namespace traits;

template <typename T = double,
          typename Container,
          std::enable_if_t<std::is_floating_point_v<T> && is_psnr_container_fine_v<Container, T> &&
                             is_psnr_container_fine_v<Container, T>,
                           bool> = true>
decltype(auto) psnr(const Container& target, const Container& preds) {
    const auto n_target = target.size();
    const auto n_preds  = preds.size();
    if (n_target != n_preds) {
#ifdef ENABLE_THROW
        throw std::invalid_argument("The size of target and preds must be equal.");
#else
        return std::numeric_limits<T>::infinity();
#endif
    }
    if (n_target == 0) {
#ifdef ENABLE_THROW
        throw std::invalid_argument("The size of target and preds must be greater than 0.");
#else
        return std::numeric_limits<T>::infinity();
#endif
    }

    T y;
    {
        using V = typename Container::value_type;
        V max   = std::numeric_limits<V>::min();
        T mse   = static_cast<T>(0.0);
        for (size_t i = 0; i < n_target; ++i) {
            const auto target_i = target[i];
            max                 = std::max(max, target_i);
            const auto diff     = target_i - preds[i];
            mse += diff * diff;
        }
        mse /= static_cast<T>(n_target);
        if (mse < EPS) {
            mse = EPS;
        }
        y = static_cast<T>(10.0) * std::log10(static_cast<T>(max * max) / mse);
    }

    return y;
}

namespace types {

enum class cwt_wavelet_t {
    MORLET,
};

}  // namespace types

using namespace types;

template <typename T = double, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
decltype(auto) integrate_wavelet(cwt_wavelet_t wavelet, size_t percision = 10, T lower = -8.0, T upper = 8.0) {
#ifdef ENABLE_ASSERT
    assert(percision > 2);
    assert(percision < sizeof(size_t) * 8);
    assert(lower < upper);
#endif

    const size_t   n = 1 << percision;
    std::vector<T> psi(n);
    std::vector<T> x(n);
    const auto     space = upper - lower;

#ifdef ENABLE_ASSERT
    assert(psi.size() == n);
    assert(x.size() == n);
#endif

    const auto n_s_1 = static_cast<T>(n - 1);

    switch (wavelet) {
    case cwt_wavelet_t::MORLET: {
        for (size_t i = 0; i < n; ++i) {
            const auto t = lower + (space * (static_cast<T>(i) / n_s_1));
            psi[i]       = std::exp(static_cast<T>(-0.5) * t * t) * std::cos(static_cast<T>(5.0) * t);
            x[i]         = t;
        }
    } break;

    default:
#ifdef ENABLE_THROW
        throw std::invalid_argument("Unsupported wavelet type.");
#else
        return std::make_pair(std::move(psi), std::move(x));
#endif
    }

    const auto step = x[1] - x[0];
    T          sum  = 0.0;
    for (size_t i = 0; i < n; ++i) {
        auto& psi_i = psi[i];
        sum += psi_i;
        psi_i = sum * step;
    }

    return std::make_pair(std::move(psi), std::move(x));
}

template <typename T = double> struct cwt_ctx_t {
    std::vector<size_t> psi_arange;
    std::vector<size_t> psi_indices;
    std::vector<T>      coefficients;

    std::vector<T>      result;
    std::vector<size_t> shape;
};

namespace traits {

template <typename Container, typename T, typename = std::void_t<>> struct is_cwt_container_fine : std::false_type {};

template <typename Container, typename T>
struct is_cwt_container_fine<
  Container,
  T,
  std::void_t<std::enable_if_t<has_size_method_with_size_t_v<Container> && has_index_access_operator_v<Container> &&
                               has_contained_type_nothrow_convertible_to_v<Container, T>>>> : std::true_type {};

template <typename Container, typename T>
constexpr bool is_cwt_container_fine_v = is_cwt_container_fine<Container, T>::value;

}  // namespace traits

using namespace traits;

template <typename T = double,
          typename P = double,
          typename Container_1,
          typename Container_2,
          std::enable_if_t<std::is_floating_point_v<T> && std::is_floating_point_v<P> &&
                             is_cwt_container_fine_v<Container_1, T> && is_cwt_container_fine_v<Container_2, T>,
                           bool> = true>
void cwt(cwt_ctx_t<T>&         ctx,
         const Container_1&    ts,
         const Container_2&    ascending_scales,
         const std::vector<T>& wavelet_psi,
         const std::vector<T>& wavelet_x) {
    const auto n_psi = wavelet_psi.size();
    const auto n_x   = wavelet_x.size();
#ifdef ENABLE_ASSERT
    assert(n_psi > 2);
    assert(n_x > 2);
    assert(n_x <= n_psi);
#endif
    const auto x_0      = wavelet_x[0];
    const auto x_1      = wavelet_x[1];
    const auto x_n      = wavelet_x[n_x - 1];
    const auto x_step   = x_1 - x_0;
    const auto x_range  = x_n - x_0;
    const auto n_scales = ascending_scales.size();
#ifdef ENABLE_ASSERT
    assert(x_step > EPS);
    assert(x_range > EPS);
    assert(n_scales > 0);
#endif
    const auto max_scale         = ascending_scales[n_scales - 1];
    const auto n_psi_indices_max = static_cast<size_t>(std::ceil((max_scale * x_range) + 1.0));
    auto&      psi_arange        = ctx.psi_arange;
    if (psi_arange.size() != n_psi_indices_max) {
        psi_arange.resize(n_psi_indices_max);
        std::iota(psi_arange.begin(), psi_arange.end(), 0);
    }

    auto& psi_indices = ctx.psi_indices;
    if (psi_indices.size() != n_psi_indices_max) {
        psi_indices.resize(n_psi_indices_max);
    }

    const auto n_ts = ts.size();
#ifdef ENABLE_ASSERT
    assert(n_ts > 0);
#endif
    const auto n_result = n_scales * n_ts;
    auto&      result   = ctx.result;
    if (result.size() != n_result) {
        result.resize(n_result);
        ctx.shape = {n_scales, n_ts};
    }
    const auto n_coefficients = n_ts + n_psi_indices_max - 1;
    auto&      coefficients   = ctx.coefficients;
    if (coefficients.size() != n_coefficients) {
        coefficients.resize(n_coefficients);
    }
    {
#ifdef ENABLE_ASSERT
        assert(ascending_scales.size() >= n_scales);
#endif
        size_t result_pos = 0;
        for (size_t i = 0; i < n_scales; ++i) {
            const auto scale           = ascending_scales[i];
            size_t     len_psi_indices = 0;

            {
                const auto psi_arange_end = static_cast<size_t>(std::ceil((scale * x_range) + 1.0));
                const auto scale_mul_step = scale * x_step;
#ifdef ENABLE_ASSERT
                assert(psi_arange.size() >= psi_arange_end);
#endif
                for (size_t j = 0; j < psi_arange_end; ++j) {
                    const auto idx = static_cast<size_t>(std::floor(static_cast<T>(psi_arange[j]) / scale_mul_step));
                    if (idx >= n_psi) {
                        break;
                    }
                    psi_indices[len_psi_indices++] = idx;
                }
            }

            const auto len_conv = n_ts + len_psi_indices - 1;
#ifdef ENABLE_ASSERT
            assert(len_conv > 1);
            assert(coefficients.size() >= len_conv);
#endif
            {
                const auto len_psi_idx_s_1 = len_psi_indices - 1;
                for (size_t j = 0; j < n_ts; ++j) {
                    const auto& ts_j = ts[j];
                    for (size_t k = 0; k < len_psi_indices; ++k) {
                        const auto psi_indices_rk = len_psi_idx_s_1 - k;
                        const auto psi_index_rk   = psi_indices[psi_indices_rk];
                        const auto psi_rk         = wavelet_psi[psi_index_rk];
                        coefficients[j + k] += ts_j * psi_rk;
                    }
                }
            }

            const auto len_diff = len_conv - 1;
#ifdef ENABLE_ASSERT
            assert(len_diff > 1);
            assert(coefficients.size() >= len_diff);
#endif
            {
                const auto negtive_sqrt_scale = -std::sqrt(scale);
                for (size_t j = 1; j < len_conv; ++j) {
                    auto&       coefficient_prev    = coefficients[j - 1];
                    const auto& coefficient_current = coefficients[j];
                    coefficient_prev                = coefficient_current - coefficient_prev;
                    coefficient_prev *= negtive_sqrt_scale;
                }
            }

            const auto d = static_cast<P>(len_diff - n_ts) / static_cast<P>(2.0);
            if (d < EPS) {
#ifdef ENABLE_THROW
                throw std::runtime_error("Selected scale is too small.");
#else
                return;
#endif
            }
            const size_t start   = static_cast<size_t>(std::floor(d));
            const size_t end_max = start + n_ts;
            const size_t end     = std::min(end_max, len_diff - static_cast<size_t>(std::ceil(d)));
            for (size_t j = start; j < end; ++j) {
                result[result_pos++] = coefficients[j];
            }
            for (size_t j = end; j < end_max; ++j) {
                result[result_pos++] = static_cast<T>(0.0);
            }
        }
    }
}

template <typename T = double,
          typename P = double,
          typename Container_1,
          typename Container_2,
          std::enable_if_t<std::is_floating_point_v<T> && std::is_floating_point_v<P> &&
                             is_cwt_container_fine_v<Container_1, T> && is_cwt_container_fine_v<Container_2, T>,
                           bool> = true>
decltype(auto) cwt(const Container_1&    ts,
                   const Container_2&    ascending_scales,
                   const std::vector<T>& wavelet_psi,
                   const std::vector<T>& wavelet_x) {
    cwt_ctx_t<T> ctx;
    cwt<T, P>(ctx, ts, ascending_scales, wavelet_psi, wavelet_x);
    const auto r = std::move(ctx.result);
    return r;
}

namespace tests {

#ifdef BUILD_TESTS

void test_eps() {
    assert(double(EPS) > 0.0);
    assert(-double(EPS) < -0.0);
    assert(float(EPS) > 0.0);
    assert(-float(EPS) < -0.0);

    double a    = 1.0;
    double b    = 2.0;
    double c    = 3.0;
    double diff = std::abs((a + b) - c);
    assert(diff < EPS);
}

void test_pi() {
    double math_pi = 2.0 * std::acos(0.0);
    double diff    = std::abs(math_pi - PI);
    assert(diff < EPS);
}

void test_sinc() {
    std::unordered_map<double, double> sinc_values = {
      {               -10.0, -3.898171832519376e-17},
      {  -9.797979797979798,  -0.019261976377391934},
      {  -9.595959595959595,   -0.03167529216345295},
      {  -9.393939393939394,    -0.0320209754858246},
      {  -9.191919191919192,   -0.01963689594706574},
      {   -8.98989898989899,  0.0011234069383833456},
      {  -8.787878787878787,   0.022390627055390324},
      {  -8.585858585858587,    0.03573323328381772},
      {  -8.383838383838384,    0.03546686916735175},
      {  -8.181818181818182,   0.021033383197518078},
      {  -7.979797979797979, -0.0025299463342708375},
      {  -7.777777777777778,   -0.02630644082738656},
      {  -7.575757575757576,   -0.04083251432108083},
      {  -7.373737373737374,   -0.03981623910599781},
      {  -7.171717171717171,  -0.022799085369922152},
      {   -6.96969696969697,  0.0043412616727512955},
      {  -6.767676767676768,   0.031360712390831075},
      {  -6.565656565656566,   0.047453364661312565},
      {  -6.363636363636363,    0.04549990608592487},
      {  -6.161616161616162,   0.025116986139960693},
      {  -5.959595959595959,  -0.006761470032864263},
      {  -5.757575757575758,   -0.03815129506783636},
      {  -5.555555555555555,  -0.056425327879361546},
      {  -5.353535353535354,   -0.05327389425526406},
      {  -5.151515151515151,  -0.028313617972092437},
      {   -4.94949494949495,   0.010161320878666369},
      {  -4.747474747474747,    0.04778489884461753},
      {  -4.545454545454546,    0.06931538911162695},
      {  -4.343434343434343,    0.06459757362729442},
      {  -4.141414141414142,    0.03303411947658101},
      { -3.9393939393939394,   -0.01529182990563534},
      {  -3.737373737373738,   -0.06256473652502492},
      { -3.5353535353535355,    -0.0894814635493308},
      {  -3.333333333333333,   -0.08269933431326874},
      { -3.1313131313131315,   -0.04075611340708284},
      {  -2.929292929292929,    0.02393991393455532},
      { -2.7272727272727275,    0.08820627236525579},
      {  -2.525252525252525,    0.12565425718891227},
      { -2.3232323232323235,     0.1164222803677198},
      {  -2.121212121212121,    0.05577180743829834},
      { -1.9191919191919187,  -0.041654451759341626},
      {  -1.717171717171718,   -0.14387325987267854},
      { -1.5151515151515156,   -0.20984657037171237},
      { -1.3131313131313131,     -0.201819279624739},
      { -1.1111111111111107,    -0.0979815536051013},
      { -0.9090909090909101,    0.09864608391270921},
      { -0.7070707070707076,     0.3582369603998354},
      { -0.5050505050505052,     0.6301742431604164},
      {-0.30303030303030276,     0.8556490093311446},
      {-0.10101010101010033,     0.9833009727996326},
      { 0.10101010101010033,     0.9833009727996326},
      { 0.30303030303030276,     0.8556490093311446},
      {  0.5050505050505052,     0.6301742431604164},
      {  0.7070707070707076,     0.3582369603998354},
      {  0.9090909090909083,    0.09864608391271118},
      {  1.1111111111111107,    -0.0979815536051013},
      {  1.3131313131313131,     -0.201819279624739},
      {  1.5151515151515156,   -0.20984657037171237},
      {  1.7171717171717162,   -0.14387325987267932},
      {  1.9191919191919187,  -0.041654451759341626},
      {   2.121212121212121,    0.05577180743829834},
      {  2.3232323232323235,     0.1164222803677198},
      {   2.525252525252524,    0.12565425718891232},
      {  2.7272727272727266,    0.08820627236525595},
      {   2.929292929292929,    0.02393991393455532},
      {  3.1313131313131315,   -0.04075611340708284},
      {   3.333333333333334,   -0.08269933431326888},
      {  3.5353535353535346,   -0.08948146354933081},
      {   3.737373737373737,   -0.06256473652502502},
      {  3.9393939393939394,   -0.01529182990563534},
      {   4.141414141414142,    0.03303411947658101},
      {  4.3434343434343425,    0.06459757362729436},
      {   4.545454545454545,    0.06931538911162698},
      {   4.747474747474747,    0.04778489884461753},
      {    4.94949494949495,   0.010161320878666369},
      {  5.1515151515151505,  -0.028313617972092246},
      {   5.353535353535353,   -0.05327389425526398},
      {   5.555555555555555,  -0.056425327879361546},
      {   5.757575757575758,   -0.03815129506783636},
      {  5.9595959595959584,  -0.006761470032864453},
      {   6.161616161616163,   0.025116986139960693},
      {   6.363636363636363,    0.04549990608592487},
      {   6.565656565656564,   0.047453364661312655},
      {   6.767676767676768,   0.031360712390831075},
      {   6.969696969696969,   0.004341261672751458},
      {   7.171717171717173,  -0.022799085369922416},
      {   7.373737373737374,   -0.03981623910599781},
      {   7.575757575757574,   -0.04083251432108087},
      {   7.777777777777779,   -0.02630644082738656},
      {   7.979797979797979, -0.0025299463342708375},
      {    8.18181818181818,   0.021033383197517852},
      {   8.383838383838384,    0.03546686916735175},
      {   8.585858585858585,   0.035733233283817806},
      {   8.787878787878789,   0.022390627055390116},
      {    8.98989898989899,  0.0011234069383833456},
      {    9.19191919191919,   -0.01963689594706564},
      {   9.393939393939394,    -0.0320209754858246},
      {   9.595959595959595,   -0.03167529216345295},
      {     9.7979797979798,  -0.019261976377391746},
      {                10.0, -3.898171832519376e-17},
    };

    for (const auto& [x, expected] : sinc_values) {
        double result = sinc<double>(x);
        double diff   = std::abs(result - expected);
        assert(diff < 1e-16);
    }

    for (const auto& [x, expected] : sinc_values) {
        double result = sinc<float>(x);
        double diff   = std::abs(result - expected);
        assert(diff < 1e-7);
    }
}

void test_hamming_window() {
    std::vector<double> expected = {
      0.08,       0.08092613, 0.08370079, 0.0883128,  0.0947436,  0.10296729, 0.11295075, 0.12465379, 0.13802929,
      0.15302337, 0.16957568, 0.18761956, 0.20708234, 0.22788567, 0.24994577, 0.27317382, 0.29747628, 0.32275531,
      0.34890909, 0.37583234, 0.40341663, 0.43155089, 0.46012184, 0.48901443, 0.51811232, 0.54729834, 0.57645498,
      0.60546483, 0.63421107, 0.66257795, 0.69045126, 0.71771876, 0.74427064, 0.77,       0.79480323, 0.81858046,
      0.84123594, 0.86267845, 0.88282165, 0.90158442, 0.91889123, 0.93467237, 0.94886431, 0.96140989, 0.97225861,
      0.98136677, 0.9886977,  0.99422189, 0.99791708, 0.99976841, 0.99976841, 0.99791708, 0.99422189, 0.9886977,
      0.98136677, 0.97225861, 0.96140989, 0.94886431, 0.93467237, 0.91889123, 0.90158442, 0.88282165, 0.86267845,
      0.84123594, 0.81858046, 0.79480323, 0.77,       0.74427064, 0.71771876, 0.69045126, 0.66257795, 0.63421107,
      0.60546483, 0.57645498, 0.54729834, 0.51811232, 0.48901443, 0.46012184, 0.43155089, 0.40341663, 0.37583234,
      0.34890909, 0.32275531, 0.29747628, 0.27317382, 0.24994577, 0.22788567, 0.20708234, 0.18761956, 0.16957568,
      0.15302337, 0.13802929, 0.12465379, 0.11295075, 0.10296729, 0.0947436,  0.0883128,  0.08370079, 0.08092613,
      0.08,
    };

    {
        auto w = hamming_window<double>(expected.size());
        assert(w.size() == expected.size());
        for (size_t i = 0; i < w.size(); ++i) {
            double diff = std::abs(w[i] - expected[i]);
            assert(diff < 1e-8);
        }
    }

    {
        auto w = hamming_window<float>(expected.size());
        assert(w.size() == expected.size());
        for (size_t i = 0; i < w.size(); ++i) {
            double diff = std::abs(w[i] - expected[i]);
            assert(diff < 1e-6);
        }
    }
}

void test_firwin() {
    std::vector<double> expected = {
      3.63503901e-04,  3.75293735e-04,  -3.96332982e-04, -4.27164335e-04, 4.68341628e-04,  5.20431460e-04,
      -5.84015203e-04, -6.59691447e-04, 7.48078949e-04,  8.49820181e-04,  -9.65585561e-04, -1.09607851e-03,
      1.24204145e-03,  1.40426296e-03,  -1.58358628e-03, -1.78091942e-03, 1.99724724e-03,  2.23364585e-03,
      -2.49129997e-03, -2.77152368e-03, 3.07578565e-03,  3.40573964e-03,  -3.76326187e-03, -4.15049683e-03,
      4.56991398e-03,  5.02437836e-03,  -5.51723930e-03, -6.05244271e-03, 6.63467485e-03,  7.26954799e-03,
      -7.96384327e-03, -8.72583217e-03, 9.56570816e-03,  1.04961751e-02,  -1.15332634e-02, -1.26974835e-02,
      1.40154916e-02,  1.55225556e-02,  -1.72663055e-02, -1.93126286e-02, 2.17552855e-02,  2.47323148e-02,
      -2.84555598e-02, -3.32674503e-02, 3.97597270e-02,  4.90504641e-02,  -6.35359851e-02, -8.94473748e-02,
      1.49633035e-01,  4.49731899e-01,  4.49731899e-01,  1.49633035e-01,  -8.94473748e-02, -6.35359851e-02,
      4.90504641e-02,  3.97597270e-02,  -3.32674503e-02, -2.84555598e-02, 2.47323148e-02,  2.17552855e-02,
      -1.93126286e-02, -1.72663055e-02, 1.55225556e-02,  1.40154916e-02,  -1.26974835e-02, -1.15332634e-02,
      1.04961751e-02,  9.56570816e-03,  -8.72583217e-03, -7.96384327e-03, 7.26954799e-03,  6.63467485e-03,
      -6.05244271e-03, -5.51723930e-03, 5.02437836e-03,  4.56991398e-03,  -4.15049683e-03, -3.76326187e-03,
      3.40573964e-03,  3.07578565e-03,  -2.77152368e-03, -2.49129997e-03, 2.23364585e-03,  1.99724724e-03,
      -1.78091942e-03, -1.58358628e-03, 1.40426296e-03,  1.24204145e-03,  -1.09607851e-03, -9.65585561e-04,
      8.49820181e-04,  7.48078949e-04,  -6.59691447e-04, -5.84015203e-04, 5.20431460e-04,  4.68341628e-04,
      -4.27164335e-04, -3.96332982e-04, 3.75293735e-04,  3.63503901e-04,
    };

    {
        auto h = firwin<double>(100, 0.5);
        assert(h.size() == expected.size());
        for (size_t i = 0; i < h.size(); ++i) {
            double diff = std::abs(h[i] - expected[i]);
            assert(diff < 1e-9);
        }
    }

    {
        auto h = firwin<float>(100, 0.5);
        assert(h.size() == expected.size());
        for (size_t i = 0; i < h.size(); ++i) {
            double diff = std::abs(h[i] - expected[i]);
            assert(diff < 1e-7);
        }
    }
}

void test_lfilter() {
    std::vector<double> data = {
      0.,         0.01010101, 0.02020202, 0.03030303, 0.04040404, 0.05050505, 0.06060606, 0.07070707, 0.08080808,
      0.09090909, 0.1010101,  0.11111111, 0.12121212, 0.13131313, 0.14141414, 0.15151515, 0.16161616, 0.17171717,
      0.18181818, 0.19191919, 0.2020202,  0.21212121, 0.22222222, 0.23232323, 0.24242424, 0.25252525, 0.26262626,
      0.27272727, 0.28282828, 0.29292929, 0.3030303,  0.31313131, 0.32323232, 0.33333333, 0.34343434, 0.35353535,
      0.36363636, 0.37373737, 0.38383838, 0.39393939, 0.4040404,  0.41414141, 0.42424242, 0.43434343, 0.44444444,
      0.45454545, 0.46464646, 0.47474747, 0.48484848, 0.49494949, 0.50505051, 0.51515152, 0.52525253, 0.53535354,
      0.54545455, 0.55555556, 0.56565657, 0.57575758, 0.58585859, 0.5959596,  0.60606061, 0.61616162, 0.62626263,
      0.63636364, 0.64646465, 0.65656566, 0.66666667, 0.67676768, 0.68686869, 0.6969697,  0.70707071, 0.71717172,
      0.72727273, 0.73737374, 0.74747475, 0.75757576, 0.76767677, 0.77777778, 0.78787879, 0.7979798,  0.80808081,
      0.81818182, 0.82828283, 0.83838384, 0.84848485, 0.85858586, 0.86868687, 0.87878788, 0.88888889, 0.8989899,
      0.90909091, 0.91919192, 0.92929293, 0.93939394, 0.94949495, 0.95959596, 0.96969697, 0.97979798, 0.98989899,
      1.,
    };
    std::vector<double> expected = {
      0.00000000e+00,  3.67175657e-06,  1.11343590e-05,  1.45935979e-05,  1.37380456e-05, 1.76132167e-05,
      2.67452714e-05,  2.99781825e-05,  2.65475437e-05,  3.06732579e-05,  4.33830143e-05, 4.63393813e-05,
      3.82242481e-05,  4.26549882e-05,  6.12702026e-05,  6.38895960e-05,  4.85199042e-05, 5.33244271e-05,
      8.06910292e-05,  8.28929852e-05,  5.70997525e-05,  6.23750616e-05,  1.02051781e-04, 1.03715755e-04,
      6.34555178e-05,  6.93560282e-05,  1.26007835e-04,  1.26929952e-04,  6.67162844e-05, 7.35195342e-05,
      1.53752562e-04,  1.53542728e-04,  6.51931752e-05,  7.34669373e-05,  1.87762670e-04, 1.85560793e-04,
      5.51015065e-05,  6.62128426e-05,  2.34117669e-04,  2.27615370e-04,  2.60360137e-05, 4.42070166e-05,
      3.12199382e-04,  2.92761849e-04,  -6.27105342e-05, -1.65695136e-05, 5.25030740e-04, 4.24853367e-04,
      -5.78832843e-04, -7.10742495e-05, 4.97943080e-03,  1.45726823e-02,  2.56773786e-02, 3.58785661e-02,
      4.54379759e-02,  5.54928450e-02,  6.59493275e-02,  7.60697751e-02,  8.59027929e-02, 9.59856320e-02,
      1.06288221e-01,  1.16395734e-01,  1.26328839e-01,  1.36418738e-01,  1.46650207e-01, 1.56753419e-01,
      1.66740134e-01,  1.76832870e-01,  1.87022230e-01,  1.97123450e-01,  2.07144227e-01, 2.17238433e-01,
      2.27399657e-01,  2.37499745e-01,  2.47544104e-01,  2.57639213e-01,  2.67780483e-01, 2.77879830e-01,
      2.87941163e-01,  2.98036898e-01,  3.08163701e-01,  3.18262509e-01,  3.28336153e-01, 3.38432358e-01,
      3.48548738e-01,  3.58647129e-01,  3.68729524e-01,  3.78826103e-01,  3.88935228e-01, 3.99033282e-01,
      4.09121582e-01,  4.19218467e-01,  4.29322907e-01,  4.39420685e-01,  4.49512563e-01, 4.59609698e-01,
      4.69711563e-01,  4.79809114e-01,  4.89902662e-01,  5.00000000e-01,
    };

    auto w = firwin<double>(100, 0.5);

    {
        auto r = lfilter<double>(w, 1.0, data);
        assert(r.size() == expected.size());
        for (size_t i = 0; i < r.size(); ++i) {
            double diff = std::abs(r[i] - expected[i]);
            assert(diff < 1e-8);
        }
    }

    {
        auto r = lfilter<float>(w, 1.0, data);
        assert(r.size() == expected.size());
        for (size_t i = 0; i < r.size(); ++i) {
            double diff = std::abs(r[i] - expected[i]);
            assert(diff < 1e-6);
        }
    }
}

void test_paa() {
    std::vector<double> data = {
      0.,          0.84147098,  0.90929743,  0.14112001,  -0.7568025,  -0.95892427, -0.2794155,  0.6569866,
      0.98935825,  0.41211849,  -0.54402111, -0.99999021, -0.53657292, 0.42016704,  0.99060736,  0.65028784,
      -0.28790332, -0.96139749, -0.75098725, 0.14987721,  0.91294525,  0.83665564,  -0.00885131, -0.8462204,
      -0.90557836, -0.13235175, 0.76255845,  0.95637593,  0.27090579,  -0.66363388, -0.98803162, -0.40403765,
      0.55142668,  0.99991186,  0.52908269,  -0.42818267, -0.99177885, -0.64353813, 0.29636858,  0.96379539,
      0.74511316,  -0.15862267, -0.91652155, -0.83177474, 0.01770193,  0.85090352,  0.90178835,  0.12357312,
      -0.76825466, -0.95375265, -0.26237485, 0.67022918,  0.98662759,  0.39592515,  -0.55878905, -0.99975517,
      -0.521551,   0.43616476,  0.99287265,  0.63673801,  -0.30481062, -0.96611777, -0.7391807,  0.1673557,
      0.92002604,  0.82682868,  -0.02655115, -0.85551998, -0.89792768, -0.11478481, 0.77389068,  0.95105465,
      0.25382336,  -0.67677196, -0.98514626, -0.38778164, 0.56610764,  0.99952016,  0.51397846,  -0.44411267,
      -0.99388865, -0.62988799, 0.31322878,  0.96836446,  0.73319032,  -0.17607562, -0.92345845, -0.82181784,
      0.0353983,   0.86006941,  0.89399666,  0.10598751,  -0.77946607, -0.94828214, -0.24525199, 0.68326171,
      0.98358775,  0.37960774,  -0.57338187, -0.99920683,
    };
    std::vector<double> expected_7 = {
      0.02105659,
      0.0975727,
      0.00562705,
      -0.12440166,
      0.0606417,
      0.08440562,
      -0.10166878,
    };
    std::vector<double> expected_10 = {
      0.19552095,
      -0.18699328,
      0.11828053,
      -0.01149837,
      -0.09898462,
      0.17760873,
      -0.19906823,
      0.15645624,
      -0.06348773,
      -0.04991475,
    };

    {
        auto r = paa<double>(data, 7);
        assert(r.size() == expected_7.size());
        for (size_t i = 0; i < r.size(); ++i) {
            double diff = std::abs(r[i] - expected_7[i]);
            assert(diff < 1e-8);
        }
    }

    {
        auto r = paa<float>(data, 10);
        assert(r.size() == expected_10.size());
        for (size_t i = 0; i < r.size(); ++i) {
            double diff = std::abs(r[i] - expected_10[i]);
            assert(diff < 1e-7);
        }
    }
}

void test_minmax_scale() {
    std::vector<double> data = {
      1.,          0.54030231,  -0.41614684, -0.9899925,  -0.65364362, 0.28366219,  0.96017029,  0.75390225,
      -0.14550003, -0.91113026, -0.83907153, 0.0044257,   0.84385396,  0.90744678,  0.13673722,  -0.75968791,
      -0.95765948, -0.27516334, 0.66031671,  0.98870462,  0.40808206,  -0.54772926, -0.99996083, -0.53283302,
      0.42417901,  0.99120281,  0.64691932,  -0.29213881, -0.96260587, -0.74805753, 0.15425145,  0.91474236,
      0.83422336,  -0.01327675, -0.84857027, -0.90369221, -0.12796369, 0.76541405,  0.95507364,  0.26664293,
      -0.66693806, -0.98733928, -0.39998531, 0.5551133,   0.99984331,  0.52532199,  -0.43217794, -0.99233547,
      -0.64014434, 0.30059254,  0.96496603,  0.7421542,   -0.16299078, -0.91828279, -0.82930983, 0.02212676,
      0.85322011,  0.89986683,  0.11918014,  -0.77108022, -0.95241298, -0.25810164, 0.67350716,  0.98589658,
      0.39185723,  -0.56245385, -0.99964746, -0.5177698,  0.44014302,  0.99339038,  0.6333192,   -0.30902273,
      -0.96725059, -0.73619272, 0.17171734,  0.92175127,  0.82433133,  -0.03097503, -0.85780309, -0.89597095,
      -0.11038724, 0.77668598,  0.9496777,   0.24954012,  -0.6800235,  -0.98437664, -0.38369844, 0.56975033,
      0.99937328,  0.51017704,  -0.44807362, -0.99436746, -0.62644445, 0.3174287,   0.96945937,  0.73017356,
      -0.18043045, -0.92514754, -0.81928825, 0.03982088,
    };
    std::vector<double> expected = {
      1.00000000e+00, 7.70146651e-01, 2.91912713e-01, 4.98426252e-03, 1.73161994e-01, 6.41824077e-01, 9.80084753e-01,
      8.76948717e-01, 4.27238765e-01, 4.44161522e-02, 8.04462243e-02, 5.02203099e-01, 9.21925450e-01, 9.53722484e-01,
      5.68360155e-01, 1.20138810e-01, 2.11510873e-02, 3.62405843e-01, 8.30155027e-01, 9.94352198e-01, 7.04035234e-01,
      2.26120212e-01, 0.00000000e+00, 2.33568478e-01, 7.12083864e-01, 9.95601320e-01, 8.23456203e-01, 3.53917941e-01,
      1.86778459e-02, 1.25954115e-01, 5.77117442e-01, 9.57370344e-01, 9.17110057e-01, 4.93351703e-01, 7.56967585e-02,
      4.81352535e-02, 4.36007108e-01, 8.82704729e-01, 9.77536382e-01, 6.33314284e-01, 1.66514644e-01, 6.31089805e-03,
      2.99993632e-01, 7.77552294e-01, 9.99921653e-01, 7.62656346e-01, 2.83897001e-01, 3.81275330e-03, 1.79911767e-01,
      6.50289422e-01, 9.82482671e-01, 8.71074573e-01, 4.18493220e-01, 4.08398200e-02, 8.53271681e-02, 5.11053801e-01,
      9.26608616e-01, 9.49932433e-01, 5.59581441e-01, 1.14442543e-01, 2.37743887e-02, 3.70936861e-01, 8.36750384e-01,
      9.92948153e-01, 6.95922659e-01, 2.18757772e-01, 1.56688283e-04, 2.41100236e-01, 7.20066028e-01, 9.96695125e-01,
      8.16656010e-01, 3.45475816e-01, 1.63554394e-02, 1.31886637e-01, 5.85850559e-01, 9.60874869e-01, 9.12163945e-01,
      4.84502387e-01, 7.10802588e-02, 5.19959582e-02, 4.44795503e-01, 8.88340804e-01, 9.74838356e-01, 6.24762709e-01,
      1.59971799e-01, 7.79224413e-03, 3.08137226e-01, 7.84870953e-01, 9.99686636e-01, 7.55083725e-01, 2.75949010e-01,
      2.79673751e-03, 1.86761847e-01, 6.58707666e-01, 9.84729384e-01, 8.65084138e-01, 4.09773215e-01, 3.74073776e-02,
      9.03380600e-02, 5.19901037e-01,
    };

    {
        auto copy = data;
        minmax_scale<double>(copy, 0.0, 1.0);
        assert(copy.size() == expected.size());
        for (size_t i = 0; i < copy.size(); ++i) {
            double diff = std::abs(copy[i] - expected[i]);
            assert(diff < 1e-8);
        }
    }

    {
        minmax_scale_ctx_t<float> ctx;
        minmax_scale(ctx, data, 0.0f, 1.0f);
        auto r = ctx.result;
        assert(r.size() == expected.size());
        for (size_t i = 0; i < r.size(); ++i) {
            double diff = std::abs(r[i] - expected[i]);
            assert(diff < 1e-7);
        }
    }
}

void test_mtf() {
    std::vector<double> data = {
      0.,
      0.09983342,
      0.19866933,
      0.29552021,
      0.38941834,
      0.47942554,
      0.56464247,
      0.64421769,
      0.71735609,
      0.78332691,
    };
    std::vector<double> expected = {
      0.66666667, 0.66666667, 0.66666667, 0.33333333, 0.33333333, 0.33333333, 0.,         0.,         0.,
      0.,         0.66666667, 0.66666667, 0.66666667, 0.33333333, 0.33333333, 0.33333333, 0.,         0.,
      0.,         0.,         0.66666667, 0.66666667, 0.66666667, 0.33333333, 0.33333333, 0.33333333, 0.,
      0.,         0.,         0.,         0.,         0.,         0.,         0.66666667, 0.66666667, 0.66666667,
      0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.,         0.,         0.,         0.66666667, 0.66666667,
      0.66666667, 0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.,         0.,         0.,         0.66666667,
      0.66666667, 0.66666667, 0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.,         0.,         0.,
      0.,         0.,         0.,         1.,         1.,         1.,         1.,         0.,         0.,
      0.,         0.,         0.,         0.,         1.,         1.,         1.,         1.,         0.,
      0.,         0.,         0.,         0.,         0.,         1.,         1.,         1.,         1.,
      0.,         0.,         0.,         0.,         0.,         0.,         1.,         1.,         1.,
      1.,
    };

    {
        auto r = mtf<double>(data, 4);
        assert(r.size() == expected.size());
        for (size_t i = 0; i < r.size(); ++i) {
            auto diff = std::abs(r[i] - expected[i]);
            assert(diff < 1e-8);
        }
    }

    {
        auto r = mtf<float>(data, 4);
        assert(r.size() == expected.size());
        for (size_t i = 0; i < r.size(); ++i) {
            auto diff = std::abs(r[i] - expected[i]);
            assert(diff < 1e-7);
        }
    }
}

void test_resize() {
    std::vector<double> data = {
      0.,         0.1010101,  0.2020202,  0.3030303,  0.4040404,  0.50505051, 0.60606061, 0.70707071, 0.80808081,
      0.90909091, 1.01010101, 1.11111111, 1.21212121, 1.31313131, 1.41414141, 1.51515152, 1.61616162, 1.71717172,
      1.81818182, 1.91919192, 2.02020202, 2.12121212, 2.22222222, 2.32323232, 2.42424242, 2.52525253, 2.62626263,
      2.72727273, 2.82828283, 2.92929293, 3.03030303, 3.13131313, 3.23232323, 3.33333333, 3.43434343, 3.53535354,
      3.63636364, 3.73737374, 3.83838384, 3.93939394, 4.04040404, 4.14141414, 4.24242424, 4.34343434, 4.44444444,
      4.54545455, 4.64646465, 4.74747475, 4.84848485, 4.94949495, 5.05050505, 5.15151515, 5.25252525, 5.35353535,
      5.45454545, 5.55555556, 5.65656566, 5.75757576, 5.85858586, 5.95959596, 6.06060606, 6.16161616, 6.26262626,
      6.36363636, 6.46464646, 6.56565657, 6.66666667, 6.76767677, 6.86868687, 6.96969697, 7.07070707, 7.17171717,
      7.27272727, 7.37373737, 7.47474747, 7.57575758, 7.67676768, 7.77777778, 7.87878788, 7.97979798, 8.08080808,
      8.18181818, 8.28282828, 8.38383838, 8.48484848, 8.58585859, 8.68686869, 8.78787879, 8.88888889, 8.98989899,
      9.09090909, 9.19191919, 9.29292929, 9.39393939, 9.49494949, 9.5959596,  9.6969697,  9.7979798,  9.8989899,
      10.,
    };
    std::vector<int16_t> data_shape = {10, 10};
    std::vector<double>  expected   = {
         1.2962963,
         1.63299663,
         1.96969697,
         4.66329966,
         5.,
         5.33670034,
         8.03030303,
         8.36700337,
         8.7037037,
    };
    std::vector<int16_t> expected_shape = {3, 3};

    {
        auto r = resize<double>(data, data_shape, expected_shape);
        assert(r.size() == expected.size());
        for (size_t i = 0; i < r.size(); ++i) {
            double diff = std::abs(r[i] - expected[i]);
            assert(diff < 1e-8);
        }
    }

    {
        auto r = resize<float>(data, data_shape, expected_shape);
        assert(r.size() == expected.size());
        for (size_t i = 0; i < r.size(); ++i) {
            double diff = std::abs(r[i] - expected[i]);
            assert(diff < 1e-6);
        }
    }
}

void test_psnr() {
    std::vector<uint8_t> original = {
      0,   2,   5,   7,   10,  12,  15,  18,  20,  23,  25,  28,  30,  33,  36,  38,  41,  43,  46,  48,
      51,  54,  56,  59,  61,  64,  66,  69,  72,  74,  77,  79,  82,  85,  87,  90,  92,  95,  97,  100,
      103, 105, 108, 110, 113, 115, 118, 121, 123, 126, 128, 131, 133, 136, 139, 141, 144, 146, 149, 151,
      154, 157, 159, 162, 164, 167, 170, 172, 175, 177, 180, 182, 185, 188, 190, 193, 195, 198, 200, 203,
      206, 208, 211, 213, 216, 218, 221, 224, 226, 229, 231, 234, 236, 239, 242, 244, 247, 249, 252, 255,
    };
    std::vector<uint8_t> compressed = {
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
      50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
      75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100,
    };
    double expected = 9.059609569978468;

    {
        auto r    = psnr<double>(original, compressed);
        auto diff = std::abs(r - expected);
        assert(diff < EPS);
    }

    {
        auto r    = psnr<float>(original, compressed);
        auto diff = std::abs(r - expected);
        assert(diff < 1e-6);
    }
}

void test_integrate_wavelet() {
    std::vector<double> expected_psi = {
      -5.29958527e-16, -1.05981018e-15, -1.28025089e-15, -5.42880983e-16, 2.33023500e-15,  9.25101796e-15,
      2.29968009e-14,  4.70885454e-14,  8.50975089e-14,  1.38883521e-13,  2.05080119e-13,  2.69022645e-13,
      2.95416632e-13,  2.15616472e-13,  -8.72069396e-14, -7.91028836e-13, -2.13762447e-12, -4.40999208e-12,
      -7.86139656e-12, -1.25642984e-11, -1.81416986e-11, -2.33464123e-11, -2.54746561e-11, -1.96516306e-11,
      1.87684328e-12,  5.01578120e-11,  1.39142574e-10,  2.83720241e-10,  4.95126440e-10,  7.72504563e-10,
      1.08948355e-09,  1.37522707e-09,  1.49081050e-09,  1.20431888e-09,  1.71975835e-10,  -2.06208357e-09,
      -6.02855901e-09, -1.22334779e-08, -2.09682507e-08, -3.20036556e-08, -4.41552101e-08, -5.47362737e-08,
      -5.89646759e-08, -4.94611531e-08, -1.60707770e-08, 5.36590965e-08,  1.72921770e-07,  3.52553588e-07,
      5.95996746e-07,  8.92148397e-07,  1.20637002e-06,  1.47064530e-06,  1.57485470e-06,  1.36230654e-06,
      6.33848519e-07,  -8.34226792e-07, -3.25311011e-06, -6.76097354e-06, -1.13377068e-05, -1.66987576e-05,
      -2.21795497e-05, -2.66315880e-05, -2.83619075e-05, -2.51569667e-05, -1.44375009e-05, 6.41139045e-06,
      3.95043693e-05,  8.57120677e-05,  1.43751842e-04,  2.09214799e-04,  2.73699068e-04,  3.24285867e-04,
      3.43644275e-04,  3.11063471e-04,  2.04667153e-04,  4.94665423e-06,  -3.00452147e-04, -7.11032565e-04,
      -1.20751961e-03, -1.74672136e-03, -2.25848283e-03, -2.64617967e-03, -2.79211776e-03, -2.56882682e-03,
      -1.85652457e-03, -5.65997147e-04, 1.33512174e-03,  3.79602394e-03,  6.66087630e-03,  9.65670511e-03,
      1.23962908e-02,  1.44004162e-02,  1.51418152e-02,  1.41101578e-02,  1.08936420e-02,  5.26871848e-03,
      -2.71419945e-03, -1.26637324e-02, -2.38146009e-02, -3.50423518e-02, -4.49348469e-02, -5.19225721e-02,
      -5.44608897e-02, -5.12476618e-02, -4.14507287e-02, -2.49131982e-02, -2.30195463e-03, 2.48326340e-02,
      5.41096816e-02,  8.24940131e-02,  1.06589217e-01,  1.23022555e-01,  1.28879542e-01,  1.22133101e-01,
      1.02006195e-01,  6.92095655e-02,  2.60081888e-02,  -2.39096743e-02, -7.57607791e-02, -1.24164054e-01,
      -1.63751668e-01, -1.89818936e-01, -1.98927685e-01, -1.89379531e-01, -1.61489711e-01, -1.17617235e-01,
      -6.19395671e-02, 4.67066711e-06,  6.19489084e-02,  1.17626576e-01,  1.61499053e-01,  1.89388872e-01,
      1.98937027e-01,  1.89828277e-01,  1.63761009e-01,  1.24173395e-01,  7.57701205e-02,  2.39190156e-02,
      -2.59988474e-02, -6.92002242e-02, -1.01996854e-01, -1.22123760e-01, -1.28870201e-01, -1.23013214e-01,
      -1.06579876e-01, -8.24846718e-02, -5.41003403e-02, -2.48232927e-02, 2.31129596e-03,  2.49225396e-02,
      4.14600700e-02,  5.12570032e-02,  5.44702310e-02,  5.19319134e-02,  4.49441882e-02,  3.50516931e-02,
      2.38239422e-02,  1.26730738e-02,  2.72354079e-03,  -5.25937714e-03, -1.08843006e-02, -1.41008165e-02,
      -1.51324739e-02, -1.43910748e-02, -1.23869494e-02, -9.64736377e-03, -6.65153497e-03, -3.78668261e-03,
      -1.32578041e-03, 5.75338481e-04,  1.86586591e-03,  2.57816815e-03,  2.80145910e-03,  2.65552101e-03,
      2.26782416e-03,  1.75606269e-03,  1.21686095e-03,  7.20373899e-04,  3.09793481e-04,  4.39467998e-06,
      -1.95325819e-04, -3.01722137e-04, -3.34302941e-04, -3.14944532e-04, -2.64357734e-04, -1.99873464e-04,
      -1.34410507e-04, -7.63707334e-05, -3.01630351e-05, 2.92994376e-06,  2.37788351e-05,  3.44983009e-05,
      3.77032417e-05,  3.59729222e-05,  3.15208840e-05,  2.60400918e-05,  2.06790410e-05,  1.61023078e-05,
      1.25944443e-05,  1.01755610e-05,  8.70748569e-06,  7.97902768e-06,  7.76647952e-06,  7.87068891e-06,
      8.13496419e-06,  8.44918582e-06,  8.74533747e-06,  8.98878063e-06,  9.16841244e-06,  9.28767512e-06,
      9.35740499e-06,  9.39079537e-06,  9.40029889e-06,  9.39607049e-06,  9.38548942e-06,  9.37333787e-06,
      9.36230246e-06,  9.35356769e-06,  9.34736277e-06,  9.34339630e-06,  9.34116224e-06,  9.34012989e-06,
      9.33984340e-06,  9.33995899e-06,  9.34024473e-06,  9.34056171e-06,  9.34083909e-06,  9.34105049e-06,
      9.34119507e-06,  9.34128406e-06,  9.34133234e-06,  9.34135386e-06,  9.34135969e-06,  9.34135756e-06,
      9.34135236e-06,  9.34134678e-06,  9.34134207e-06,  9.34133862e-06,  9.34133635e-06,  9.34133500e-06,
      9.34133430e-06,  9.34133400e-06,  9.34133392e-06,  9.34133394e-06,  9.34133401e-06,  9.34133407e-06,
      9.34133413e-06,  9.34133417e-06,  9.34133419e-06,  9.34133420e-06,  9.34133421e-06,  9.34133421e-06,
      9.34133421e-06,  9.34133421e-06,  9.34133421e-06,  9.34133421e-06,
    };
    std::vector<double> expected_x = {
      -8.,         -7.9372549,  -7.8745098,  -7.81176471, -7.74901961, -7.68627451, -7.62352941, -7.56078431,
      -7.49803922, -7.43529412, -7.37254902, -7.30980392, -7.24705882, -7.18431373, -7.12156863, -7.05882353,
      -6.99607843, -6.93333333, -6.87058824, -6.80784314, -6.74509804, -6.68235294, -6.61960784, -6.55686275,
      -6.49411765, -6.43137255, -6.36862745, -6.30588235, -6.24313725, -6.18039216, -6.11764706, -6.05490196,
      -5.99215686, -5.92941176, -5.86666667, -5.80392157, -5.74117647, -5.67843137, -5.61568627, -5.55294118,
      -5.49019608, -5.42745098, -5.36470588, -5.30196078, -5.23921569, -5.17647059, -5.11372549, -5.05098039,
      -4.98823529, -4.9254902,  -4.8627451,  -4.8,        -4.7372549,  -4.6745098,  -4.61176471, -4.54901961,
      -4.48627451, -4.42352941, -4.36078431, -4.29803922, -4.23529412, -4.17254902, -4.10980392, -4.04705882,
      -3.98431373, -3.92156863, -3.85882353, -3.79607843, -3.73333333, -3.67058824, -3.60784314, -3.54509804,
      -3.48235294, -3.41960784, -3.35686275, -3.29411765, -3.23137255, -3.16862745, -3.10588235, -3.04313725,
      -2.98039216, -2.91764706, -2.85490196, -2.79215686, -2.72941176, -2.66666667, -2.60392157, -2.54117647,
      -2.47843137, -2.41568627, -2.35294118, -2.29019608, -2.22745098, -2.16470588, -2.10196078, -2.03921569,
      -1.97647059, -1.91372549, -1.85098039, -1.78823529, -1.7254902,  -1.6627451,  -1.6,        -1.5372549,
      -1.4745098,  -1.41176471, -1.34901961, -1.28627451, -1.22352941, -1.16078431, -1.09803922, -1.03529412,
      -0.97254902, -0.90980392, -0.84705882, -0.78431373, -0.72156863, -0.65882353, -0.59607843, -0.53333333,
      -0.47058824, -0.40784314, -0.34509804, -0.28235294, -0.21960784, -0.15686275, -0.09411765, -0.03137255,
      0.03137255,  0.09411765,  0.15686275,  0.21960784,  0.28235294,  0.34509804,  0.40784314,  0.47058824,
      0.53333333,  0.59607843,  0.65882353,  0.72156863,  0.78431373,  0.84705882,  0.90980392,  0.97254902,
      1.03529412,  1.09803922,  1.16078431,  1.22352941,  1.28627451,  1.34901961,  1.41176471,  1.4745098,
      1.5372549,   1.6,         1.6627451,   1.7254902,   1.78823529,  1.85098039,  1.91372549,  1.97647059,
      2.03921569,  2.10196078,  2.16470588,  2.22745098,  2.29019608,  2.35294118,  2.41568627,  2.47843137,
      2.54117647,  2.60392157,  2.66666667,  2.72941176,  2.79215686,  2.85490196,  2.91764706,  2.98039216,
      3.04313725,  3.10588235,  3.16862745,  3.23137255,  3.29411765,  3.35686275,  3.41960784,  3.48235294,
      3.54509804,  3.60784314,  3.67058824,  3.73333333,  3.79607843,  3.85882353,  3.92156863,  3.98431373,
      4.04705882,  4.10980392,  4.17254902,  4.23529412,  4.29803922,  4.36078431,  4.42352941,  4.48627451,
      4.54901961,  4.61176471,  4.6745098,   4.7372549,   4.8,         4.8627451,   4.9254902,   4.98823529,
      5.05098039,  5.11372549,  5.17647059,  5.23921569,  5.30196078,  5.36470588,  5.42745098,  5.49019608,
      5.55294118,  5.61568627,  5.67843137,  5.74117647,  5.80392157,  5.86666667,  5.92941176,  5.99215686,
      6.05490196,  6.11764706,  6.18039216,  6.24313725,  6.30588235,  6.36862745,  6.43137255,  6.49411765,
      6.55686275,  6.61960784,  6.68235294,  6.74509804,  6.80784314,  6.87058824,  6.93333333,  6.99607843,
      7.05882353,  7.12156863,  7.18431373,  7.24705882,  7.30980392,  7.37254902,  7.43529412,  7.49803922,
      7.56078431,  7.62352941,  7.68627451,  7.74901961,  7.81176471,  7.8745098,   7.9372549,   8.,
    };

    {
        auto [psi, x] = integrate_wavelet<double>(cwt_wavelet_t::MORLET, 8);
        assert(psi.size() == expected_psi.size());
        assert(x.size() == expected_x.size());
        for (size_t i = 0; i < psi.size(); ++i) {
            auto diff = std::abs(psi[i] - expected_psi[i]);
            assert(diff < 1e-9);
        }
        for (size_t i = 0; i < x.size(); ++i) {
            auto diff = std::abs(x[i] - expected_x[i]);
            assert(diff < 1e-8);
        }
    }

    {
        auto [psi, x] = integrate_wavelet<float>(cwt_wavelet_t::MORLET, 8);
        assert(psi.size() == expected_psi.size());
        assert(x.size() == expected_x.size());
        for (size_t i = 0; i < psi.size(); ++i) {
            auto diff = std::abs(psi[i] - expected_psi[i]);
            assert(diff < 1e-6);
        }
        for (size_t i = 0; i < x.size(); ++i) {
            auto diff = std::abs(x[i] - expected_x[i]);
            assert(diff < 1e-6);
        }
    }
}

void test_cwt() {
    std::vector<double> data = {
      1.,
      2.,
      3.,
      4.,
      5.,
      6.,
      7.,
      8.,
      9.,
      10.,
    };

    {
        auto [psi, x] = integrate_wavelet<double>(cwt_wavelet_t::MORLET, 10);
        cwt_ctx_t<double> ctx;
        cwt(ctx, data, data, psi, x);
    }
}

#endif

}  // namespace tests

}  // namespace dsp
