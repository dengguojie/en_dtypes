/* Copyright 2024 The en_dtypes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef EN_DTYPES_HIFLOAT_H_
#define EN_DTYPES_HIFLOAT_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>
#include <type_traits>
#include <utility>

#ifdef __has_include
#if __has_include(<version>)
#include <version>
#endif
#endif

#if (defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L)
#include <bit>
#endif

#include "Eigen/Core"
#include "common_inc.h"

namespace en_dtypes {
namespace hifloat_internal {

class hifloat8 {
 protected:
  // Constructor tag to allow constexpr construction from bit representation.
  struct ConstructFromRepTag {};
  constexpr hifloat8(uint8_t rep, ConstructFromRepTag) : rep_(rep){}

 public:
  static constexpr uint8_t bits = 8;
  static constexpr uint8_t sign_mask = (((uint8_t)1) << (bits - 1));

  constexpr hifloat8() : rep_(0) {}

  template <typename T>
  explicit EIGEN_DEVICE_FUNC hifloat8(
      T i, std::enable_if_t<std::is_integral_v<T>, int> = 0)
      : hifloat8(ConvertFrom(static_cast<float>(i)).rep(),
                 ConstructFromRepTag{}) {}
  template <typename T>
  explicit EIGEN_DEVICE_FUNC hifloat8(
      T f, std::enable_if_t<std::is_floating_point_v<T>, int> = 0)
      : hifloat8(ConvertFrom(f).rep(), ConstructFromRepTag{}) {}
  explicit EIGEN_DEVICE_FUNC hifloat8(Eigen::bfloat16 bf16)
      : hifloat8(ConvertFrom(bf16).rep(), ConstructFromRepTag{}) {}
  explicit EIGEN_DEVICE_FUNC hifloat8(Eigen::half f16)
      : hifloat8(ConvertFrom(f16).rep(), ConstructFromRepTag{}) {}

  constexpr uint8_t rep() const { return rep_; }

  template <typename T,
            typename EnableIf = std::enable_if<std::is_arithmetic_v<T>>>
  explicit EIGEN_DEVICE_FUNC operator T() const {
    return static_cast<T>(static_cast<float>(derived()));
  }
  explicit EIGEN_DEVICE_FUNC operator double() const {
    return ConvertTo<double>(derived());
  }
  explicit EIGEN_DEVICE_FUNC operator float() const {
    return ConvertTo<float>(derived());
  }
  explicit EIGEN_DEVICE_FUNC operator Eigen::bfloat16() const {
    return ConvertTo<Eigen::bfloat16>(derived());
  }
  explicit EIGEN_DEVICE_FUNC operator Eigen::half() const {
    return ConvertTo<Eigen::half>(derived());
  }
  explicit EIGEN_DEVICE_FUNC operator bool() const {
    // 0b1'0000'000 is for NaN. 0b0'0000'000 if zero.
    return (rep()) != 0;
  }

  constexpr hifloat8 operator-() const {
    // NaN & Zero
    if ((rep() & 0x7f) == 0x00) {
      return *this;
    }
    return hifloat8(static_cast<uint8_t>(rep() ^ sign_mask), ConstructFromRepTag{});
  }

  constexpr const hifloat8& derived() const {
    return *static_cast<const hifloat8*>(this);
  }

  constexpr hifloat8& derived() { return *static_cast<hifloat8*>(this); }

  static constexpr hifloat8 FromRep(uint8_t rep) {
    return hifloat8(rep, ConstructFromRepTag{});
  }

  // Conversions allowing saturation and truncation.
  template <bool kSaturate = false, bool kTruncate = false, typename From>
  static inline EIGEN_DEVICE_FUNC hifloat8 ConvertFrom(From from);

  template <typename To, bool kSaturate = false, bool kTruncate = false>
  static inline EIGEN_DEVICE_FUNC To ConvertTo(hifloat8 from);

  // Operators via float32.
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC hifloat8
  operator+(const hifloat8& other) const {
    return hifloat8{float{derived()} + float{other}};
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC hifloat8
  operator-(const hifloat8& other) const {
    return hifloat8{float{derived()} - float{other}};
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC hifloat8
  operator*(const hifloat8& other) const {
    return hifloat8{float{derived()} * float{other}};
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC hifloat8
  operator/(const hifloat8& other) const {
    return hifloat8{float{derived()} / float{other}};
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator==(
    const hifloat8& other) const {
    return Compare(derived(), other) == Ordering::kEquivalent;
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator!=(const hifloat8& other) const {
    return Compare(derived(), other) != Ordering::kEquivalent;
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator<(
      const hifloat8& other) const {
    return Compare(derived(), other) == Ordering::kLess;
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator<=(
      const hifloat8& other) const {
    return Compare(derived(), other) <= Ordering::kEquivalent;
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator>(
      const hifloat8& other) const {
    return Compare(derived(), other) == Ordering::kGreater;
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator>=(
      const hifloat8& other) const {
    Ordering ordering = Compare(derived(), other);
    return ordering == Ordering::kGreater || ordering == Ordering::kEquivalent;
  }

  // Compound assignment.
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC hifloat8& operator+=(
      const hifloat8& other) {
    derived() = derived() + other;
    return derived();
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC hifloat8& operator-=(
      const hifloat8& other) {
    derived() = derived() - other;
    return derived();
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC hifloat8& operator*=(
      const hifloat8& other) {
    derived() = derived() * other;
    return derived();
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC hifloat8& operator/=(
      const hifloat8& other) {
    derived() = derived() / other;
    return derived();
  }

 private:
  static EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC std::pair<uint8_t, uint8_t>
  SignAndMagnitude(hifloat8 x) {
    const uint8_t x_bits = x.rep();
    const uint8_t x_abs_bits = x_bits ^ sign_mask;
    const uint8_t x_sign = x_bits ^ x_abs_bits;
    return {x_sign, x_abs_bits};
  }

  static EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC int16_t
  SignDotAndMagnitudeComplete(const uint8_t x_sign, const uint8_t x_abs_bits) {
    uint8_t x_dot = x_abs_bits & 0x60;
    uint8_t x_exp_sign = x_abs_bits & 0x10;
    uint8_t x_exp_mantissa = x_abs_bits & 0x0F;
    if (x_dot == 0) { // 0bS001 / 0bS0001 / 0bS0000
      x_dot = x_abs_bits & 0x10;
      x_exp_sign = 0;
      x_exp_mantissa = x_abs_bits & 0x07;
      if (x_dot == 0) { // 0bS0001 / 0bS0000
        x_dot = x_abs_bits & 0x08;
      } else { // 0bS001
        x_exp_sign = (x_abs_bits & 0x08) << 1;
      }
    }
    return ((x_dot << 1) | x_exp_mantissa) ^ (
      static_cast<int16_t>(x_sign ^ (x_exp_sign << 3)) < 0 ? -1 : 0
    );
  }

  enum Ordering : int8_t {
    kLess = -1,
    kEquivalent = 0,
    kGreater = 1,
    kUnordered = 2,
  };

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC friend Ordering Compare(
      const hifloat8& lhs, const hifloat8& rhs) {
    // nan
    if (Eigen::numext::isnan(lhs) || Eigen::numext::isnan(rhs)) {
      return Ordering::kUnordered;
    }
    // zero
    auto [lhs_sign, lhs_mag] = SignAndMagnitude(lhs);
    auto [rhs_sign, rhs_mag] = SignAndMagnitude(rhs);
    if (lhs_mag == 0 && rhs_mag == 0) {
      return Ordering::kEquivalent;
    }

    if (lhs_sign != rhs_sign) {
      return lhs_sign == 0 ? Ordering::kGreater : Ordering::kLess;
    }

    if (lhs_mag == rhs_mag) {
      return Ordering::kEquivalent;
    }

    // inf: 0b0'11'0'111'1 = 0x6F
    if (lhs_mag == 0x6F) {
      return lhs_sign == 0 ? Ordering::kGreater : Ordering::kLess;
    }
    if (rhs_mag == 0x6F) {
      return rhs_sign == 0 ? Ordering::kLess : Ordering::kGreater;
    }

    // others
    int16_t lhs_twos_complement = SignDotAndMagnitudeComplete(lhs_sign, lhs_mag);
    int16_t rhs_twos_complement = SignDotAndMagnitudeComplete(rhs_sign, rhs_mag);
    return lhs_twos_complement < rhs_twos_complement ? Ordering::kLess : Ordering::kGreater;
  }

  uint8_t rep_;
};

constexpr double ConstexprAbs(double x) { return x < 0.0 ? -x : x; }

constexpr double ConstexprCeil(double x) {
  constexpr double kIntegerThreshold =
      uint64_t{1} << (std::numeric_limits<double>::digits - 1);
  // Too big or NaN inputs get returned unchanged.
  if (!(ConstexprAbs(x) < kIntegerThreshold)) {
    return x;
  }
  const double x_trunc = static_cast<double>(static_cast<int64_t>(x));
  return x_trunc < x ? x_trunc + 1.0 : x_trunc;
}

constexpr double ConstexprFloor(double x) { return -ConstexprCeil(-x); }

constexpr double kLog10Of2 = 0.3010299956639812;
// C17 5.2.4.2.2p11:
// "number of decimal digits, q, such that any floating-point number with q
// decimal digits can be rounded into a floating-point number with p radix b
// digits and back again without change to the q decimal digits"
// floor((p - 1) * log10(2));
constexpr int Digits10FromDigits(int digits) {
  return static_cast<int>(ConstexprFloor((digits - 1) * kLog10Of2));
}

// C17 5.2.4.2.2p11:
// "number of decimal digits, n, such that any floating-point number with p
// radix b digits can be rounded to a floating-point number with n decimal
// digits and back again without change to the value"
// ceil(1 + p * log10(2));
constexpr int MaxDigits10FromDigits(int digits) {
  return static_cast<int>(ConstexprCeil(1.0 + (digits * kLog10Of2)));
}

// C17 5.2.4.2.2p11:
// "minimum negative integer such that 10 raised to that power is in the range
// of normalized floating-point numbers"
// ceil(log10(2**(emin - 1))) == ceil((emin - 1) * log10(2));
constexpr int MinExponent10FromMinExponent(int min_exponent) {
  return static_cast<int>(ConstexprCeil((min_exponent - 1) * kLog10Of2));
}

// C17 5.2.4.2.2p11:
// "maximum integer such that 10 raised to that power is in the range of
// representable finite floating-point numbers"
// floor(log10((1 - 2**-p) * 2**emax)) == floor(log10(1 - 2**-p) +
// emax * log10(2))
constexpr int MaxExponent10FromMaxExponentAndDigits(int max_exponent,
                                                    int digits) {
  constexpr double kLog10OfOnePredecessor[] = {
      // log10(1 - 2**-4)
      -0.028028723600243537,
  };
  return static_cast<int>(ConstexprFloor(kLog10OfOnePredecessor[digits - 4] +
                                         max_exponent * kLog10Of2));
}

// Structures for use in specializing std::numeric_limits.
struct numeric_limits_hifloat8 {
 private:
  static inline constexpr const int kMantissaBits = 3;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const bool is_specialized = true;
  static inline constexpr const bool is_signed = true;
  static inline constexpr const bool is_integer = false;
  static inline constexpr const bool is_exact = false;
  static inline constexpr const bool has_quiet_NaN = true;
  static inline constexpr const std::float_denorm_style has_denorm =
      std::denorm_present;
  static inline constexpr const bool has_denorm_loss = false;
  static inline constexpr const std::float_round_style round_style =
      std::round_to_nearest;
  static inline constexpr const bool is_bounded = true;
  static inline constexpr const bool is_modulo = false;
  static inline constexpr const int radix = std::numeric_limits<float>::radix;
  static inline constexpr const bool traps = std::numeric_limits<float>::traps;
  static inline constexpr const bool tinyness_before =
      std::numeric_limits<float>::tinyness_before;

  static inline constexpr const bool is_iec559 = false;
  static inline constexpr const bool has_infinity = true;
  static inline constexpr const bool has_signaling_NaN = true;

  static inline constexpr const int digits = kMantissaBits + 1;
  static inline constexpr const int digits10 = Digits10FromDigits(digits);
  static inline constexpr const int max_digits10 =
      MaxDigits10FromDigits(digits);
  static inline constexpr const int min_exponent = -14;
  static inline constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static inline constexpr const int max_exponent = 16;
  static inline constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  // NOLINTEND

  // 0b0'11'1111'0 = 2 ^ (-15) (min normal)
  static constexpr hifloat8 min() {
    return hifloat8::FromRep(0b0'11'1111'0);
  }
  // 0b1'11'0111'0 = -32768
  static constexpr hifloat8 lowest() {
    return hifloat8::FromRep(0b1'11'0111'0);
  }
  // 0b0'11'0111'0 = 32768
  static constexpr hifloat8 max() {
    return hifloat8::FromRep(0b0'11'0111'0);
  }
  // 0b0'01'11'000 = 0.125, difference between 1.0 and the next representable value.
  static constexpr hifloat8 epsilon() {
    return hifloat8::FromRep(0b0'01'11'000);
  }
  // 0b0'001'1'000 = 0.5
  // Returns the largest possible rounding error in ULPs (units in the last place) as defined by ISO 10967,
  // which can vary from 0.5 (rounding to the nearest digit) to 1.0 (rounding to zero or to infinity).
  static constexpr hifloat8 round_error() {
    return hifloat8::FromRep(0b0'001'1'000);
  }
  static constexpr hifloat8 infinity() {
    return hifloat8::FromRep(0b0'11'0111'1);
  }
  // NaN.
  static constexpr hifloat8 quiet_NaN() {
    return hifloat8::FromRep(0b1'0000'000);
  }
  static constexpr hifloat8 signaling_NaN() {
    return hifloat8::FromRep(0b1'0000'000);
  }
  // 0b0'0000'001 = 2^-22  (min denormal)
  static constexpr hifloat8 denorm_min() {
    return hifloat8::FromRep(0b0'0000'001);
  }
};

} // namespace hifloat_internal
// Exported types.
using hifloat8 = hifloat_internal::hifloat8;
} // namespace en_dtypes

namespace std {
// Standard-library overrides.  Note that these are picked up by Eigen as well.
template <>
struct numeric_limits<en_dtypes::hifloat_internal::hifloat8>
    : public en_dtypes::hifloat_internal::numeric_limits_hifloat8 {};
}  // namespace std

namespace en_dtypes {
namespace convert_internal {
template <typename To, bool kSaturate, bool kTruncate>
struct ConvertImpl</*From=*/hifloat8, To, kSaturate, kTruncate,
                    std::enable_if_t<!std::is_same_v<hifloat8, To>>> {
  static EIGEN_DEVICE_FUNC To run(hifloat8 from) {
    // special values
    if (Eigen::numext::isnan(from)) {
      return Eigen::NumTraits<To>::quiet_NaN();
    }

    const bool from_sign_bit = Eigen::numext::bit_cast<uint8_t>(from) >> 7;
    if (Eigen::numext::isinf(from)) {
      return from_sign_bit ? -Eigen::NumTraits<To>::infinity()
                          : Eigen::NumTraits<To>::infinity();
    }

    /*
      When From=hifloat8, we convert hifloat8 -> float -> To
    */
    const uint8_t from_abs_bits = Eigen::numext::bit_cast<uint8_t>(from) & 0x7F;
    if (from_abs_bits == 0) {
      return To{};
    }

    uint8_t exp = 0;
    uint8_t exp_sign = 0;
    uint8_t dot = from_abs_bits & 0x60;
    uint32_t float_bits = (Eigen::numext::bit_cast<uint8_t>(from) & 0x80) << 24;
    uint8_t mantissa = from_abs_bits & 0x07;
    switch(dot) {
      case 0x60:
        exp_sign = from_abs_bits & 0x10;
        exp = ((from_abs_bits & 0x0E) | 0x10) >> 1;
        mantissa = (from_abs_bits & 0x01) << 2;
        break;
      case 0x40:
        exp_sign = from_abs_bits & 0x10;
        exp = ((from_abs_bits & 0x0C) | 0x10) >> 2;
        mantissa = (from_abs_bits & 0x03) << 1;
        break;
      case 0x20:
        exp_sign = from_abs_bits & 0x10;
        exp = ((from_abs_bits & 0x08) | 0x10) >> 3;
        break;
      default:
        dot = from_abs_bits & 0x10;
        if (dot == 0) {
          dot = from_abs_bits & 0x08;
          if (dot == 0) {
            // subnormal
            exp_sign = 1;
            exp = 23 - mantissa; // bias in subnormal
            mantissa = 0;
          }
        } else {
          // 0x10:
          exp_sign = from_abs_bits & 0x08;
          exp = 1;
        }
        break;
    }

    const uint8_t float_exp = exp_sign > 0 ? 127 - exp : 127 + exp;
    float_bits |= ((float_exp << 23) | (mantissa << 20));
    float float_v = Eigen::numext::bit_cast<float>(float_bits);
    if constexpr (std::is_same_v<To, float>) {
      return float_v;
    } else {
      return ConvertImpl<float, To, kSaturate, kTruncate>::run(float_v);
    }
  }
};

template <typename From, bool kSaturate, bool kTruncate>
struct ConvertImpl<From, /*To=*/hifloat8, kSaturate, kTruncate,
                    std::enable_if_t<!std::is_same_v<From, hifloat8>>> {
  using FromTraits = Traits<From>;
  using FromBits = typename FromTraits::BitsType;
  static constexpr int kFromBits = FromTraits::kBits;
  static constexpr int kFromMantissaBits = FromTraits::kMantissaBits;
  static constexpr int kFromExponentBias = FromTraits::kExponentBias;

  using ToTraits = Traits<hifloat8>;
  using ToBits = typename ToTraits::BitsType;
  static constexpr int kToBits = ToTraits::kBits;

  static EIGEN_DEVICE_FUNC hifloat8 run(From from) {
    const bool from_sign_bit =
        Eigen::numext::bit_cast<FromBits>(from) >> (kFromBits - 1);
    // inf
    if (Eigen::numext::isinf(from)) {
      return from_sign_bit ? -Eigen::NumTraits<hifloat8>::infinity()
                          : Eigen::NumTraits<hifloat8>::infinity();
    }
    // nan
    if (Eigen::numext::isnan(from)) {
      return Eigen::NumTraits<hifloat8>::quiet_NaN();
    }

    const FromBits from_bits =
        Eigen::numext::bit_cast<FromBits>(Eigen::numext::abs(from));
    // +0.0 / -0.0
    if (from_bits == 0) {
      // hifloat8 has only +0.0
      return hifloat8{};
    }

    /*
      When To=hifloat8, we convert From -> float -> hifloat8
      consider hifloat8 as float8_e5m3 without signbit.
    */
    if constexpr (std::is_same_v<From, float>) {
      using WideBits = GetUnsignedInteger<sizeof(float)>;
      const int unbiased_exponent = (from_bits >> kFromMantissaBits) - kFromExponentBias;
      if (unbiased_exponent > 15 || from_bits >= 0x47400000) {  // 49152 = 2**15*(1+0.5)
        // Overflowed values map to highest or infinity depending on kSaturate.
        hifloat8 to = kSaturate ? Eigen::NumTraits<hifloat8>::highest()
                                : Eigen::NumTraits<hifloat8>::infinity();
        return from_sign_bit ? -to : to;
      }

      if (unbiased_exponent < -23) {
        // hifloat8 has only +0.0
        return hifloat8{};
      }

      int kToMantissaBits = 0;
      const int abs_unbiased_exponent = abs(unbiased_exponent);
      if (abs_unbiased_exponent <= 3) {
        kToMantissaBits = 3;
      } else if (abs_unbiased_exponent <= 7) {
        kToMantissaBits = 2;
      } else if (abs_unbiased_exponent <= 15) {
        kToMantissaBits = 1;
      } else {
        kToMantissaBits = 0;
      }

      int kDigitShift = kToMantissaBits - kFromMantissaBits;

      // Round the mantissa if it is shrinking.
      WideBits rounded_from_bits = from_bits;
      if constexpr (!kTruncate) {
        // hifloat8 only support round half to away.
        rounded_from_bits = RoundBitsHalfToAway(from_bits, -kDigitShift);
      }
      // Zero-out tail bits.
      rounded_from_bits &= ~((WideBits{1} << (-kDigitShift)) - 1);

      const int unbiased_to_exponent = (rounded_from_bits >> kFromMantissaBits) - kFromExponentBias;
      const int abs_unbiased_to_exponent = abs(unbiased_to_exponent);
      int to_exponent_bits;
      ToBits to_bits;
      if (abs_unbiased_to_exponent <= 3) {
        kToMantissaBits = 3;
        if (abs_unbiased_to_exponent == 1) {
          to_bits = 0b0'001'0'000;
          to_exponent_bits = 1;
        } else if (abs_unbiased_to_exponent == 0) {
          to_bits = 0b0'0001'000;
          to_exponent_bits = 0;
        } else {
          to_bits = 0b0'01'00'000;
          to_exponent_bits = 2;
        }
      } else if (abs_unbiased_to_exponent <= 7) {
        kToMantissaBits = 2;
        to_bits = 0b0'10'000'00;
        to_exponent_bits = 3;
      } else if (abs_unbiased_to_exponent <= 15) {
        kToMantissaBits = 1;
        to_bits = 0b0'11'0000'0;
        to_exponent_bits = 4;
      } else {
        // subnormal
        kToMantissaBits = 0;
        to_bits = 0b0'0000'000;
        to_exponent_bits = 0;
      }

      if (kToMantissaBits > 0) {
        kDigitShift = kToMantissaBits - kFromMantissaBits;
        // extract hifloat8 mantissa.
        to_bits |= static_cast<ToBits>(
            (rounded_from_bits >> -kDigitShift) & ((ToBits{1} << kToMantissaBits) - 1)
          );
        // extract hifloat8 exponent.
        if (to_exponent_bits > 0) {
          // remove the implicit leading 1 bit of exponent.
          ToBits to_exponent = static_cast<ToBits>(abs_unbiased_to_exponent) & (
              (ToBits{1} << (to_exponent_bits - 1)) - 1
            );
          // insert exponent sign bit.
          if (unbiased_to_exponent < 0) {
            to_exponent |= (ToBits{1} << (to_exponent_bits - 1));
          }
          to_bits |= to_exponent << kToMantissaBits;
        }
      } else {
        // subnormal
        to_bits |= static_cast<ToBits>(23 + (unbiased_to_exponent == -23 ? -22 : unbiased_to_exponent));
      }

      hifloat8 to = Eigen::numext::bit_cast<hifloat8>(to_bits);
      // Insert sign bit.
      return from_sign_bit ? -to : to;
    } else {
      const float float_v = ConvertImpl<From, float, kSaturate, kTruncate>::run(from);
      return ConvertImpl<float, hifloat8, kSaturate, kTruncate>::run(float_v);
    }
  }
};
} // namespace convert_internal
} // namespace en_dtypes

namespace en_dtypes {
namespace hifloat_internal {

// Free-functions for use with ADL and in Eigen.
constexpr inline hifloat8 abs(const hifloat8& a) {
  return ((a.rep() & 0x7f) == 0x00) ? hifloat8::FromRep(a.rep())
                                  : hifloat8::FromRep(a.rep() & 0b0111'1111);
}

constexpr inline bool(isnan)(const hifloat8& a) {
  return a.rep() == std::numeric_limits<hifloat8>::quiet_NaN().rep();
}

constexpr inline bool(isinf)(const hifloat8& a) {
  return abs(a).rep() ==
            std::numeric_limits<hifloat8>::infinity().rep();
}

constexpr inline bool(isfinite)(const hifloat8& a) {
  return !isnan(a) && !isinf(a);
}

std::ostream& operator<<(std::ostream& os, const hifloat8& f) {
  os << static_cast<float>(f);
  return os;
}

//==============================================================================
// Inline conversion routines between hifloat8 and other types.
//==============================================================================
template <bool kSaturate, bool kTruncate, typename From>
EIGEN_DEVICE_FUNC hifloat8 hifloat8::ConvertFrom(const From from) {
  return en_dtypes::convert_internal::ConvertFrom<kSaturate, kTruncate, From, hifloat8>(from);
}

template <typename To, bool kSaturate, bool kTruncate>
EIGEN_DEVICE_FUNC To hifloat8::ConvertTo(hifloat8 from) {
  return en_dtypes::convert_internal::ConvertImpl</*From=*/hifloat8, To, kSaturate, kTruncate>::run(from);
}

} // namespace hifloat_internal

// Exported types.
//using hifloat8 = hifloat_internal::hifloat8;

}  // namespace en_dtypes



// Eigen-specific overrides.
namespace Eigen {
namespace numext {

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC en_dtypes::hifloat8
bit_cast<en_dtypes::hifloat8, uint8_t>(const uint8_t& src) {
  return en_dtypes::hifloat8::FromRep(src);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint8_t
bit_cast<uint8_t, en_dtypes::hifloat8>(const en_dtypes::hifloat8& src) {
  return src.rep();
}
}  // namespace numext

// Work-around for isinf/isnan/isfinite issue on aarch64.
namespace internal {
template <>
EIGEN_DEVICE_FUNC inline bool isinf_impl<en_dtypes::hifloat8>(
    const en_dtypes::hifloat8& x) {
  return en_dtypes::hifloat_internal::isinf(x);
}

template <>
EIGEN_DEVICE_FUNC inline bool isnan_impl<en_dtypes::hifloat8>(
    const en_dtypes::hifloat8& x) {
  return en_dtypes::hifloat_internal::isnan(x);
}

template <>
EIGEN_DEVICE_FUNC inline bool isfinite_impl<en_dtypes::hifloat8>(
    const en_dtypes::hifloat8& x) {
  return en_dtypes::hifloat_internal::isfinite(x);
}

}  // namespace internal
}  // namespace Eigen

#endif  // EN_DTYPES_HIFLOAT_H_

