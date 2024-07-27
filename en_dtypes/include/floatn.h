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

#ifndef EN_DTYPES_FLOATN_H_
#define EN_DTYPES_FLOATN_H_

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

namespace en_dtypes {
namespace floatn_internal {

// Forward-declarations of classes.
class float4_e2m1;
class float4_e1m2;
class float6_e2m3;
class float6_e3m2;
class float8_e8m0;


// Stores the n-bit float value in the low n bits of a byte.  The upper
// bits are left as 0(bit).
template <int N, typename Derived>
class floatn_base {
 protected:
  // Constructor tag to allow constexpr construction from bit representation.
  struct ConstructFromRepTag {};
  constexpr floatn_base(uint8_t rep, ConstructFromRepTag) : rep_(Mask(rep)){}

 public:
  static constexpr uint8_t bits = N;
  static constexpr uint8_t abs_mask = (((uint8_t)1) << N) - 1;
  static constexpr uint8_t sign_mask = (((uint8_t)1) << (N - 1));

  constexpr floatn_base() : rep_(0) {}

  template <typename T>
  explicit EIGEN_DEVICE_FUNC floatn_base(
      T i, std::enable_if_t<std::is_integral_v<T>, int> = 0)
      : floatn_base(ConvertFrom(static_cast<float>(i)).rep(),
                    ConstructFromRepTag{}) {}
  template <typename T>
  explicit EIGEN_DEVICE_FUNC floatn_base(
      T f, std::enable_if_t<std::is_floating_point_v<T>, int> = 0)
      : floatn_base(ConvertFrom(f).rep(), ConstructFromRepTag{}) {}
  explicit EIGEN_DEVICE_FUNC floatn_base(Eigen::bfloat16 bf16)
      : floatn_base(ConvertFrom(bf16).rep(), ConstructFromRepTag{}) {}
  explicit EIGEN_DEVICE_FUNC floatn_base(Eigen::half f16)
      : floatn_base(ConvertFrom(f16).rep(), ConstructFromRepTag{}) {}

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
    return (rep() & abs_mask) != 0;
  }

  constexpr Derived operator-() const {
    return Derived(static_cast<uint8_t>(rep() ^ sign_mask), ConstructFromRepTag{});
  }

  constexpr const Derived& derived() const {
    return *static_cast<const Derived*>(this);
  }

  constexpr Derived& derived() { return *static_cast<Derived*>(this); }

  static constexpr Derived FromRep(uint8_t rep) {
    return Derived(rep, ConstructFromRepTag{});
  }

  // Conversions allowing saturation and truncation.
  template <bool kSaturate = false, bool kTruncate = false, typename From>
  static inline EIGEN_DEVICE_FUNC Derived ConvertFrom(From from);

  template <typename To, bool kSaturate = false, bool kTruncate = false>
  static inline EIGEN_DEVICE_FUNC To ConvertTo(Derived from);

  // Operators via float32.
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived
  operator+(const Derived& other) const {
    return Derived{float{derived()} + float{other}};
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived
  operator-(const Derived& other) const {
    return Derived{float{derived()} - float{other}};
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived
  operator*(const Derived& other) const {
    return Derived{float{derived()} * float{other}};
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived
  operator/(const Derived& other) const {
    return Derived{float{derived()} / float{other}};
  }

  constexpr bool operator==(const Derived& other) const {
    return Compare(derived(), other) == Ordering::kEquivalent;
  }

  constexpr bool operator!=(const Derived& other) const {
    return Compare(derived(), other) != Ordering::kEquivalent;
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator<(
      const Derived& other) const {
    return Compare(derived(), other) == Ordering::kLess;
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator<=(
      const Derived& other) const {
    return Compare(derived(), other) <= Ordering::kEquivalent;
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator>(
      const Derived& other) const {
    return Compare(derived(), other) == Ordering::kGreater;
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator>=(
      const Derived& other) const {
    Ordering ordering = Compare(derived(), other);
    return ordering == Ordering::kGreater || ordering == Ordering::kEquivalent;
  }

  // Compound assignment.
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived& operator+=(
      const Derived& other) {
    derived() = derived() + other;
    return derived();
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived& operator-=(
      const Derived& other) {
    derived() = derived() - other;
    return derived();
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived& operator*=(
      const Derived& other) {
    derived() = derived() * other;
    return derived();
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Derived& operator/=(
      const Derived& other) {
    derived() = derived() / other;
    return derived();
  }

 private:
   // Mask the upper bits.
  static inline constexpr uint8_t Mask(uint8_t v) {
    return static_cast<uint8_t>(static_cast<uint8_t>(v) << (8 - N)) >> (8 - N);
  }

  static EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC std::pair<uint8_t, uint8_t>
  SignAndMagnitude(Derived x) {
    const uint8_t x_abs_bits =
        Eigen::numext::bit_cast<uint8_t>(Eigen::numext::abs(x));
    const uint8_t x_bits = Eigen::numext::bit_cast<uint8_t>(x);
    const uint8_t x_sign = x_bits ^ x_abs_bits;
    return {x_sign, x_abs_bits};
  }

  static EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC int8_t
  SignAndMagnitudeToTwosComplement(uint8_t sign, uint8_t magnitude) {
    return magnitude ^ (static_cast<int8_t>(sign << (8 - N)) < 0 ? -1 : 0);
  }

  enum Ordering : int8_t {
    kLess = -1,
    kEquivalent = 0,
    kGreater = 1,
    kUnordered = 2,
  };

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC friend Ordering Compare(
      const Derived& lhs, const Derived& rhs) {
    if (Eigen::numext::isnan(lhs) || Eigen::numext::isnan(rhs)) {
      return Ordering::kUnordered;
    }
    // no need to consider nan/inf
    auto [lhs_sign, lhs_mag] = SignAndMagnitude(lhs);
    auto [rhs_sign, rhs_mag] = SignAndMagnitude(rhs);
    if (lhs_mag == 0 && rhs_mag == 0) {
      return Ordering::kEquivalent;
    }
    int8_t lhs_twos_complement =
        SignAndMagnitudeToTwosComplement(lhs_sign, lhs_mag);
    int8_t rhs_twos_complement =
        SignAndMagnitudeToTwosComplement(rhs_sign, rhs_mag);
    if (lhs_twos_complement < rhs_twos_complement) {
      return Ordering::kLess;
    }
    if (lhs_twos_complement > rhs_twos_complement) {
      return Ordering::kGreater;
    }
    return Ordering::kEquivalent;
  }

  uint8_t rep_;
};

template <int N, typename T>
using RequiresIsDerivedFromFloatNBase =
    std::enable_if_t<std::is_base_of_v<floatn_base<N, T>, T>, int>;

class float4_e2m1 : public floatn_base<4, float4_e2m1> {
  // Exponent: 2, Mantissa: 1, bias: 1.
  // Extended range: no inf, no NaN.
 private:
  using Base = floatn_base<4, float4_e2m1>;
  friend class floatn_base<4, float4_e2m1>;
  using Base::Base;

 public:
  template <typename T, RequiresIsDerivedFromFloatNBase<4, T> = 0>
  explicit EIGEN_DEVICE_FUNC float4_e2m1(T f)
      : float4_e2m1(ConvertFrom(f)) {}
};

class float4_e1m2 : public floatn_base<4, float4_e1m2> {
  // Exponent: 1, Mantissa: 2, bias: 1.
  // Extended range: no inf, no NaN.
 private:
  using Base = floatn_base<4, float4_e1m2>;
  friend class floatn_base<4, float4_e1m2>;
  using Base::Base;

 public:
  template <typename T, RequiresIsDerivedFromFloatNBase<4, T> = 0>
  explicit EIGEN_DEVICE_FUNC float4_e1m2(T f)
      : float4_e1m2(ConvertFrom(f)) {}
};

class float6_e2m3 : public floatn_base<6, float6_e2m3> {
  // Exponent: 2, Mantissa: 3, bias: 1.
  // Extended range: no inf, no NaN.
 private:
  using Base = floatn_base<6, float6_e2m3>;
  friend class floatn_base<6, float6_e2m3>;
  using Base::Base;

 public:
  template <typename T, RequiresIsDerivedFromFloatNBase<6, T> = 0>
  explicit EIGEN_DEVICE_FUNC float6_e2m3(T f)
      : float6_e2m3(ConvertFrom(f)) {}
};

class float6_e3m2 : public floatn_base<6, float6_e3m2> {
  // Exponent: 2, Mantissa: 3, bias: 1.
  // Extended range: no inf, no NaN.
 private:
  using Base = floatn_base<6, float6_e3m2>;
  friend class floatn_base<6, float6_e3m2>;
  using Base::Base;

 public:
  template <typename T, RequiresIsDerivedFromFloatNBase<6, T> = 0>
  explicit EIGEN_DEVICE_FUNC float6_e3m2(T f)
      : float6_e3m2(ConvertFrom(f)) {}
};

class float8_e8m0 : public floatn_base<8, float8_e8m0> {
  // Exponent: 8, Mantissa: 0, bias: 127.
  // Extended range: no inf/Zero, with NaN as 0b1111'1111.
 private:
  using Base = floatn_base<8, float8_e8m0>;
  friend class floatn_base<8, float8_e8m0>;
  using Base::Base;

 public:
  template <typename T, RequiresIsDerivedFromFloatNBase<8, T> = 0>
  explicit EIGEN_DEVICE_FUNC float8_e8m0(T f)
      : float8_e8m0(ConvertFrom(f)) {}
  explicit EIGEN_DEVICE_FUNC operator bool() const {
    return false;
  }
  constexpr float8_e8m0 operator-() const {
    return Base::derived();
  }

  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC float8_e8m0
  operator-(const float8_e8m0& other) const {
    return float8_e8m0{float{Base::derived()} - float{other}};
  }
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
  // We only support digits in {2,3,4}. This table would grow if we wanted to
  // handle more values.
  constexpr double kLog10OfOnePredecessor[] = {
      // log10(1 - 2**-2)
      -0.12493873660829993,
      // log10(1 - 2**-3)
      -0.057991946977686754,
      // log10(1 - 2**-4)
      -0.028028723600243537,
  };
  return static_cast<int>(ConstexprFloor(kLog10OfOnePredecessor[digits - 2] +
                                         max_exponent * kLog10Of2));
}

struct numeric_limits_floatn_base {
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const bool is_specialized = true;
  static inline constexpr const bool is_signed = true;
  static inline constexpr const bool is_integer = false;
  static inline constexpr const bool is_exact = false;
  static inline constexpr const bool has_quiet_NaN = false;
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
  static inline constexpr const bool is_iec559 = false;  // IEEE-754
  static inline constexpr const bool has_infinity = false;
  static inline constexpr const bool has_signaling_NaN = false;
  // NOLINTEND
};

struct numeric_limits_float4_e2m1 : public numeric_limits_floatn_base {
 private:
  static inline constexpr const int kExponentBias = 1;
  static inline constexpr const int kMantissaBits = 1;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = kMantissaBits + 1;
  static inline constexpr const int digits10 = Digits10FromDigits(digits);
  static inline constexpr const int max_digits10 =
      MaxDigits10FromDigits(digits);
  static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
  static inline constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static inline constexpr const int max_exponent =
      (0b11 - kExponentBias) + 1;  // Extended format.
  static inline constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  // NOLINTEND

  // 1.0 * 2^(0b01 - 1) = 1.0 * 2^0 = 1.0
  static constexpr float4_e2m1 min() {
    return float4_e2m1::FromRep(0b0'01 << kMantissaBits);
  }
  // -(1 + 0b1 * 2^-1) * 2**(0b11 - 1) = -1.5 * 2^2 = -6
  static constexpr float4_e2m1 lowest() {
    return float4_e2m1::FromRep(0b1'11'1);
  }
  // (1 + 0b1 * 2^-1) * 2**(0b11 - 1) = 1.5 * 2^2 = 6
  static constexpr float4_e2m1 max() {
    return float4_e2m1::FromRep(0b0'11'1);
  }
  // 0.5, difference between 1.0 and the next representable value.
  static constexpr float4_e2m1 epsilon() {
    return float4_e2m1::FromRep(0b0'00'1);
  }
  // 0.5
  static constexpr float4_e2m1 round_error() {
    return float4_e2m1::FromRep(0b0'00'1);
  }
  static constexpr float4_e2m1 infinity() {
    return float4_e2m1::FromRep(0b0'11'1);
  }
  // NaN.
  static constexpr float4_e2m1 quiet_NaN() {
    return float4_e2m1::FromRep(0b0'00'0);
  }
  static constexpr float4_e2m1 signaling_NaN() {
    return float4_e2m1::FromRep(0b0'00'0);
  }
  // 0.5
  static constexpr float4_e2m1 denorm_min() {
    return float4_e2m1::FromRep(0b0'00'1);
  }
};

struct numeric_limits_float4_e1m2 : public numeric_limits_floatn_base {
 private:
  static inline constexpr const int kExponentBias = 1;
  static inline constexpr const int kMantissaBits = 2;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = kMantissaBits + 1;
  static inline constexpr const int digits10 = Digits10FromDigits(digits);
  static inline constexpr const int max_digits10 =
      MaxDigits10FromDigits(digits);
  static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
  static inline constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static inline constexpr const int max_exponent =
      (0b1 - kExponentBias) + 1;  // Extended format.
  static inline constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  // NOLINTEND

  // 1.0 * 2^(0b1 - 1) = 1.0 * 2^0 = 1.0
  static constexpr float4_e1m2 min() {
    return float4_e1m2::FromRep(0b0'1 << kMantissaBits);
  }
  // -(1 + 0b11 * 2^-2) * 2**(0b1 - 1) = -1.75 * 2^0 = -1.75
  static constexpr float4_e1m2 lowest() {
    return float4_e1m2::FromRep(0b1'1'11);
  }
  // (1 + 0b11 * 2^-2) * 2**(0b1 - 1) = 1.75 * 2^0 = 1.75
  static constexpr float4_e1m2 max() {
    return float4_e1m2::FromRep(0b0'1'11);
  }
  // 0.25, difference between 1.0 and the next representable value.
  static constexpr float4_e1m2 epsilon() {
    return float4_e1m2::FromRep(0b0'0'01);
  }
  // 0.5
  static constexpr float4_e1m2 round_error() {
    return float4_e1m2::FromRep(0b0'0'10);
  }
  static constexpr float4_e1m2 infinity() {
    return float4_e1m2::FromRep(0b0'1'11);
  }
  // NaN.
  static constexpr float4_e1m2 quiet_NaN() {
    return float4_e1m2::FromRep(0b0'0'00);
  }
  static constexpr float4_e1m2 signaling_NaN() {
    return float4_e1m2::FromRep(0b0'0'00);
  }
  // 0.25
  static constexpr float4_e1m2 denorm_min() {
    return float4_e1m2::FromRep(0b0'0'01);
  }
};

struct numeric_limits_float6_e2m3 : public numeric_limits_floatn_base {
 private:
  static inline constexpr const int kExponentBias = 1;
  static inline constexpr const int kMantissaBits = 3;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = kMantissaBits + 1;
  static inline constexpr const int digits10 = Digits10FromDigits(digits);
  static inline constexpr const int max_digits10 =
      MaxDigits10FromDigits(digits);
  static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
  static inline constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static inline constexpr const int max_exponent =
      (0b11 - kExponentBias) + 1;  // Extended format.
  static inline constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  // NOLINTEND

  // 1.0 * 2^(0b01 - 1) = 1.0 * 2^0 = 1.0
  static constexpr float6_e2m3 min() {
    return float6_e2m3::FromRep(0b0'01 << kMantissaBits);
  }
  // -(1 + 0b111 * 2^-3) * 2**(0b11 - 1) = -1.875 * 2^2 = -7.5
  static constexpr float6_e2m3 lowest() {
    return float6_e2m3::FromRep(0b1'11'111);
  }
  // (1 + 0b111 * 2^-3) * 2**(0b11 - 1) = 1.875 * 2^2 = 7.5
  static constexpr float6_e2m3 max() {
    return float6_e2m3::FromRep(0b0'11'111);
  }
  // 0.125, difference between 1.0 and the next representable value.
  static constexpr float6_e2m3 epsilon() {
    return float6_e2m3::FromRep(0b0'00'001);
  }
  // 0.5
  static constexpr float6_e2m3 round_error() {
    return float6_e2m3::FromRep(0b0'00'100);
  }
  static constexpr float6_e2m3 infinity() {
    return float6_e2m3::FromRep(0b0'11'111);
  }
  // NaN.
  static constexpr float6_e2m3 quiet_NaN() {
    return float6_e2m3::FromRep(0b0'00'000);
  }
  static constexpr float6_e2m3 signaling_NaN() {
    return float6_e2m3::FromRep(0b0'00'000);
  }
  // 0.125
  static constexpr float6_e2m3 denorm_min() {
    return float6_e2m3::FromRep(0b0'00'001);
  }
};

struct numeric_limits_float6_e3m2 : public numeric_limits_floatn_base {
 private:
  static inline constexpr const int kExponentBias = 3;
  static inline constexpr const int kMantissaBits = 2;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = kMantissaBits + 1;
  static inline constexpr const int digits10 = Digits10FromDigits(digits);
  static inline constexpr const int max_digits10 =
      MaxDigits10FromDigits(digits);
  static inline constexpr const int min_exponent = (1 - kExponentBias) + 1;
  static inline constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static inline constexpr const int max_exponent =
      (0b111 - kExponentBias) + 1;  // Extended format.
  static inline constexpr const int max_exponent10 =
      MaxExponent10FromMaxExponentAndDigits(max_exponent, digits);
  // NOLINTEND

  // 1.0 * 2^(0b001 - 3) = 1.0 * 2^-2 = 0.25
  static constexpr float6_e3m2 min() {
    return float6_e3m2::FromRep(0b0'001 << kMantissaBits);
  }
  // -(1 + 0b11 * 2^-2) * 2**(0b111 - 3) = -1.75 * 2^4 = -28
  static constexpr float6_e3m2 lowest() {
    return float6_e3m2::FromRep(0b1'111'11);
  }
  // (1 + 0b11 * 2^-2) * 2**(0b111 - 3) = 1.75 * 2^4 = 28
  static constexpr float6_e3m2 max() {
    return float6_e3m2::FromRep(0b0'111'11);
  }
  // 0.25, difference between 1.0 and the next representable value.
  static constexpr float6_e3m2 epsilon() {
    return float6_e3m2::FromRep(0b0'001'00);
  }
  // 0.5
  static constexpr float6_e3m2 round_error() {
    return float6_e3m2::FromRep(0b0'010'00);
  }
  static constexpr float6_e3m2 infinity() {
    return float6_e3m2::FromRep(0b0'111'11);
  }
  // NaN.
  static constexpr float6_e3m2 quiet_NaN() {
    return float6_e3m2::FromRep(0b0'000'00);
  }
  static constexpr float6_e3m2 signaling_NaN() {
    return float6_e3m2::FromRep(0b0'000'00);
  }
  // 0.0625
  static constexpr float6_e3m2 denorm_min() {
    return float6_e3m2::FromRep(0b0'000'01);
  }
};

struct numeric_limits_float8_e8m0 : public numeric_limits_floatn_base {
  static inline constexpr const bool is_signed = false;
  static inline constexpr const bool has_quiet_NaN = true;
  static inline constexpr const std::float_denorm_style has_denorm =
      std::denorm_absent;
  static inline constexpr const bool has_signaling_NaN = true;

 private:
  static inline constexpr const int kExponentBias = 127;
  static inline constexpr const int kMantissaBits = 0;

 public:
  // NOLINTBEGIN: these names must match std::numeric_limits.
  static inline constexpr const int digits = kMantissaBits + 1;
  static inline constexpr const int digits10 = 0;
  static inline constexpr const int max_digits10 = 0;
  static inline constexpr const int min_exponent = 1 - kExponentBias;
  static inline constexpr const int min_exponent10 =
      MinExponent10FromMinExponent(min_exponent);
  static inline constexpr const int max_exponent = 0b1111'1111 - kExponentBias;
  static inline constexpr const int max_exponent10 =
      static_cast<int>(ConstexprFloor(max_exponent * kLog10Of2));
  // NOLINTEND

  // 2**(0b0000'0000 - 127) = 2^-127
  static constexpr float8_e8m0 min() {
    return float8_e8m0::FromRep(0b0000'0000);
  }
  // 2**(0b0000'0000 - 127) = 2^-127
  static constexpr float8_e8m0 lowest() {
    return float8_e8m0::FromRep(0b0000'0000);
  }
  // 2**(0b1111'1110 - 127) = 2^127
  static constexpr float8_e8m0 max() {
    return float8_e8m0::FromRep(0b1111'1110);
  }
  // 1, difference between 1.0 and the next representable value.
  // 1.0 = 2**(0) = 2**(127-127) = 2**(0b0111'1111 - 127)
  // 2**(0b1000'0000 - 127) = 2
  // difference = 2 - 1 = 1 = 2**(0b0111'1111 - 127)
  static constexpr float8_e8m0 epsilon() {
    return float8_e8m0::FromRep(0b0111'1111);
  }
  // 1
  static constexpr float8_e8m0 round_error() {
    return float8_e8m0::FromRep(0b0111'1111);
  }
  // infinity to NaN
  static constexpr float8_e8m0 infinity() {
    return float8_e8m0::FromRep(0b1111'1111);
  }
  // NaN.
  static constexpr float8_e8m0 quiet_NaN() {
    return float8_e8m0::FromRep(0b1111'1111);
  }
  static constexpr float8_e8m0 signaling_NaN() {
    return float8_e8m0::FromRep(0b1111'1111);
  }
  // no use, since has_denormal is absent
  static constexpr float8_e8m0 denorm_min() {
    return float8_e8m0::FromRep(0b0000'0000);
  }
};

}  // namespace floatn_internal
}  // namespace en_dtypes

namespace std {
// Standard-library overrides.  Note that these are picked up by Eigen as well.
template <>
struct numeric_limits<en_dtypes::floatn_internal::float4_e2m1>
    : public en_dtypes::floatn_internal::numeric_limits_float4_e2m1 {};

template <>
struct numeric_limits<en_dtypes::floatn_internal::float4_e1m2>
    : public en_dtypes::floatn_internal::numeric_limits_float4_e1m2 {};

template <>
struct numeric_limits<en_dtypes::floatn_internal::float6_e2m3>
    : public en_dtypes::floatn_internal::numeric_limits_float6_e2m3 {};

template <>
struct numeric_limits<en_dtypes::floatn_internal::float6_e3m2>
    : public en_dtypes::floatn_internal::numeric_limits_float6_e3m2 {};

template <>
struct numeric_limits<en_dtypes::floatn_internal::float8_e8m0>
    : public en_dtypes::floatn_internal::numeric_limits_float8_e8m0 {};
}  // namespace std

namespace en_dtypes {
namespace floatn_internal {

// Free-functions for use with ADL and in Eigen.
constexpr inline float4_e2m1 abs(const float4_e2m1& a) {
  return float4_e2m1::FromRep(a.rep() & 0b0000'0'11'1);
}

constexpr inline float4_e1m2 abs(const float4_e1m2& a) {
  return float4_e1m2::FromRep(a.rep() & 0b0000'0'1'11);
}

constexpr inline float6_e2m3 abs(const float6_e2m3& a) {
  return float6_e2m3::FromRep(a.rep() & 0b00'0'11'111);
}

constexpr inline float6_e3m2 abs(const float6_e3m2& a) {
  return float6_e3m2::FromRep(a.rep() & 0b00'0'111'11);
}

constexpr inline float8_e8m0 abs(const float8_e8m0& a) {
  return float8_e8m0::FromRep(a.rep());
}

template <int N, typename FloatN>
constexpr inline bool(isnan)(const floatn_base<N, FloatN>& a) {
  // No NaN representation except float8_e8m0.
  return false;
}

template <>
constexpr inline bool(isnan)(const floatn_base<8, float8_e8m0>& a) {
  return a.rep() == std::numeric_limits<float8_e8m0>::quiet_NaN().rep();
}

template <int N, typename FloatN>
constexpr inline bool(isinf)(const floatn_base<N, FloatN>& a) {
  // No inf representation.
  return false;
}

template <int N, typename FloatN>
constexpr inline bool(isfinite)(const floatn_base<N, FloatN>& a) {
  return true;
}

template <>
constexpr inline bool(isfinite)(const floatn_base<8, float8_e8m0>& a) {
  return !isnan(a.derived()) && !isinf(a.derived());
}

template <int N, typename FloatN>
std::ostream& operator<<(std::ostream& os, const floatn_base<N, FloatN>& f) {
  os << static_cast<float>(f.derived());
  return os;
}

//==============================================================================
// Inline conversion routines between float4 and other types.
//==============================================================================

// Helper for getting a bit representation provided a byte size.
template <int kNumBytes>
using GetUnsignedInteger =
    typename Eigen::numext::get_integer_by_size<kNumBytes>::unsigned_type;

// Converts between two floating-point types.
template <typename From, typename To, bool kSaturate, bool kTruncate,
          typename EnableIf = void>
struct ConvertImpl;

// Convert to same type.  We need explicit specializations for all combinations
// of template parameters to avoid ambiguities.
template <typename Scalar>
struct IdentityConversion {
  static EIGEN_DEVICE_FUNC inline Scalar run(Scalar from) { return from; }
};

template <typename Scalar, bool kSaturate, bool kTruncate>
struct ConvertImpl<Scalar, Scalar, /*kSaturate=*/kSaturate,
                   /*kTruncate=*/kTruncate>
    : public IdentityConversion<Scalar> {};

template <typename Float>
struct TraitsBase {
  using BitsType = GetUnsignedInteger<sizeof(Float)>;
  static constexpr int kBits = sizeof(Float) * CHAR_BIT;
  static constexpr int kMantissaBits = Eigen::NumTraits<Float>::digits() - 1;
  static constexpr int kExponentBits = kBits - kMantissaBits - 1;
  static constexpr BitsType kExponentMask = ((BitsType{1} << kExponentBits) - 1)
                                            << kMantissaBits;
  static constexpr BitsType kSignMask = BitsType{1} << (kBits - 1);
  static constexpr BitsType kMantissaMask = (BitsType{1} << kMantissaBits) - 1;
  static constexpr int kExponentBias = (1 << (kExponentBits - 1)) - 1;
};

template <typename Float>
struct Traits : public TraitsBase<Float> {};

template <>
struct Traits<float4_e2m1> : public TraitsBase<float4_e2m1> {
  using Base = TraitsBase<float4_e2m1>;
  static constexpr int kBits = 4;
  static constexpr int kExponentBits = 2;
  static constexpr Base::BitsType kExponentMask = ((Base::BitsType{1} << kExponentBits) - 1)
                                                  << Base::kMantissaBits;
  static constexpr Base::BitsType kSignMask = Base::BitsType{1} << (kBits - 1);
  static constexpr Base::BitsType kMantissaMask = (Base::BitsType{1} << Base::kMantissaBits) - 1;
  static constexpr int kExponentBias = 1;
};

template <>
struct Traits<float4_e1m2> : public TraitsBase<float4_e1m2> {
  using Base = TraitsBase<float4_e1m2>;
  static constexpr int kBits = 4;
  static constexpr int kExponentBits = 1;
  static constexpr Base::BitsType kExponentMask = ((Base::BitsType{1} << kExponentBits) - 1)
                                                  << Base::kMantissaBits;
  static constexpr Base::BitsType kSignMask = Base::BitsType{1} << (kBits - 1);
  static constexpr Base::BitsType kMantissaMask = (Base::BitsType{1} << Base::kMantissaBits) - 1;
  static constexpr int kExponentBias = 1;
};

template <>
struct Traits<float6_e2m3> : public TraitsBase<float6_e2m3> {
  using Base = TraitsBase<float6_e2m3>;
  static constexpr int kBits = 6;
  static constexpr int kExponentBits = 2;
  static constexpr Base::BitsType kExponentMask = ((Base::BitsType{1} << kExponentBits) - 1)
                                                  << Base::kMantissaBits;
  static constexpr Base::BitsType kSignMask = Base::BitsType{1} << (kBits - 1);
  static constexpr Base::BitsType kMantissaMask = (Base::BitsType{1} << Base::kMantissaBits) - 1;
  static constexpr int kExponentBias = 1;
};

template <>
struct Traits<float6_e3m2> : public TraitsBase<float6_e3m2> {
  using Base = TraitsBase<float6_e3m2>;
  static constexpr int kBits = 6;
  static constexpr int kExponentBits = 3;
  static constexpr Base::BitsType kExponentMask = ((Base::BitsType{1} << kExponentBits) - 1)
                                                  << Base::kMantissaBits;
  static constexpr Base::BitsType kSignMask = Base::BitsType{1} << (kBits - 1);
  static constexpr Base::BitsType kMantissaMask = (Base::BitsType{1} << Base::kMantissaBits) - 1;
  static constexpr int kExponentBias = 3;
};

template <>
struct Traits<float8_e8m0> : public TraitsBase<float8_e8m0> {
  using Base = TraitsBase<float8_e8m0>;
  static constexpr int kBits = 8;
  static constexpr int kExponentBits = 8;
  static constexpr Base::BitsType kExponentMask = (Base::BitsType{1} << kExponentBits) - 1;
  static constexpr Base::BitsType kSignMask = Base::BitsType{0};
  static constexpr Base::BitsType kMantissaMask = Base::BitsType{0};
  static constexpr int kExponentBias = 127;
};

template <typename Bits>
constexpr inline Bits RoundBitsToNearestEven(Bits bits, int roundoff) {
  // TODO: how to understand this?: refer to BFP16
  // Round to nearest even by adding a bias term.
  // Consider a bit pattern
  //   FFF...FLRTT...T,
  // where bits RTT...T need to be rounded-off.  We add a bias term to the
  // bit pattern s.t. a carry is introduced to round up only if
  // - L is 1, R is 1, OR
  // - L is 0, R is 1, any T is one.
  // We do this by adding L to a bit pattern consisting of all T = 1.
  Bits bias = roundoff == 0
                  ? 0
                  : ((bits >> roundoff) & 1) + (Bits{1} << (roundoff - 1)) - 1;
  return bits + bias;
}

#if (defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L)
using std::countl_zero;
#else
static constexpr inline int countl_zero(uint64_t x) {
  int zeroes = 60;
  if (x >> 32) {
    zeroes -= 32;
    x >>= 32;
  }
  if (x >> 16) {
    zeroes -= 16;
    x >>= 16;
  }
  if (x >> 8) {
    zeroes -= 8;
    x >>= 8;
  }
  if (x >> 4) {
    zeroes -= 4;
    x >>= 4;
  }
  return "\4\3\2\2\1\1\1\1\0\0\0\0\0\0\0"[x] + zeroes;
}
static constexpr inline int countl_zero(uint32_t x) {
  int zeroes = 28;
  if (x >> 16) {
    zeroes -= 16;
    x >>= 16;
  }
  if (x >> 8) {
    zeroes -= 8;
    x >>= 8;
  }
  if (x >> 4) {
    zeroes -= 4;
    x >>= 4;
  }
  return "\4\3\2\2\1\1\1\1\0\0\0\0\0\0\0"[x] + zeroes;
}
static constexpr inline int countl_zero(uint16_t x) {
  int zeroes = 12;
  if (x >> 8) {
    zeroes -= 8;
    x >>= 8;
  }
  if (x >> 4) {
    zeroes -= 4;
    x >>= 4;
  }
  return "\4\3\2\2\1\1\1\1\0\0\0\0\0\0\0"[x] + zeroes;
}
static constexpr inline int countl_zero(uint8_t x) {
  int zeroes = 4;
  if (x >> 4) {
    zeroes -= 4;
    x >>= 4;
  }
  return "\4\3\2\2\1\1\1\1\0\0\0\0\0\0\0"[x] + zeroes;
}
#endif

template <typename From, typename To, bool kSaturate, bool kTruncate>
struct ConvertImpl<From, To, kSaturate, kTruncate,
                   std::enable_if_t<!std::is_same_v<From, To>>> {
  using FromTraits = Traits<From>;
  using FromBits = typename FromTraits::BitsType;
  static constexpr int kFromBits = FromTraits::kBits;
  static constexpr int kFromMantissaBits = FromTraits::kMantissaBits;
  static constexpr int kFromExponentBits = FromTraits::kExponentBits;
  static constexpr int kFromExponentBias = FromTraits::kExponentBias;
  static constexpr FromBits kFromExponentMask = FromTraits::kExponentMask;

  using ToTraits = Traits<To>;
  using ToBits = typename ToTraits::BitsType;
  static constexpr int kToBits = ToTraits::kBits;
  static constexpr int kToMantissaBits = ToTraits::kMantissaBits;
  static constexpr int kToExponentBits = ToTraits::kExponentBits;
  static constexpr int kToExponentBias = ToTraits::kExponentBias;
  static constexpr ToBits kToExponentMask = ToTraits::kExponentMask;

  // `WideBits` is wide enough to accommodate the largest exponent and mantissa
  // in either `From` or `To`.
  static constexpr int kWideBits =
      (std::max(kToMantissaBits, kFromMantissaBits)) +  // Max significand.
      (std::max(kToExponentBits, kFromExponentBits));   // Max exponent.
  static constexpr int kWideBytes = (kWideBits + (CHAR_BIT - 1)) / CHAR_BIT;
  using WideBits = GetUnsignedInteger<kWideBytes>;
  static constexpr int kExponentOffset = kToExponentBias - kFromExponentBias;
  static constexpr int kDigitShift = kToMantissaBits - kFromMantissaBits;

  static EIGEN_DEVICE_FUNC inline To run(From from) {
    // Shift bits to destination type, without sign bit.
    const bool from_sign_bit =
        Eigen::numext::bit_cast<FromBits>(from) >> (kFromBits - 1);
    const FromBits from_bits =
        Eigen::numext::bit_cast<FromBits>(Eigen::numext::abs(from));

    // Special values, preserving sign.
    if (Eigen::numext::isinf(from)) {
      return from_sign_bit ? -Eigen::NumTraits<To>::infinity()
                           : Eigen::NumTraits<To>::infinity();
    }
    if (Eigen::numext::isnan(from)) {
      return from_sign_bit ? -Eigen::NumTraits<To>::quiet_NaN()
                           : Eigen::NumTraits<To>::quiet_NaN();
    }
    if (from_bits == 0) {
      return from_sign_bit ? -To{} : To{};
    }

    const int biased_from_exponent = from_bits >> kFromMantissaBits;

    // `To` supports more exponents near zero which means that some subnormal
    // values in `From` may become normal.
    if constexpr (std::numeric_limits<To>::min_exponent <
                  std::numeric_limits<From>::min_exponent) {
      if (biased_from_exponent == 0) {
        // Subnormals.
        WideBits bits = from_bits;

        // Determine exponent in target type.
        const int normalization_factor =
            countl_zero(from_bits) - (kFromBits - kFromMantissaBits) + 1;
        const int biased_exponent = kExponentOffset - normalization_factor + 1;
        if (biased_exponent <= 0) {
          // Result is subnormal.  Adjust the subnormal bits to account for
          // the difference in exponent bias.
          if constexpr (kExponentOffset < sizeof(WideBits) * CHAR_BIT) {
            bits <<= kExponentOffset;
          }
        } else {
          // Result is normal. Shift the mantissa to account for the number of
          // leading zero digits, and clear the hidden bit.
          bits <<= normalization_factor;
          bits &= ~(WideBits{1} << kFromMantissaBits);
          // Insert the exponent bits.
          bits |= static_cast<WideBits>(biased_exponent) << kFromMantissaBits;
        }

        // Truncate/round mantissa if necessary.
        if constexpr (kDigitShift > 0) {
          bits <<= kDigitShift;
        } else {
          if constexpr (!kTruncate) {
            bits = RoundBitsToNearestEven(bits, -kDigitShift);
          }
          bits >>= -kDigitShift;
        }
        To to = Eigen::numext::bit_cast<To>(static_cast<ToBits>(bits));
        return from_sign_bit ? -to : to;
      }
    }
    // `To` supports fewer exponents near zero which means that some values in
    // `From` may become subnormal.
    if constexpr (std::numeric_limits<To>::min_exponent >
                  std::numeric_limits<From>::min_exponent) {
      const int unbiased_exponent = biased_from_exponent - kFromExponentBias;
      const int biased_to_exponent = unbiased_exponent + kToExponentBias;
      // Subnormals and zero.
      if (biased_to_exponent <= 0) {
        // Round and shift mantissa down.
        FromBits from_has_leading_one = (biased_from_exponent > 0 ? 1 : 0);
        int exponent_shift =
            -kDigitShift - biased_to_exponent + from_has_leading_one;
        // Insert the implicit leading 1 bit on the mantissa for normalized
        // inputs.
        FromBits rounded_from_bits =
            (from_bits & FromTraits::kMantissaMask) |
            (from_has_leading_one << kFromMantissaBits);
        ToBits bits = 0;
        if (exponent_shift > 0) {
          // To avoid UB, limit rounding and shifting to the full mantissa plus
          // leading 1.
          if (exponent_shift <= kFromMantissaBits + 1) {
            if constexpr (!kTruncate) {
              // NOTE: we need to round again from the original from_bits,
              // otherwise the lower precision bits may already be lost.  There
              // is an edge-case where rounding to a normalized value would
              // normally round down, but for a subnormal, we need to round up.
              rounded_from_bits =
                  RoundBitsToNearestEven(rounded_from_bits, exponent_shift);
            }
            bits = rounded_from_bits >> exponent_shift;
          }
        } else {
          bits = rounded_from_bits << -exponent_shift;
        }
        // Insert sign and return.
        To to = Eigen::numext::bit_cast<To>(bits);
        return from_sign_bit ? -to : to;
      }
    }

    // Round the mantissa if it is shrinking.
    WideBits rounded_from_bits = from_bits;
    if constexpr (kDigitShift < 0) {
      if constexpr (!kTruncate) {
        rounded_from_bits = RoundBitsToNearestEven(from_bits, -kDigitShift);
      }
      // Zero-out tail bits.
      rounded_from_bits &= ~((WideBits{1} << (-kDigitShift)) - 1);
    }

    // Re-bias the exponent.
    rounded_from_bits += static_cast<WideBits>(kExponentOffset)
                         << kFromMantissaBits;

    ToBits bits;
    // Check for overflows by aligning the significands. We always align the
    // narrower significand to the wider significand.
    const WideBits kToHighestRep =
        Eigen::numext::bit_cast<ToBits>(Eigen::NumTraits<To>::highest());
    WideBits aligned_highest{kToHighestRep};
    if constexpr (kDigitShift < 0) {
      aligned_highest <<= -kDigitShift;
      // Shift down, all dropped bits should already be zero.
      bits = static_cast<ToBits>(rounded_from_bits >> -kDigitShift);
    } else if constexpr (kDigitShift >= 0) {
      // Shift up, inserting zeros in the newly created digits.
      rounded_from_bits <<= kDigitShift;
      bits = ToBits{rounded_from_bits};
    }

    To to = Eigen::numext::bit_cast<To>(bits);
    // `From` supports larger values than `To`, we may overflow.
    if constexpr (std::make_pair(std::numeric_limits<To>::max_exponent,
                                 std::numeric_limits<To>::digits) <
                  std::make_pair(std::numeric_limits<From>::max_exponent,
                                 std::numeric_limits<From>::digits)) {
      if (rounded_from_bits > aligned_highest) {
        // Overflowed values map to highest or infinity depending on kSaturate.
        to = kSaturate ? Eigen::NumTraits<To>::highest()
                       : Eigen::NumTraits<To>::infinity();
      }
    }
    // Insert sign bit.
    return from_sign_bit ? -to : to;
  }
};

template <int N, typename Derived>
template <bool kSaturate, bool kTruncate, typename From>
EIGEN_DEVICE_FUNC Derived floatn_base<N, Derived>::ConvertFrom(const From from) {
  // We are rounding long double -> float -> float4/float6. This can induce
  // double-rounding which may alter the results. We can correct for this using
  // a trick explained in: Boldo, Sylvie, and Guillaume Melquiond. "When double
  // rounding is odd." 17th IMACS World Congress. 2005.
  if constexpr (std::is_floating_point_v<From> &&
                sizeof(From) > sizeof(double)) {
    // binary64, float80, binary128, etc. end up here.
    static_assert(std::numeric_limits<From>::digits >=
                  std::numeric_limits<float>::digits + 2);
    static_assert(std::numeric_limits<float>::min_exponent >=
                  std::numeric_limits<From>::min_exponent + 2);
    static_assert(std::numeric_limits<float>::is_iec559);
    static_assert(std::numeric_limits<float>::radix == 2);
    const bool is_negative = std::signbit(from);
    const From abs_wide = std::fabs(from);
    float abs_narrow = static_cast<float>(abs_wide);
    const From abs_narrow_as_wide = static_cast<From>(abs_narrow);

    uint32_t narrow_bits = Eigen::numext::bit_cast<uint32_t>(abs_narrow);
    // We can keep the narrow value as-is if narrowing was exact (no rounding
    // error), the wide value was NaN (the narrow value is also NaN and should
    // be preserved) or if we rounded to the odd value.
    const bool keep_narrow = (abs_wide == abs_narrow_as_wide) ||
                             std::isnan(abs_narrow) || (narrow_bits & 1);
    // We morally performed a round-down if `abs_narrow` is smaller than
    // `abs_wide`.
    const bool narrow_is_rd = abs_wide > abs_narrow_as_wide;
    // If the narrow value is odd or exact, pick it.
    // Otherwise, narrow is even and corresponds to either the rounded-up or
    // rounded-down value. If narrow is the rounded-down value, we want the
    // rounded-up value as it will be odd.
    narrow_bits += keep_narrow ? 0 : narrow_is_rd ? 1 : -1;
    abs_narrow = Eigen::numext::bit_cast<float>(narrow_bits);
    return ConvertImpl<float, Derived, kSaturate, kTruncate>::run(
        is_negative ? -abs_narrow : abs_narrow);
  } else {
    return ConvertImpl<From, Derived, kSaturate, kTruncate>::run(from);
  }
}

template <typename To, bool kSaturate, bool kTruncate>
struct ConvertImpl</*From=*/float8_e8m0, To, kSaturate, kTruncate,
                   std::enable_if_t<!std::is_same_v<float8_e8m0, To>>> {
  using FromTraits = Traits<float8_e8m0>;
  static constexpr int kFromExponentBias = FromTraits::kExponentBias;

  using ToTraits = Traits<To>;
  using ToBits = typename ToTraits::BitsType;
  static constexpr int kToMantissaBits = ToTraits::kMantissaBits;
  static constexpr int kToExponentBias = ToTraits::kExponentBias;

  static EIGEN_DEVICE_FUNC To run(float8_e8m0 from) {
    if (Eigen::numext::isnan(from)) {
      return Eigen::NumTraits<To>::quiet_NaN();
    }

    /*
      When From=float8_e8m0, we convert float8_e8m0 -> float -> To
    */
    const uint8_t from_bits =
        Eigen::numext::bit_cast<uint8_t>(Eigen::numext::abs(from));
    float float_v = Eigen::numext::bit_cast<float>(
        static_cast<uint32_t>(static_cast<uint32_t>(from_bits) << 23));
    if (from_bits == 0) {
      float_v = Eigen::numext::bit_cast<float>(
        static_cast<uint32_t>(static_cast<uint32_t>(1) << 22));
    }

    return std::is_same_v<To, float> ? float_v : \
           ConvertImpl<float, To, kSaturate, kTruncate>::run(float_v);
  }
};

// float8_e8m0 has no sign bit. only do Truncate only.
template <typename From, bool kSaturate, bool kTruncate>
struct ConvertImpl<From, /*To=*/float8_e8m0, kSaturate, kTruncate,
                   std::enable_if_t<!std::is_same_v<From, float8_e8m0>>> {
  using FromTraits = Traits<From>;
  using FromBits = typename FromTraits::BitsType;
  static constexpr int kFromBits = FromTraits::kBits;
  static constexpr int kFromMantissaBits = FromTraits::kMantissaBits;
  static constexpr int kFromExponentBias = FromTraits::kExponentBias;

  using ToTraits = Traits<float8_e8m0>;
  static constexpr int kToExponentBias = ToTraits::kExponentBias;

  static EIGEN_DEVICE_FUNC float8_e8m0 run(From from) {
    // Special values. only NaN in float8_e8m0.
    if (Eigen::numext::isinf(from) || Eigen::numext::isnan(from)) {
      return Eigen::NumTraits<float8_e8m0>::quiet_NaN();
    }

    const bool from_sign_bit =
        Eigen::numext::bit_cast<FromBits>(from) >> (kFromBits - 1);
    if (from_sign_bit > 0) {
      // negative. return NaN.
      return Eigen::NumTraits<float8_e8m0>::quiet_NaN();
    }

    const FromBits from_bits =
        Eigen::numext::bit_cast<FromBits>(Eigen::numext::abs(from));
    if (from_bits == 0) {
      // float8_e8m0 has no zero.
      return Eigen::NumTraits<float8_e8m0>::quiet_NaN();
    }

    /*
      When To=float8_e8m0, we convert From -> float -> float8_e8m0
    */
    if (std::is_same_v<From, float>) {
      // 2**-127
      const uint32_t float8_min_bits = static_cast<uint32_t>(static_cast<uint32_t>(1) << 22);
      if (from_bits == float8_min_bits) {
        return float8_e8m0{};  // 0b0000'0000
      } else if (from_bits < float8_min_bits) {
        // Overflowed values map to lowest or NaN depending on kSaturate.
        return kSaturate ? Eigen::NumTraits<float8_e8m0>::lowest()
                         : Eigen::NumTraits<float8_e8m0>::quiet_NaN();
      } else {
        // Truncate exponent directely
        const int biased_from_exponent = from_bits >> kFromMantissaBits;
        const int from_exponent = biased_from_exponent - kFromExponentBias;
        return Eigen::numext::bit_cast<float8_e8m0>(
                static_cast<uint8_t>(static_cast<uint8_t>(from_exponent + kToExponentBias))
          );
      }
    } else {
      const float float_v = ConvertImpl<From, float, kSaturate, kTruncate>::run(from);
      return ConvertImpl<float, float8_e8m0, kSaturate, kTruncate>::run(float_v);
    }
  }
};

template <int N, typename Derived>
template <typename To, bool kSaturate, bool kTruncate>
EIGEN_DEVICE_FUNC To floatn_base<N, Derived>::ConvertTo(Derived from) {
  return ConvertImpl</*From=*/Derived, To, kSaturate, kTruncate>::run(from);
}

}  // namespace floatn_internal

// Exported types.
using float4_e2m1 = floatn_internal::float4_e2m1;
using float4_e1m2 = floatn_internal::float4_e1m2;
using float6_e2m3 = floatn_internal::float6_e2m3;
using float6_e3m2 = floatn_internal::float6_e3m2;
using float8_e8m0 = floatn_internal::float8_e8m0;

}  // namespace en_dtypes

// Eigen-specific overrides.
namespace Eigen {
namespace numext {

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC en_dtypes::float4_e2m1
bit_cast<en_dtypes::float4_e2m1, uint8_t>(const uint8_t& src) {
  return en_dtypes::float4_e2m1::FromRep(src);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint8_t
bit_cast<uint8_t, en_dtypes::float4_e2m1>(
    const en_dtypes::float4_e2m1& src) {
  return src.rep();
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC en_dtypes::float4_e1m2
bit_cast<en_dtypes::float4_e1m2, uint8_t>(const uint8_t& src) {
  return en_dtypes::float4_e1m2::FromRep(src);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint8_t
bit_cast<uint8_t, en_dtypes::float4_e1m2>(const en_dtypes::float4_e1m2& src) {
  return src.rep();
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC en_dtypes::float6_e2m3
bit_cast<en_dtypes::float6_e2m3, uint8_t>(const uint8_t& src) {
  return en_dtypes::float6_e2m3::FromRep(src);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint8_t
bit_cast<uint8_t, en_dtypes::float6_e2m3>(const en_dtypes::float6_e2m3& src) {
  return src.rep();
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC en_dtypes::float6_e3m2
bit_cast<en_dtypes::float6_e3m2, uint8_t>(const uint8_t& src) {
  return en_dtypes::float6_e3m2::FromRep(src);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint8_t
bit_cast<uint8_t, en_dtypes::float6_e3m2>(const en_dtypes::float6_e3m2& src) {
  return src.rep();
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC en_dtypes::float8_e8m0
bit_cast<en_dtypes::float8_e8m0, uint8_t>(const uint8_t& src) {
  return en_dtypes::float8_e8m0::FromRep(src);
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC uint8_t
bit_cast<uint8_t, en_dtypes::float8_e8m0>(const en_dtypes::float8_e8m0& src) {
  return src.rep();
}

}  // namespace numext

// Work-around for isinf/isnan/isfinite issue on aarch64.
namespace internal {
// TODO: test
template <>
EIGEN_DEVICE_FUNC inline bool isnan_impl<en_dtypes::float8_e8m0>(
    const en_dtypes::float8_e8m0& x) {
  return en_dtypes::floatn_internal::isnan(x);
}

template <>
EIGEN_DEVICE_FUNC inline bool isfinite_impl<en_dtypes::float8_e8m0>(
    const en_dtypes::float8_e8m0& x) {
  return en_dtypes::floatn_internal::isfinite(x);
}

}  // namespace internal
}  // namespace Eigen

#endif  // EN_DTYPES_FLOATN_H_
