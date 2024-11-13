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
#ifndef EN_DTYPES_COMMON_INC_H_
#define EN_DTYPES_COMMON_INC_H_


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
namespace convert_internal {
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

  template <typename Bits>
  constexpr inline Bits RoundBitsToNearestEven(Bits bits, int roundoff) {
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

  template <typename Bits>
  constexpr inline Bits RoundBitsHalfToAway(Bits bits, int roundoff) {
    // Round Half To Away
    // Consider a bit pattern
    //   FFF...FLRTT...T,
    // where bits RTT...T need to be rounded-off.
    // we add one bit to R and then truncate bits from R.
    return roundoff == 0
              ? bits
              : bits + (Bits{1} << (roundoff - 1));
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

  template <bool kSaturate, bool kTruncate, typename From, typename To>
  EIGEN_DEVICE_FUNC To ConvertFrom(const From from) {
    // We are rounding long double -> float -> To. This can induce
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
      return ConvertImpl<float, To, kSaturate, kTruncate>::run(
          is_negative ? -abs_narrow : abs_narrow);
    } else {
      return ConvertImpl<From, To, kSaturate, kTruncate>::run(from);
    }
  }
} // convert_internal
} // namespace en_dtypes
#endif // EN_DTYPES_COMMON_INC_H_

