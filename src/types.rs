// Copyright 2016-2019 Matthew D. Michelotti
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Standard definitions of `NoisyFloat`.
//!
//! Definitions in this module all use `debug_assert!`
//! to check for valid values, so there is no overhead
//! when running in an optimized build.

use crate::{
    NoisyFloat,
    checkers::{NumChecker, FiniteChecker},
};

/// A floating point number behaving like `f32` that does not allow NaN.
///
/// The "N" in the name stands for "Number", since all values of this type
/// are "numbers", i.e. they are not "not-a-number".
pub type N32 = NoisyFloat<f32, NumChecker>;

/// A floating point number behaving like `f64` that does not allow NaN.
///
/// The "N" in the name stands for "Number", since all values of this type
/// are "numbers", i.e. they are not "not-a-number".
pub type N64 = NoisyFloat<f64, NumChecker>;

/// A floating point number behaving like `f32` that does not allow NaN or +/- Infinity.
///
/// The "R" in the name stands for "Real", since in Mathematics, the Real
/// numbers do not include NaN or +/- Infinity.
pub type R32 = NoisyFloat<f32, FiniteChecker>;

/// A floating point number behaving like `f64` that does not allow NaN or +/- Infinity.
///
/// The "R" in the name stands for "Real", since in Mathematics, the Real
/// numbers do not include NaN or +/- Infinity.
pub type R64 = NoisyFloat<f64, FiniteChecker>;


/// Shorthand for `N32::new(value)`.
#[inline]
pub fn n32(value: f32) -> N32 {
    N32::new(value)
}

/// Shorthand for `N64::new(value)`.
#[inline]
pub fn n64(value: f64) -> N64 {
    N64::new(value)
}

/// Shorthand for `R32::new(value)`.
#[inline]
pub fn r32(value: f32) -> R32 {
    R32::new(value)
}

/// Shorthand for `R64::new(value)`.
#[inline]
pub fn r64(value: f64) -> R64 {
    R64::new(value)
}

#[cfg(feature = "alga-real")]
macro_rules! impl_alga_real {
    ($t:ty, $c:path, $f:ty) => {
        impl ::approx::AbsDiffEq for $t {
            type Epsilon = $t;

            fn default_epsilon() -> Self::Epsilon {
                $c(<$f>::default_epsilon())
            }
            fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                self.raw().abs_diff_eq(&other.raw(), epsilon.raw())
            }
        }

        impl ::approx::UlpsEq for $t {
            fn default_max_ulps() -> u32 {
                <$f>::default_max_ulps()
            }
            fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                self.raw().ulps_eq(&other.raw(), epsilon.raw(), max_ulps)
            }
        }

        impl ::approx::RelativeEq for $t {
            fn default_max_relative() -> Self::Epsilon {
                use approx::UlpsEq;
                $c(f32::default_max_ulps() as $f)
            }
            fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
                self.raw().relative_eq(&other.raw(), epsilon.raw(), max_relative.raw())
            }
        }

        impl ::alga::general::Identity<::alga::general::Additive> for $t {
            #[inline]
            fn identity() -> Self { $c(0.0) }
        }
        impl ::alga::general::Identity<::alga::general::Multiplicative> for $t {
            #[inline]
            fn identity() -> Self { $c(1.0) }
        }
        impl ::alga::general::AbstractSemigroup<::alga::general::Additive> for $t {}
        impl ::alga::general::AbstractSemigroup<::alga::general::Multiplicative> for $t {}
        impl ::alga::general::AbstractQuasigroup<::alga::general::Additive> for $t {}
        impl ::alga::general::AbstractQuasigroup<::alga::general::Multiplicative> for $t {}
        impl ::alga::general::TwoSidedInverse<::alga::general::Additive> for $t {
            #[inline]
            fn two_sided_inverse(&self) -> Self {
                $c(-self.raw())
            }
        }
        impl ::alga::general::TwoSidedInverse<::alga::general::Multiplicative> for $t {
            #[inline]
            fn two_sided_inverse(&self) -> Self {
                $c(1.0 / self.raw())
            }
        }
        impl ::alga::general::AbstractMagma<::alga::general::Additive> for $t {
            #[inline]
            fn operate(&self, lhs: &Self) -> Self {
                $c(self.raw() + lhs.raw())
            }
        }
        impl ::alga::general::AbstractMagma<::alga::general::Multiplicative> for $t {
            #[inline]
            fn operate(&self, lhs: &Self) -> Self {
                $c(self.raw() * lhs.raw())
            }
        }
        impl ::alga::general::AbstractMonoid<::alga::general::Additive> for $t {}
        impl ::alga::general::AbstractMonoid<::alga::general::Multiplicative> for $t {}
        impl ::alga::general::AbstractLoop<::alga::general::Additive> for $t {}
        impl ::alga::general::AbstractLoop<::alga::general::Multiplicative> for $t {}
        impl ::alga::general::AbstractGroup<::alga::general::Additive> for $t {}
        impl ::alga::general::AbstractGroup<::alga::general::Multiplicative> for $t {}
        impl ::alga::general::AbstractGroupAbelian<::alga::general::Additive> for $t {}
        impl ::alga::general::AbstractGroupAbelian<::alga::general::Multiplicative> for $t {}
        impl ::alga::general::AbstractRing for $t {}
        impl ::alga::general::AbstractRingCommutative for $t {}
        impl ::alga::general::AbstractField for $t {}

        impl ::alga::general::SubsetOf<$t> for $t {
            #[inline]
            fn to_superset(&self) -> $t {
                *self as $t
            }

            #[inline]
            unsafe fn from_superset_unchecked(element: &$t) -> $t {
                *element as $t
            }

            #[inline]
            fn is_in_subset(_: &$t) -> bool {
                true
            }
        }

        impl ::alga::general::SubsetOf<$t> for f64 {
            #[inline]
            fn to_superset(&self) -> $t {
                $c(*self as $f)
            }

            #[inline]
            unsafe fn from_superset_unchecked(element: &$t) -> f64 {
                element.raw() as f64
            }

            #[inline]
            fn is_in_subset(_: &$t) -> bool {
                true
            }
        }

        impl ::alga::general::JoinSemilattice for $t {
            #[inline]
            fn join(&self, other: &Self) -> Self {
                if *self >= *other {
                    *self
                }
                else {
                    *other
                }
            }
        }
        impl ::alga::general::MeetSemilattice for $t {
            #[inline]
            fn meet(&self, other: &Self) -> Self {
                if *self <= *other {
                    *self
                }
                else {
                    *other
                }
            }
        }
        impl ::alga::general::Lattice for $t {}

        #[rustfmt::skip]
        impl ::alga::general::Real for $t {
            fn floor(self) -> Self {$c(self.raw().floor())}
            fn ceil(self) -> Self {$c(self.raw().ceil())}
            fn round(self) -> Self {$c(self.raw().round())}
            fn trunc(self) -> Self {$c(self.raw().trunc())}
            fn fract(self) -> Self {$c(self.raw().fract())}
            fn abs(self) -> Self {$c(self.raw().abs())}
            fn signum(self) -> Self {$c(self.raw().signum())}
            fn is_sign_positive(self) -> bool {self.raw().is_sign_positive()}
            fn is_sign_negative(self) -> bool {self.raw().is_sign_negative()}
            fn mul_add(self, a: Self, b: Self) -> Self  {$c(self.raw().mul_add(a.raw(), b.raw()))}
            fn recip(self) -> Self {$c(self.raw().recip())}
            fn powi(self, n: i32) -> Self {$c(self.raw().powi(n))}
            fn powf(self, n: Self) -> Self {$c(self.raw().powf(n.raw()))}
            fn sqrt(self) -> Self {$c(self.raw().sqrt())}
            fn exp(self) -> Self {$c(self.raw().exp())}
            fn exp2(self) -> Self {$c(self.raw().exp2())}
            fn ln(self) -> Self {$c(self.raw().ln())}
            fn log(self, base: Self) -> Self {$c(self.raw().log(base.raw()))}
            fn log2(self) -> Self {$c(self.raw().log2())}
            fn log10(self) -> Self {$c(self.raw().log10())}
            fn max(self, other: Self) -> Self {$c(self.raw().max(other.raw()))}
            fn min(self, other: Self) -> Self {$c(self.raw().min(other.raw()))}
            fn cbrt(self) -> Self {$c(self.raw().cbrt())}
            fn hypot(self, other: Self) -> Self {$c(self.raw().hypot(other.raw()))}
            fn sin(self) -> Self {$c(self.raw().sin())}
            fn cos(self) -> Self {$c(self.raw().cos())}
            fn tan(self) -> Self {$c(self.raw().tan())}
            fn asin(self) -> Self {$c(self.raw().asin())}
            fn acos(self) -> Self {$c(self.raw().acos())}
            fn atan(self) -> Self {$c(self.raw().atan())}
            fn atan2(self, other: Self) -> Self {$c(self.raw().atan2(other.raw()))}
            fn sin_cos(self) -> (Self, Self) {let (a, b) = self.raw().sin_cos(); ($c(a), $c(b))}
            fn exp_m1(self) -> Self {$c(self.raw().exp_m1())}
            fn ln_1p(self) -> Self {$c(self.raw().ln_1p())}
            fn sinh(self) -> Self {$c(self.raw().sinh())}
            fn cosh(self) -> Self {$c(self.raw().cosh())}
            fn tanh(self) -> Self {$c(self.raw().tanh())}
            fn asinh(self) -> Self {$c(self.raw().asinh())}
            fn acosh(self) -> Self {$c(self.raw().acosh())}
            fn atanh(self) -> Self {$c(self.raw().atanh())}
            fn pi() -> Self {($c(<$f>::two_pi()))}
            fn two_pi() -> Self {($c(<$f>::two_pi()))}
            fn frac_pi_2() -> Self {($c(<$f>::frac_pi_2()))}
            fn frac_pi_3() -> Self {($c(<$f>::frac_pi_3()))}
            fn frac_pi_4() -> Self {($c(<$f>::frac_pi_4()))}
            fn frac_pi_6() -> Self {($c(<$f>::frac_pi_6()))}
            fn frac_pi_8() -> Self {($c(<$f>::frac_pi_8()))}
            fn frac_1_pi() -> Self {($c(<$f>::frac_1_pi()))}
            fn frac_2_pi() -> Self {($c(<$f>::frac_2_pi()))}
            fn frac_2_sqrt_pi() -> Self {($c(<$f>::frac_2_sqrt_pi()))}
            fn e() -> Self {($c(<$f>::e()))}
            fn log2_e() -> Self {($c(<$f>::log2_e()))}
            fn log10_e() -> Self {($c(<$f>::log10_e()))}
            fn ln_2() -> Self {($c(<$f>::ln_2()))}
            fn ln_10() -> Self {($c(<$f>::ln_10()))}
        }
    };
}

#[cfg(feature = "alga-real")]
impl_alga_real!(R32, r32, f32);
#[cfg(feature = "alga-real")]
impl_alga_real!(R64, r64, f64);
