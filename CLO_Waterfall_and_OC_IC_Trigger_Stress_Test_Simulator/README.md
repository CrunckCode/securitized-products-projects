# CLO Waterfall and OC/IC Trigger Stress-Test Simulator

**Status:** Built (Python).

## What it is
A full CLO capital structure (AAA through equity) built on a synthetic 200-loan
broadly-syndicated leveraged loan collateral pool, with real Moody's WARF rating factors,
floating-rate coupons pegged to the real current SOFR, and OC/IC test cure mechanics
(diverting principal to delever the senior-most tranche on a covenant breach) stress-tested
under 3 default-rate scenarios consistent with real rating-agency CLO stress conventions.

## Data (real where it matters, clearly labeled where constructed)
- **Real current SOFR** (FRED, 3.88% as of 2026-09-24) sets every loan's and every
  tranche's floating-rate coupon base.
- **Real Moody's WARF (Weighted Average Rating Factor) table** - the actual published
  factor-per-rating-notch scale (Ba1=940 through Caa1=4,770) used industry-wide to score
  collateral pool credit quality.
- **Real typical leveraged loan spread-by-rating levels** (Ba1 ~2.75% over SOFR through
  Caa1 ~7.00%) and **real typical BSL CLO 2.0 tranche subordination percentages**
  (AAA ~62%, down to ~9% equity).
- Individual loan identities/industries are constructed (real trustee-level CLO loan
  tapes aren't freely downloadable via API), but every rate, spread, and structural
  parameter applied to them is a real, published industry convention.

## Method
1. Build the 200-loan pool, compute portfolio WARF (2,296 - actually **better** credit
   quality than the typical BSL CLO target range of 2,600-3,200, an honest artifact of
   this run's random rating draw skewing toward stronger credits) and weighted-average
   spread (4.30% over SOFR).
2. Size the CLO capital structure using real typical BSL CLO subordination levels.
3. **Calibrate OC test thresholds with a real closing-date cushion** (4-6 points below
   each tranche's actual par-value coverage ratio at inception) - an early version of this
   script set test thresholds independent of the actual sizing and produced nonsensical
   immediate breaches even in the base case; fixed by deriving each test level from the
   real inception OC ratio minus a realistic cushion, matching how real deals are
   actually structured to pass their own covenants at closing.
4. Simulate 5 years of loan-level defaults under 3 scenarios, running the interest and
   principal waterfall each year, checking OC tests cumulatively down the stack, and
   diverting principal to pay down the AAA tranche whenever a test fails (the real CLO
   cure mechanic).

## Results (this run, real-SOFR-priced coupons)
| Scenario | Annual default rate | 5Y realized default rate | Cumulative losses | OC breaches |
|---|---|---|---|---|
| Base | 2% | 7.5% | $27.2M | First breach at Year 2 (not immediately) |
| Adverse | 5% | 22.5% | $102.9M | Breaches from Year 0 |
| Severely Adverse | 9% | 33.0% | $182.9M | Breaches from Year 0, BBB/BB cures exhausted |

**The base case correctly does NOT breach immediately** - tests hold through Year 1 and
only start failing in Year 2 as real cumulative defaults erode the collateral cushion,
which is the behaviorally correct pattern for a properly-cushioned deal (a genuine fix
from the earlier broken version, and the more interesting result to be able to explain:
*why* a well-structured deal should pass its covenants at closing and only fail after real
credit deterioration accumulates).

**AAA remains protected in all three scenarios**, even severely adverse (33% cumulative
default rate) - a real, expected finding: AAA's 62% subordination and its position at the
top of a cure-mechanic waterfall that diverts principal specifically to protect it is
exactly why AAA CLO tranches have such a strong real-world loss history even through the
2008 and 2020 credit cycles. In the severely adverse scenario, the BBB and BB cure amounts
drop to $0 (their own OC cushion is fully exhausted diverting to AAA first), showing the
capital structure's real subordination hierarchy under genuine stress.

## Honesty note on scope
Individual loan identities and the loan-level default draws are synthetic; the WARF
factors, rating-to-spread mapping, tranche subordination percentages, and the SOFR base
rate are real, published, or live market values. The waterfall models OC test cure
mechanics but simplifies the interest waterfall (doesn't fully model interest diversion
separately from principal diversion).

## Skills demonstrated
Real Moody's WARF-based collateral quality scoring, CLO capital structure construction
using real subordination conventions, OC/IC test cure-mechanic simulation, and - notably -
catching and fixing a real structural calibration bug (test thresholds inconsistent with
actual sizing) that had been producing nonsensical immediate breaches.

## Files
- `clo_waterfall.py` - full script, runnable end to end (`py -3 clo_waterfall.py`); pulls
  the real current SOFR rate from FRED on every run
