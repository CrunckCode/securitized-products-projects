# Agency MBS Option-Adjusted Spread (OAS) Model

**Status:** Built (Python).

## What it is
Simulates Hull-White interest-rate paths (real SOFR/Treasury-calibrated), drives a
path-dependent prepayment model off each path's own refinancing incentive, and solves via
bisection for the Option-Adjusted Spread that makes the average simulated present value
equal a target market price - the real, standard Agency MBS relative-value methodology.

## Data (real)
Same real Hull-White calibration (SOFR/Treasury curve) as the Counterparty Exposure
Engine project; real current 30-year mortgage rate (FRED `MORTGAGE30US`, 7.03%) used to
derive the pool's target market price.

## Method
1. Reuse the real-curve-calibrated Hull-White short-rate simulation (1,000 paths, 360
   monthly steps over a 30-year pool tenor).
2. Drive path-dependent CPR using the refinancing-incentive functional form already
   built in the RMBS prepayment project (note rate vs. each path's own simulated
   mortgage rate at each month).
3. Project principal/interest cash flows along every path given that path's own CPR
   schedule, discount at that path's own simulated short rate plus a trial OAS, and
   average across all paths.
4. Bisect the trial OAS until the average present value matches a target market price.
5. Repeat with a single deterministic path (today's real forward curve, no rate
   uncertainty) to get the Z-spread, and take Z-spread minus OAS as the real, standard
   decomposition of the prepayment option's cost.

## An honest correction on the "real observed market price" input
An initial version tried to use the real MBB (iShares MBS ETF) share price directly as
the OAS-solve target - but an ETF share price reflects the fund's AUM per share, not a
specific pool's price relative to a $1/$100 par bond convention, so this was not a valid
comparison and was producing a nonsensical negative OAS (real Agency MBS practically
never trade at negative OAS, since that would mean the market price already exceeds fair
value even after accounting for the prepayment option's cost). Fixed by deriving the
target price honestly from the real current mortgage-rate environment instead: a pool
carrying a below-market 5.50% note rate against the real current 7.03% market rate
should trade at a real discount to par, approximated via a simple duration-based
discount (not claimed to be a specific real pool's exact real price).

## Results (this run)
- **Target price (duration-approximated discount to par): 0.9082**
- **Solved OAS: 0.33%** - a real, plausible, positive value (real Agency MBS OAS
  commonly trades in the 0-100bp range depending on the coupon/rate environment).
- **Z-spread (no prepayment optionality): 2.29%**
- **Option cost (Z-spread minus OAS): 1.96%** - on the higher end of typically-cited real
  Agency MBS option costs (often cited around 30-80bps in normal conditions). This likely
  reflects the model's volatility calibration, drawn from real but somewhat elevated
  2023-2026 SOFR volatility, projected forward over the full 30-year pool horizon - a
  real, honest limitation to flag rather than presenting the option-cost number as
  precisely calibrated to long-run historical norms.

## Skills demonstrated
Monte Carlo OAS solving via bisection, path-dependent prepayment modeling reused across
two projects in this portfolio, Z-spread/OAS decomposition to isolate the real dollar
value of the borrower's prepayment option, and - importantly - catching and correcting a
real conceptual error (treating an ETF share price as if it were a par-based bond price)
before trusting a nonsensical negative OAS result.

## Files
- `agency_mbs_oas.py` - full script, runnable end to end
  (`py -3 agency_mbs_oas.py`); recalibrates against live SOFR/Treasury/mortgage-rate data
  on every run
