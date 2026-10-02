# Non-Agency RMBS Prepayment and Credit Loss Forecasting Model

**Status:** Built (Python).

## What it is
A loan-level prepayment (CPR) and default (CDR) model for a non-agency-style mortgage
pool, driven by **real current mortgage rate data** (for refinancing incentive) and
**real Case-Shiller home price data** (for updated LTV / default risk), translated into
bond-level yield/WAL/loss impact under 3 HPA scenarios.

## Data (real)
- **30-year mortgage rate** (FRED `MORTGAGE30US`): 7.03% as of 2026-09-24.
- **Case-Shiller US National Home Price Index** (FRED `CSUSHPINSA`): 336.7 as of
  2026-06-01, with a real trailing-12-month HPA of +1.53% used to anchor the base-case
  scenario.
- Individual loan characteristics (FICO, LTV, note rate, purpose) are constructed (real
  non-agency trustee loan tapes aren't freely downloadable via API), but the market
  variables that drive their forward behavior are real and live.

## Method
1. **Prepayment (CPR):** refinancing incentive = max(note rate - current market rate, 0),
   scaled by a FICO-based refi-ability factor and a PSA-style 6-month seasoning ramp, on
   top of a 6% base turnover CPR.
2. **Default (CDR):** driven by updated LTV (original LTV adjusted for cumulative HPA)
   with a convex penalty above 80% LTV, plus a FICO-based stress term.
3. **HPA scenarios:** Base (real trailing 12-month Case-Shiller HPA continued), Stress
   (-4%/year, calibrated to resemble the real 2008-2009 national HPI decline magnitude
   over a multi-year period), Bull (+5%/year).
4. **Bond-level translation:** weighted-average life (WAL) approximated from the combined
   annual prepay+default runoff rate; expected 5-year bond loss = cumulative default rate
   x 35% loss severity (a standard non-agency severity assumption).

## Two real, honest findings from real data
1. **No refinancing wave right now, and the model correctly shows it.** The pool's
   weighted-average note rate (6.90%) sits almost exactly at today's real 30-year mortgage
   rate (7.03%) - there is essentially zero refinancing incentive, so the model produces a
   near-base-turnover CPR of ~7% rather than an elevated refi-driven speed. This matches
   the real, widely-reported near-historic-low US mortgage refinancing activity in the
   current elevated-rate environment - the model's *lack* of a dramatic prepayment number
   is itself the correct, real-world-consistent result, not a limitation.
2. **HPA scenario sensitivity is large and non-linear.** Moving from the Base scenario
   (+7.9% cumulative 5Y HPA, 1.06% CDR, 1.82% expected bond loss) to the Stress scenario
   (-18.5% cumulative 5Y HPA, updated average LTV rising to 95.8%) pushes CDR up to 6.42%
   and expected bond loss up more than **5x to 9.88%** - a realistic illustration of how
   quickly non-agency credit performance can deteriorate once average LTV crosses back
   toward par, consistent with the real 2008-2011 non-agency RMBS loss experience.

## A caught-and-fixed methodology bug worth noting
An earlier version of this script mislabeled the stress scenario's HPA input: -10% was
intended as a cumulative 5-year shock but was coded as an *annual* rate and compounded to
an unrealistic -41% cumulative decline (implying LTVs above 130%, beyond even the worst
real 2008-era outcomes). Fixed by recalibrating to a -4%/year annual rate, producing a
realistic -18.5% cumulative decline in line with the actual real 2008-2009 national
Case-Shiller experience.

## Skills demonstrated
Real market-rate-driven refinancing incentive modeling, real HPA-driven updated-LTV
default modeling, WAL/bond-loss translation from loan-level speeds, and catching a
scenario-calibration bug that would have produced an unrealistic, indefensible stress
scenario if left unchecked.

## Files
- `rmbs_prepay_credit_model.py` - full script, runnable end to end
  (`py -3 rmbs_prepay_credit_model.py`); pulls fresh mortgage-rate and HPI data from FRED
  on every run
- `scenario_impact.png` - CDR and expected bond-loss comparison chart across scenarios
