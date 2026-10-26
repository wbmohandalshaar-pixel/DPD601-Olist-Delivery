# Data and evaluation decisions

The unit of analysis is one order. Item and payment tables are aggregated separately before joining, preventing many-to-many inflation. The seven supplied source tables are retained unchanged. Eligible delivered orders with valid milestone dates define the conditional modelling population (96,394 orders; 6,532 late). This does not estimate cancellation or non-delivery risk.

A delivery is late when its arrival calendar date exceeds its estimated calendar date. A punctual label is available at arrival; a late label is available at the first midnight after the estimated date. This relies on the supplied milestone timestamps. Arrival and carrier dates never enter model features.

Cohort membership depends only on approval time. A fixed 90-day embargo precedes each training boundary. The entire experiment fails if any training label is unavailable at the evaluation clock; individual slow deliveries are never silently removed. Final training precedes 2017-10-03; validation is January 2018, split at January 16 for calibration and threshold choice. Historical final evaluation begins May 2018. Unused approval-date gaps are not outcome-based exclusions.

Thirteen candidates are compared in three forward temporal folds (39 fits) using mean average precision. Preprocessing is fitted only on training data. Log/cyclic feature transforms are deterministic. The winning family configuration is frozen before final-period metrics. Early validation fits a sigmoid calibration; later validation chooses an F1 threshold. This is retrospective development completed before the final evaluation clock, not a simulated live January deployment.

The final period overlaps data already examined in v1, so it is historical out-of-time evidence, not a new untouched holdout. Model improvements need genuinely new data for independent confirmation. Calibration did not improve final-period Brier score and is not presented as a success. Bootstrap intervals condition on the fitted model; they do not cover model-selection uncertainty or future drift.

The raw dataset is supplied for this assignment. No additional external source, invented observation, team member, or peer review is asserted.
