# Experiment history and evaluation status

V1 used per-row arrival-based boundary exclusions. Its final May-August 2018 outcomes were inspected and reported. A later review measured outcome selection and requested a substantive revision.

V2 uses approval-only cohort membership, a fixed 90-day gap, and label availability based on the date-based target. Temporal CV folds are June, August and October 2017; final January 2018 validation is split into disjoint calibration and threshold blocks. CV labels must be known before January 2018; train labels must be known before each evaluation clock.

The first V2 candidate set compared raw-input Logistic Regression, Random Forest, histogram gradient boosting and Dummy. The selection summary and historical test metrics are retained in outputs/v2_round1. A second development pass adds a declared deterministic log/cyclic representation for Logistic Regression and compares it on the same development folds. The historical final-period results had already been observed. No revised final-period score is presented as a new untouched or prospectively independent holdout. Candidate selection still uses only temporal CV; calibration and threshold use validation only. Results describe a transparent retrospective revision and require genuinely new data for independent confirmation.

No Git team history, reviews or student identities were fabricated. Local code assistance and preparation must be disclosed according to course policy.
