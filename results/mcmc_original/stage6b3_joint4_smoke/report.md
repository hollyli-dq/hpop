# Stage B3 smoke — joint scalar sampler

Active: `beta, omega, lambda_rep, lambda_back`; fixed: `{}`.

| check | result |
|---|---|
| all_coordinates_move | **PASS** |
| each_coordinate_accepts | **PASS** |
| each_coordinate_rejects | **PASS** |
| no_nans_all_finite | **PASS** |
| q0_reset_and_state_reproducible | **PASS** |
| state_serialises_and_loads | **PASS** |
| deterministic_resume | **PASS** |

| coordinate | post burn-in acceptance |
|---|---|
| beta | 0.454, 0.462 |
| omega | 0.465, 0.439 |
| lambda_rep | 0.411, 0.427 |
| lambda_back | 0.471, 0.469 |
