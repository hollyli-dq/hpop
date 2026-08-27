# Stage B2 smoke — joint scalar sampler

Active: `beta, omega, lambda_rep`; fixed: `{'lambda_back': 0.25}`.

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
| beta | 0.446, 0.442 |
| omega | 0.475, 0.451 |
| lambda_rep | 0.465, 0.455 |
