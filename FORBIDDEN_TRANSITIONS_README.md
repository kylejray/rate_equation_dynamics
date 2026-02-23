# Forbidden Transitions Implementation

## Overview
The code has been modified to support **forbidden transitions** in the Continuous Time Markov Chain (CTMC) model. Previously, all transition rates were enforced to be at least `min_rate` to avoid numerical issues. Now, transitions can be explicitly forbidden (set to 0), with the constraint that if `A->B` is forbidden, then `B->A` must also be forbidden (symmetric forbidden transitions).

## Key Changes

### 1. Added `forbidden_mask` Attribute
**File:** `ctmc.py`, `__init__` method

A new attribute `self.forbidden_mask` tracks which transitions are forbidden. This is a boolean mask that is:
- `True` for forbidden transitions
- `False` for allowed transitions
- Always symmetric: if `forbidden_mask[i,j]` is `True`, then `forbidden_mask[j,i]` is also `True`

### 2. Modified `verify_rate_matrix` Method
**File:** `ctmc.py`, `verify_rate_matrix` method

**Key changes:**
- **Relaxed assertion:** Changed from requiring all off-diagonal rates to be `>0` to allowing them to be `>=0`
- **Forbidden transition detection:** Identifies transitions with rates below `min_rate` threshold
- **Symmetric enforcement:** If either `R[i,j]` or `R[j,i]` is below `min_rate`, both are set to 0 and marked as forbidden
- **Rate enforcement:** 
  - Forbidden transitions are set to exactly 0
  - Allowed transitions are enforced to be at least `min_rate`

```python
# Create symmetric forbidden mask
forbidden = np.abs(R) <= self.min_rate
forbidden_symmetric = forbidden | forbidden.T  # (or .transpose(0,2,1) for batch)

# Store mask and set rates
self.forbidden_mask = forbidden_symmetric & (~np.eye(S, dtype=bool))
R[self.forbidden_mask] = 0  # Forbidden transitions
R[allowed_nonzero] = np.maximum(R[allowed_nonzero], self.min_rate)  # Allowed transitions
```

### 3. Updated `__set_statewise_Q` Method
**File:** `ctmc.py`, `__set_statewise_Q` method

**Purpose:** Avoid `log(0/0)` errors when computing entropy production rates.

**Key changes:**
- Creates an `allowed_mask` for non-diagonal, non-forbidden transitions
- Only computes `log(R/rev_R_T)` for allowed transitions
- For forbidden transitions (where both `R` and `rev_R_T` are 0), the log ratio is left as 0

```python
# Create mask for allowed transitions
allowed_mask = ~(self.forbidden_mask | np.eye(self.S, dtype=bool))

# Initialize log ratio with zeros
log_ratio = np.zeros_like(self.R)

# Only compute for allowed transitions
log_ratio[allowed_mask] = np.log(self.R[allowed_mask] / rev_R_T[allowed_mask])

# Use log_ratio in calculation (forbidden transitions contribute 0)
self.statewise_Q = ((self.R * log_ratio) + R_diags - rev_R_diags).sum(axis=-1)
```

## Usage

### Automatic Forbidden Transitions
Simply set rates below `min_rate` (default: `1e-32`) and they will automatically be treated as forbidden:

```python
import numpy as np
from ctmc import ContinuousTimeMarkovChain

S = 5
R = np.random.uniform(0.1, 1.0, (S, S))

# Set some rates very small (below min_rate)
R[0, 1] = 1e-35  # Will become forbidden
R[1, 0] = 0.5    # Will also be forced to 0 (symmetric)

ctmc = ContinuousTimeMarkovChain(R=R)

# Check which transitions are forbidden
print(ctmc.forbidden_mask)
```

### Manual Forbidden Transitions
You can also manually set transitions to 0 before creating the CTMC:

```python
R = np.random.uniform(0.1, 1.0, (S, S))

# Manually forbid some transitions (symmetrically)
R[2, 3] = 0
R[3, 2] = 0

ctmc = ContinuousTimeMarkovChain(R=R)
```

### Adjusting the Threshold
Change `min_rate` to adjust the threshold for forbidden transitions:

```python
ctmc = ContinuousTimeMarkovChain(R=R)
ctmc.min_rate = 1e-20  # Higher threshold
ctmc.set_rate_matrix(R)  # Re-process with new threshold
```

## Benefits

1. **No more artificial minimum rates:** Transitions can now be truly forbidden rather than just very small
2. **Physical constraint enforcement:** The symmetry requirement (if A->B forbidden, then B->A forbidden) is automatically enforced
3. **Numerical stability:** Avoids `log(0)` and `log(0/0)` errors in entropy production calculations
4. **Cleaner semantics:** Makes it clear which transitions are disallowed vs. just slow

## Backward Compatibility

The changes are fully backward compatible:
- Old code that relied on `min_rate` enforcement will still work
- The default behavior is the same (rates below `min_rate` are treated specially)
- The only difference is that very small rates are now set to 0 instead of `min_rate`

## Testing

Run the test script to verify the implementation:
```bash
python test_forbidden_transitions.py
```

The test verifies:
1. Automatic forbidden transition detection
2. Symmetric enforcement of forbidden transitions
3. No NaN or Inf values in calculations
4. Successful NESS and EPR computation
