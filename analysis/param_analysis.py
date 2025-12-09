from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class MatchResult:
    H: int
    D_ff: int
    L: int
    r: int
    k: Optional[int]
    Dff_prime_exact: float
    Dff_prime_rounded: int
    params_dense: int
    params_struct: int
    param_diff: int
    flops_dense: Optional[float]
    flops_struct: Optional[float]
    flops_ratio: Optional[float]

def _round_to(x: float, mode: Literal["round","floor","ceil"]="floor",
              multiple_of: Optional[int]=None) -> int:
    import math
    if mode == "round":
        y = int(round(x))
    elif mode == "ceil":
        y = int(math.ceil(x))
    else:
        y = int(math.floor(x))
    if multiple_of and multiple_of > 1:
        # round to nearest multiple (ties to nearest)
        y = max(multiple_of, int(round(y / multiple_of) * multiple_of))
    return y

def match_structmoe_to_dense(
    H: int,
    *,
    D_ff: Optional[int] = None,
    expansion_factor: Optional[float] = None,
    L: int,
    r: int,
    k: Optional[int] = None,                 # for FLOPs (optional)
    rounding: Literal["round","floor","ceil"] = "floor",
    multiple_of: Optional[int] = None        # e.g., 256 if you want padded FFN dims
) -> MatchResult:
    """
    Solve for D_ff' that matches dense params:
      P_dense = 2 H D_ff
      P_struct = 2 H D_ff' + L r (H + D_ff') + H L
      D_ff' = (2 H D_ff - H L (r + 1)) / (2 H + L r)

    If expansion_factor is given, D_ff = int(expansion_factor * H).
    Returns both exact and rounded D_ff', plus (optional) FLOPs.
    """
    assert (D_ff is None) ^ (expansion_factor is None), "Provide exactly one of D_ff or expansion_factor."
    if D_ff is None:
        D_ff = int(round(expansion_factor * H))

    # exact D_ff' from the paper equation
    numerator   = 2 * H * D_ff - H * L * (r + 1)
    denominator = 2 * H + L * r
    Dff_prime_exact = numerator / denominator

    # rounded D_ff' (integer and optional multiple-of)
    Dff_prime_rounded = _round_to(Dff_prime_exact, mode=rounding, multiple_of=multiple_of)

    # parameter counts
    params_dense  = 2 * H * D_ff
    params_struct = 2 * H * Dff_prime_rounded + L * r * (H + Dff_prime_rounded) + H * L
    param_diff    = params_struct - params_dense  # >0 means StructMoE ended up larger (due to rounding)

    # optional FLOPs (multiply-add counted as 2 FLOPs)
    if k is not None:
        flops_dense  = 4 * H * D_ff
        flops_struct = 4 * H * Dff_prime_rounded + 2 * H * L + 2 * L * r * (H + Dff_prime_rounded) + Dff_prime_rounded
        flops_ratio  = flops_struct / flops_dense
    else:
        flops_dense = flops_struct = flops_ratio = None

    return MatchResult(
        H=H, D_ff=D_ff, L=L, r=r, k=k,
        Dff_prime_exact=Dff_prime_exact,
        Dff_prime_rounded=Dff_prime_rounded,
        params_dense=params_dense,
        params_struct=params_struct,
        param_diff=param_diff,
        flops_dense=flops_dense,
        flops_struct=flops_struct,
        flops_ratio=flops_ratio,
    )

# --- Example (your paper’s numbers) ---
if __name__ == "__main__":
    res = match_structmoe_to_dense(
        H=2048,
        D_ff=8192,         # or expansion_factor=3.5
        L=16,
        r=16,
        k=16,               # for FLOPs ratio; omit if you don’t care
        rounding="floor",  # match the paper (gives 6618)
        multiple_of=None   # set to 256 if you want padded FFN dims
    )
    print(res)
    # Dff_prime_exact ≈ 6618.353; Dff_prime_rounded = 6618