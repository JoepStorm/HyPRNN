"""Canonical material properties shared across the project.

NOTE: the models in trained_models/ were trained with these values.
Do not change them without retraining.
"""

# Wood (chips): Neo-Hookean, Lame constants derived from E and nu.
WOOD_E = 13e3
WOOD_NU = 0.375
WOOD_MU = WOOD_E / (2 * (1 + WOOD_NU))                                # ~4.727e3
WOOD_LAMBDA = WOOD_E * WOOD_NU / ((1 + WOOD_NU) * (1 - 2 * WOOD_NU))  # ~14.182e3

# Fungi (mycelium matrix): Neo-Hookean reference values (~E=1.3, nu=0.27).
# mu is typically varied per sample around FUNGI_MU; lambda is kept fixed.
FUNGI_MU = 0.51
FUNGI_LAMBDA = 0.62
