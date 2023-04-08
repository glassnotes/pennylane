"""
Custom decompositions of operations into the {GPI, GPI2, MS} native gate set.
"""
import pennylane.math as math
import numpy as np


def _rescale_phases(phases, renormalize=False):
    """Rescale a phase value into a fixed range between -np.pi and np.pi.

    Args:
        phases (tensor): The phases to rescale.
        renormalize (bool): By default, we rescale into the range -np.pi to
            np.pi. If this is set to True, rescale instead into the range -1 to
            1 (-2\pi to 2\pi) as this the range of phases accepted by IonQ's
            native gate input specs.

    Return:
        (tensor): The rescaled phases.

    """
    scaled_phases = math.arctan2(math.sin(phases), math.cos(phases))

    if renormalize:
        scaled_phases = scaled_phases / (2 * np.pi)

    return scaled_phases


def extract_gpi2_gpi_gpi2_angles(U):
    """Given a matrix U, recovers a set of three angles alpha, beta, and
    gamma such that
        U = GPI2(alpha) GPI(beta) GPI2(gamma)
    up to a global phase.

    Args:
        U (tensor): A unitary matrix.

    Returns:
        (tensor): Rotation angles for the GPI/GPI2 gates. The order of the
        returned angles corresponds to the order in which they would be
        implemented in the circuit.
    """
    det = math.angle(math.linalg.det(U))
    su2_mat = math.exp(-1j * det / 2) * U

    phase_00 = math.angle(su2_mat[0, 0])
    phase_10 = math.angle(su2_mat[1, 0])

    alpha = phase_10 - phase_00 + np.pi
    beta = math.arccos(math.abs(su2_mat[0, 0])) + phase_10 + np.pi
    gamma = phase_10 + phase_00 + np.pi

    return _rescale_phases([gamma, beta, alpha])
