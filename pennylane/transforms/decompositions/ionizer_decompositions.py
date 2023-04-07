"""
Custom decompositions of operations into the {GPI, GPI2, MS} native gate set.
"""
import pennylane.math as math
import numpy as np

from pennylane.ops import GPI, GPI2, MS
from .ionizer_decomposition_utils import extract_gpi2_gpi_gpi2_angles


# Non-parametrized operations (up to phases)
def gpi_pauli_x(wires):
    return [GPI(0.0, wires=wires)]


def gpi_pauli_y(wires):
    return [GPI(np.pi / 2, wires=wires)]


def gpi_pauli_z(wires):
    return [GPI(0.0, wires=wires), GPI(-np.pi / 2, wires=wires)]


def gpi_hadamard(wires):
    return [GPI(0.0, wires=wires), GPI2(-np.pi / 2, wires=wires)]


def gpi_sx(wires):
    return [GPI2(0.0, wires=wires)]


def gpi_cnot(wires):
    return [
        GPI2(np.pi / 2, wires=wires[0]),
        MS(wires=wires),
        GPI2(np.pi, wires=wires[0]),
        GPI2(np.pi, wires=wires[1]),
        GPI2(-np.pi / 2, wires=wires[0]),
    ]


# Parametrized operations (up to phases)
def gpi_rx(phi, wires):
    return [
        GPI2(np.pi / 2, wires=wires),
        GPI(phi / 2 - np.pi / 2, wires=wires),
        GPI2(np.pi / 2, wires=wires),
    ]


def gpi_ry(phi, wires):
    return [GPI2(np.pi, wires=wires), GPI(phi / 2, wires=wires), GPI2(np.pi, wires=wires)]


def gpi_rz(phi, wires):
    return [GPI(-phi / 2, wires=wires), GPI(0.0, wires=wires)]


def gpi_single_qubit_unitary(U, wires):
    # Single-qubit unitary as a sequence of 3 GPI/GPI2 operations.
    # Check in case we have the identity
    if math.allclose(U, math.eye(2)):
        return []

    # Special case: if we have off-diagonal elements this is a single GPI
    if math.isclose(U[0, 0], 0.0):
        angle = math.angle(U[1, 0])
        return [GPI(angle, wires=wires)]

    # Special case: if we have off-diagonal 0s but it is not the identity,
    # this is an RZ which is a sequence of two GPIs.
    if math.allclose([U[0, 1], U[1, 0]], [0.0, 0.0]):
        return gpi_rz(2 * math.angle(U[1, 1]), wires)

    # Special case: if both diagonal elements are 1/sqrt(2), this is a GPI2
    if math.allclose([U[0, 0], U[1, 1]], [1 / np.sqrt(2), 1 / np.sqrt(2)]):
        angle = math.angle(U[1, 0]) + np.pi / 2
        return [GPI2(angle, wires=wires)]

    # In the general case we must compute and return all three angles.
    gamma, beta, alpha = extract_gpi2_gpi_gpi2_angles(U)

    return [GPI2(gamma, wires=wires), GPI(beta, wires=wires), GPI2(alpha, wires=wires)]


decomp_map = {
    "PauliX": gpi_pauli_x,
    "PauliY": gpi_pauli_y,
    "PauliZ": gpi_pauli_z,
    "Hadamard": gpi_hadamard,
    "SX": gpi_sx,
    "CNOT": gpi_cnot,
    "RX": gpi_rx,
    "RY": gpi_ry,
    "RZ": gpi_rz,
}
