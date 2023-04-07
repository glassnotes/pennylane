"""
Utility functions for transpiling normal gates into trapped-ion gates, such
as circuit identites.
"""
import numpy as np
import pennylane as qml

from pennylane.ops import GPI, GPI2


def _apply_two_gate_identities(gates_to_apply, atol=1e-6):
    first_gate, second_gate = gates_to_apply

    if first_gate.name == "GPI2" and second_gate.name == "GPI":
        # GPI(0) GPI2(0) = GPI2(0)
        if np.allclose([first_gate.data[0], second_gate.data[0]], [0.0, 0.0], atol=atol):
            GPI2(np.pi, wires=first_gate.wires)
            return True
        # GPI(0) GPI2(\pm π) = GPI2(0)
        elif np.allclose(
            [np.abs(first_gate.data[0]), second_gate.data[0]], [np.pi, 0.0], atol=atol
        ):
            GPI2(0.0, wires=first_gate.wires)
            return True
    elif first_gate.name == "GPI2" and second_gate.name == "GPI2":
        # GPI2(x) GPI2(x) = GPI(x)
        if np.isclose(first_gate.data[0], second_gate.data[0]):
            GPI(first_gate.data[0], wires=first_gate.wires)
            return True
        # GPI2(0) GPI2(\pm π) = GPI2(\pm π) GPI2(0) = I
        elif np.allclose(
            np.sort([np.abs(first_gate.data[0]), np.abs(second_gate.data[0])]),
            [0.0, np.pi],
            atol=atol,
        ):
            return True
    elif first_gate.name == "GPI" and second_gate.name == "GPI2":
        # GPI2(\pm π) GPI(\pm π) =  GPI2(0)
        if np.allclose(
            [np.abs(first_gate.data[0]), np.abs(second_gate.data[0])], [np.pi, np.pi], atol=atol
        ):
            GPI2(0, wires=first_gate.wires)
            return True
        # GPI2(0) GPI(0) = GPI2(np.pi)
        elif np.allclose([first_gate.data[0], second_gate.data[0]], [0.0, 0.0], atol=atol):
            GPI2(np.pi, wires=first_gate.wires)
            return True
        # GPI2(np.pi) GPI(0) = GPI2(0)
        elif np.allclose([first_gate.data[0], second_gate.data[0]], [0.0, np.pi], atol=atol):
            GPI2(0, wires=first_gate.wires)
            return True
        # GPI(0) GPI2(\theta) = GPI2(-\theta) GPI(0); prioritize 0s on the right for commutation
        # through the MS gates.
        elif np.isclose(first_gate.data[0], 0.0, atol=atol):
            GPI2(-second_gate.data[0], wires=second_gate.wires)
            GPI(0, wires=first_gate.wires)
            return True

    return False


def _apply_three_gate_identities(gates_to_apply, atol=1e-6):
    gamma, beta, alpha = [gate.data[0] for gate in gates_to_apply]

    # Check for a number of common circuit identities
    # GPI2(0) GPI(π/4) GPI2(0) = GPI2(-np.pi/2)
    if np.allclose([alpha, beta, gamma], [0, np.pi / 4, 0], atol=atol):
        GPI2(-np.pi / 2, wires=gates_to_apply[0].wires)
        return
    # GPI2(π) GPI(π/2) GPI2(0) = GPI(π/2) GPI(0)
    elif np.allclose([alpha, beta, gamma], [np.pi, np.pi / 2, 0], atol=atol):
        GPI(0, wires=gates_to_apply[0].wires)
        GPI(np.pi / 2, wires=gates_to_apply[0].wires)
        return
    # GPI2(π/2) GPI(π/4) GPI2(0) = GPI(π/4) GPI(0)
    elif np.allclose([alpha, beta, gamma], [np.pi / 2, np.pi / 4, 0], atol=atol):
        GPI(0, wires=gates_to_apply[0].wires)
        GPI(np.pi / 4, wires=gates_to_apply[0].wires)
        return
    # GPI2(-π/2) GPI(-π/4) GPI2(-π/2) = GPI2(π)
    elif np.allclose([alpha, beta, gamma], [-np.pi / 2, -np.pi / 4, -np.pi / 2], atol=atol):
        GPI2(np.pi, wires=gates_to_apply[0].wires)
        return
    # GPI2(π/2) GPI(-π/4) GPI2(-π) = GPI(π/4) GPI(0)
    elif np.allclose([alpha, beta, gamma], [np.pi / 2, -np.pi / 4, -np.pi], atol=atol):
        GPI(0, wires=gates_to_apply[0].wires)
        GPI(-np.pi / 4, wires=gates_to_apply[0].wires)
        return
    # GPI2(π/2) GPI(-π/4) GPI2(π/2) = GPI2(0)
    elif np.allclose([alpha, beta, gamma], [np.pi / 2, -np.pi / 4, np.pi / 2], atol=atol):
        GPI2(0, wires=gates_to_apply[0].wires)
        return
    elif np.allclose([alpha, beta, gamma], [np.pi, -np.pi / 4, 0], atol=atol):
        GPI2(np.pi / 2, wires=gates_to_apply[0].wires)
        GPI(0.0, wires=gates_to_apply[0].wires)
        return
    # GPI2(\pm π) GPI(0) GPI2(\pm π) = I
    elif np.allclose([np.abs(alpha), beta, np.abs(gamma)], [np.pi, 0, np.pi], atol=atol):
        return

    # Apply the gates
    for gate in gates_to_apply:
        qml.apply(gate)
