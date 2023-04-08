"""
Utility functions for transpiling normal gates into trapped-ion gates, such
as circuit identites.
"""
import numpy as np
import pennylane as qml

from pennylane.ops import GPI, GPI2

from .ionizer_identity_hunter import lookup_gate_identity


def search_and_apply_two_gate_identities(gates_to_apply):
    with qml.QueuingManager.stop_recording():
        identity_to_apply = lookup_gate_identity(gates_to_apply)

    # If there is an identity apply it and move on; otherwise just apply the gates
    if identity_to_apply is not None:
        if len(identity_to_apply) > 0:
            for gate in identity_to_apply:
                qml.apply(gate)
        return True

    # Another special case with no fixed angles: GPI2(x) GPI2(x) = GPI(x)
    if gates_to_apply[0].name == "GPI2" and gates_to_apply[1].name == "GPI2":
        if qml.math.isclose(gates_to_apply[0].data[0], gates_to_apply[1].data[0]):
            qml.apply(GPI(gates_to_apply[0].data[0]))
            return True

    qml.apply(gates_to_apply[0])
    qml.apply(gates_to_apply[1])
    return False


def search_and_apply_three_gate_identities(gates_to_apply):
    # First, check if we can apply an identity to all three gates
    with qml.QueuingManager.stop_recording():
        three_gate_identity_to_apply = lookup_gate_identity(gates_to_apply)

    if three_gate_identity_to_apply is not None:
        if len(three_gate_identity_to_apply) > 0:
            for gate in three_gate_identity_to_apply:
                qml.apply(gate)
        return
    # If we can't apply a 3-gate identity, see if there is a 2-gate one
    else:
        with qml.QueuingManager.stop_recording():
            identity_to_apply = lookup_gate_identity(gates_to_apply)

        if identity_to_apply is not None:
            search_and_apply_two_gate_identities(gates_to_apply[:2])
            qml.apply(gates_to_apply[2])
        else:
            # If not, apply the first gate, then check if there is anything to be
            # done between the second and third.
            qml.apply(gates_to_apply[0])
            search_and_apply_two_gate_identities(gates_to_apply[1:])
