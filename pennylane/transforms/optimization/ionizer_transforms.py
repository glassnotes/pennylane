"""
Utility transforms for transpiling normal gates into trapped-ion gates.
"""
import numpy as np

import pennylane as qml
import pennylane.math as math
from pennylane.transforms.optimization.optimization_utils import find_next_gate
from pennylane.transforms import qfunc_transform

from pennylane.transforms.decompositions.ionizer_decomposition_utils import _rescale_phases, extract_gpi2_gpi_gpi2_angles
from pennylane.transforms.decompositions.ionizer_decompositions import decomp_map
from pennylane.ops import GPI, GPI2, MS
from .ionizer_transform_utils import (
    search_and_apply_two_gate_identities,
    search_and_apply_three_gate_identities,
)


@qfunc_transform
def commute_through_ms_gates(tape, direction="right"):
    """Apply a transform that passes through a tape and pushes GPI/GPI2
    gates with appropriate (commuting) angles through MS gates."""
    list_copy = tape.operations.copy()

    if direction == "left":
        list_copy = list_copy[::-1]

    with qml.QueuingManager.stop_recording():
        with qml.tape.QuantumTape() as commuted_tape:
            while len(list_copy) > 0:
                current_gate = list_copy[0]
                list_copy.pop(0)

                # Apply MS as we find them
                if len(current_gate.wires) == 2:
                    qml.apply(current_gate)
                    continue

                # Find the next gate that acts on the same wires
                next_gate_idx = find_next_gate(current_gate.wires, list_copy)

                # If there is no next gate, just apply this one and move on
                if next_gate_idx is None:
                    qml.apply(current_gate)
                    continue

                next_gate = list_copy[next_gate_idx]

                # To limit code duplication, decide whether to apply next gate or not.
                apply_current_gate = True

                # Check if next gate is MS and see if we can commute through it.
                if next_gate.name == "MS":
                    # The following commute through MS gates on either qubit:
                    # GPI2(0), GPI2(π), GPI2(-π), GPI(0), GPI(π), GPI(-π)
                    if current_gate.name in ["GPI", "GPI2"]:
                        angle = math.abs(current_gate.data[0])
                        if math.isclose(angle, 0.0) or math.isclose(angle, np.pi):
                            list_copy.insert(next_gate_idx + 1, current_gate)
                            apply_current_gate = False

                # If we didn't commute this gate through, apply it.
                if apply_current_gate:
                    qml.apply(current_gate)

    if direction == "left":
        for op in commuted_tape.operations[::-1]:
            qml.apply(op)
    else:
        for op in commuted_tape.operations:
            qml.apply(op)

    for m in tape.measurements:
        qml.apply(m)


@qfunc_transform
def virtualize_rz_gates(tape):
    """When dealing with GPI/GPI2/MS gates, RZ gates can be implemented virtually
    by pushing them through such gates and simply adjusting the phases of the
    gates we pushed them through:
        - GPI(x) RZ(z) = GPI(x - z/2)
        - RZ(z) GPI(x) = GPI(x + z/2)
        - GPI2(x) RZ(z) = RZ(z) GPI2(x - z)
        - RZ(z) GPI2(x) = GPI2(x + z) RZ(z)

    This transform rolls through a tape, and adjusts the circuits so that
    all the RZs get implemented virtually.
    """
    list_copy = tape.operations

    while len(list_copy) > 0:
        current_gate = list_copy[0]
        list_copy.pop(0)

        if current_gate.name == "RZ":
            next_gate_idx = find_next_gate(current_gate.wires, list_copy)

            # No gate afterwards; just apply this one but use GPI gates
            if next_gate_idx is None:
                GPI(-current_gate.data[0] / 2, wires=current_gate.wires)
                GPI(0.0, wires=current_gate.wires)
                continue

            # As long as there are more single-qubit gates afterwards, push the
            # RZ through and apply the phase-adjusted gates
            accumulated_rz_phase = current_gate.data[0]
            apply_accumulated_phase_gate = True

            while next_gate_idx is not None:
                next_gate = list_copy[next_gate_idx]

                # If the next gate is an RZ, accumulate its phase into this gate,
                # then remove it from the queue and don't apply anything yet.
                if next_gate.name == "RZ":
                    accumulated_rz_phase += next_gate.data[0]
                    list_copy.pop(next_gate_idx)
                # Apply the identity GPI(θ) RZ(ϕ) = GPI(θ - ϕ/2); then there are no more
                # RZs to process so we leave the loop.
                elif next_gate.name == "GPI":
                    GPI(
                        _rescale_phases(next_gate.data[0] - accumulated_rz_phase / 2),
                        wires=current_gate.wires,
                    )
                    apply_accumulated_phase_gate = False
                    list_copy.pop(next_gate_idx)
                    break
                # Apply the identity GPI2(θ) RZ(ϕ) = RZ(ϕ) GPI2(θ - ϕ); apply the GPI2 gate
                # with adjusted phase.
                elif next_gate.name == "GPI2":
                    GPI2(
                        _rescale_phases(next_gate.data[0] - accumulated_rz_phase),
                        wires=current_gate.wires,
                    )
                    list_copy.pop(next_gate_idx)
                # If it's anything else, we want to just apply it normally
                else:
                    break

                next_gate_idx = find_next_gate(current_gate.wires, list_copy)

            # Once we pass through all the gates, if there is any remaining
            # accumulated phase, apply the RZ gate as two GPI gates. Apply the GPI(0)
            # last, because this gate is likely to be right before an MS gate, and
            # this way we can commute it through.
            if apply_accumulated_phase_gate:
                GPI(-_rescale_phases(accumulated_rz_phase) / 2, wires=current_gate.wires)
                GPI(0.0, wires=current_gate.wires)

        else:
            qml.apply(current_gate)

    for m in tape.measurements:
        qml.apply(m)


@qfunc_transform
def single_qubit_fusion_gpi(tape):
    """Perform single-qubit fusion of all sequences of single-qubit gates into
    no more than 3 GPI/GPI2 gates."""
    # Make a working copy of the list to traverse
    list_copy = tape.operations.copy()

    while len(list_copy) > 0:
        current_gate = list_copy[0]
        list_copy.pop(0)

        # Ignore 2-qubit gates
        if len(current_gate.wires) > 1:
            qml.apply(current_gate)
            continue

        # Find the next gate that acts on the same wires
        next_gate_idx = find_next_gate(current_gate.wires, list_copy)

        # If there is no next gate, just apply this one
        if next_gate_idx is None:
            qml.apply(current_gate)
            continue

        gates_to_apply = [current_gate]

        # Loop as long as a valid next gate exists
        while next_gate_idx is not None:
            next_gate = list_copy[next_gate_idx]

            if len(next_gate.wires) > 1:
                break

            gates_to_apply.append(next_gate)
            list_copy.pop(next_gate_idx)

            next_gate_idx = find_next_gate(current_gate.wires, list_copy)

        # We should only actually do fusion if we find more than 3 gates
        # Otherwise, we just apply them normally
        if len(gates_to_apply) == 1:
            qml.apply(gates_to_apply[0])
        # Try applying identities to sequences of two-qubit gates.
        elif len(gates_to_apply) == 2:
            search_and_apply_two_gate_identities(gates_to_apply)
        # If we have exactly three gates, try applying identities to the sequence
        elif len(gates_to_apply) == 3:
            search_and_apply_three_gate_identities(gates_to_apply)
        # If we have more than three gates, we need to fuse.
        else:
            running_matrix = qml.matrix(gates_to_apply[0])

            for gate in gates_to_apply[1:]:
                running_matrix = np.dot(qml.matrix(gate), running_matrix)

            gamma, beta, alpha = extract_gpi2_gpi_gpi2_angles(running_matrix)

            # If all three angles are the same, GPI2(θ) GPI(θ) GPI2(θ) = I
            if all(math.isclose([gamma, beta, alpha], [gamma])):
                continue

            # Construct the three new operations to apply
            with qml.QueuingManager.stop_recording():
                first_gate = GPI2(gamma, wires=current_gate.wires)
                second_gate = GPI(beta, wires=current_gate.wires)
                third_gate = GPI2(alpha, wires=current_gate.wires)

            gates_to_apply = [first_gate, second_gate, third_gate]
            search_and_apply_three_gate_identities(gates_to_apply)

    # Queue the measurements normally
    for m in tape.measurements:
        qml.apply(m)


@qfunc_transform
def convert_to_gpi(tape, exclude_list=[]):
    """Transpile a specific set of gates into native trapped ion gates."""
    for op in tape.operations:
        if op.name not in exclude_list and op.name in decomp_map.keys():
            if op.num_params > 0:
                decomp_map[op.name](*op.data, op.wires)
            else:
                decomp_map[op.name](op.wires)
        else:
            qml.apply(op)

    for op in tape.measurements:
        qml.apply(op)


@qfunc_transform
def ionize(tape):
    """A full set of transpilation passes to apply to convert the circuit
    into native gates and optimize it.
    """
    initial_transform_list = [
        qml.transforms.merge_rotations(),
        convert_to_gpi(exclude_list=["RZ"]),  # Leave RZ as-is, we will virtualize
        virtualize_rz_gates,
    ]

    with qml.QueuingManager.stop_recording():
        optimized_tape = tape

        for transform in initial_transform_list:
            optimized_tape = transform(optimized_tape)

        for _ in range(10):
            optimized_tape = commute_through_ms_gates(direction="left")(optimized_tape)
            optimized_tape = single_qubit_fusion_gpi(optimized_tape)

    for op in optimized_tape:
        qml.apply(op)
