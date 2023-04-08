from itertools import product
import pickle

import numpy as np
import pennylane as qml
import pennylane.math as math

from pennylane.ops import GPI, GPI2

DOUBLE_IDENTITY_FILENAME = "double_gate_identities.pkl"
TRIPLE_IDENTITY_FILENAME = "triple_gate_identities.pkl"


def _test_matrix_equivalence(mat1, mat2):
    """Checks the equivalence of two unitary matrices."""
    mat_product = math.dot(mat1, math.conj(math.T(mat2)))

    if math.isclose(mat_product[0, 0], 0.0):
        mat_product = mat_product / mat_product[0, 1]
    else:
        mat_product = mat_product / mat_product[0, 0]

    if math.allclose(mat_product, math.eye(mat_product.shape[0])):
        return True

    return False


def generate_gate_identities():
    """Generates all 2- and 3-gate identities involving GPI/GPI2 and special angles.

    Results are stored in pkl files which can be used later on.
    """

    id_angles = [
        -np.pi,
        -3 * np.pi / 4,
        -np.pi / 2,
        -np.pi / 4,
        0.0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4,
        np.pi,
    ]

    single_gates = {
        "GPI": [([angle], GPI.compute_matrix(angle)) for angle in id_angles],
        "GPI2": [([angle], GPI2.compute_matrix(angle)) for angle in id_angles],
    }

    double_gate_identities = {}

    # Check which combinations of 2 gates reduces to a single one
    for g1, g2 in product([GPI, GPI2], repeat=2):
        combo_name = g1.__name__ + g2.__name__

        for a1, a2 in product(id_angles, repeat=2):
            matrix = math.dot(g1.compute_matrix(a1), g2.compute_matrix(a2))

            # Test in case we produced the identity;
            if not math.isclose(matrix[0, 0], 0.0):
                if math.allclose(matrix / matrix[0, 0], math.eye(2)):
                    if combo_name not in double_gate_identities.keys():
                        double_gate_identities[combo_name] = []
                    double_gate_identities[combo_name].append(([a1, a2], "Identity", 0.0))
                    continue

            for id_gate in single_gates.keys():
                angles, matrices = [x[0] for x in single_gates[id_gate]], [
                    x[1] for x in single_gates[id_gate]
                ]

                for ref_angle, ref_matrix in zip(angles, matrices):
                    if _test_matrix_equivalence(matrix, ref_matrix):
                        if combo_name not in double_gate_identities.keys():
                            double_gate_identities[combo_name] = []

                        if not any(
                            np.allclose([a1, a2], database_angles)
                            for database_angles in [
                                identity[0] for identity in double_gate_identities[combo_name]
                            ]
                        ):
                            double_gate_identities[combo_name].append(
                                ([a1, a2], id_gate, ref_angle)
                            )

    with open(DOUBLE_IDENTITY_FILENAME, "wb") as outfile:
        pickle.dump(double_gate_identities, outfile)

    triple_gate_identities = {}

    # Check which combinations of 2 gates reduces to a single one
    for g1, g2, g3 in product([GPI, GPI2], repeat=3):
        combo_name = g1.__name__ + g2.__name__ + g3.__name__

        for a1, a2, a3 in product(id_angles, repeat=3):
            matrix = math.linalg.multi_dot(
                [g1.compute_matrix(a1), g2.compute_matrix(a2), g3.compute_matrix(a3)]
            )

            # Test in case we produced the identity;
            if not math.isclose(matrix[0, 0], 0.0):
                if math.allclose(matrix / matrix[0, 0], math.eye(2)):
                    if combo_name not in triple_gate_identities.keys():
                        triple_gate_identities[combo_name] = []
                    triple_gate_identities[combo_name].append(([a1, a2, a3], "Identity", 0.0))
                    continue

            for id_gate in single_gates.keys():
                angles, matrices = [x[0] for x in single_gates[id_gate]], [
                    x[1] for x in single_gates[id_gate]
                ]

                for ref_angle, ref_matrix in zip(angles, matrices):
                    if _test_matrix_equivalence(matrix, ref_matrix):
                        if combo_name not in triple_gate_identities.keys():
                            triple_gate_identities[combo_name] = []

                        if not any(
                            np.allclose([a1, a2, a3], database_angles)
                            for database_angles in [
                                identity[0] for identity in triple_gate_identities[combo_name]
                            ]
                        ):
                            triple_gate_identities[combo_name].append(
                                ([a1, a2, a3], id_gate, ref_angle)
                            )

    with open(TRIPLE_IDENTITY_FILENAME, "wb") as outfile:
        pickle.dump(triple_gate_identities, outfile)


def lookup_gate_identity(gates):
    """Given a pair of input gates in the order they come in the circuit,
    look up if there is a circuit identity in our database. Note that the
    database is constructed using matrix multiplication so we will need to
    exchange the order of the gates."""

    if len(gates) == 2:
        try:
            with open(DOUBLE_IDENTITY_FILENAME, "rb") as infile:
                gate_identities = pickle.load(infile)
        except:
            # Generate the file first and then load it
            generate_gate_identities()
            with open(DOUBLE_IDENTITY_FILENAME, "rb") as infile:
                gate_identities = pickle.load(infile)
    elif len(gates) == 3:
        try:
            with open(TRIPLE_IDENTITY_FILENAME, "rb") as infile:
                gate_identities = pickle.load(infile)
        except:
            # Generate the file first and then load it
            generate_gate_identities()
            with open(TRIPLE_IDENTITY_FILENAME, "rb") as infile:
                gate_identities = pickle.load(infile)

    # Get the information about this particular combination of gates
    combo_name = "".join([gate.name for gate in gates[::-1]])
    combo_angles = [float(gate.data[0]) for gate in gates[::-1]]

    combo_db = gate_identities[combo_name]
    all_angle_combos = [combo[0] for combo in combo_db]
    angle_check = [np.allclose(combo_angles, test_angles) for test_angles in all_angle_combos]

    if any(angle_check):
        idx = np.where(angle_check)[0][0]
        new_gate_name = combo_db[idx][1]
        new_gate_angle = combo_db[idx][2]

        if new_gate_name == "GPI":
            return [GPI(*new_gate_angle, wires=gates[0].wires)]
        elif new_gate_name == "GPI2":
            return [GPI2(*new_gate_angle, wires=gates[0].wires)]
        elif new_gate_name == "Identity":
            return []

    return None
