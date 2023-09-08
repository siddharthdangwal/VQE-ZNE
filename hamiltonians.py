import numpy as np
from numpy.linalg import eigh

from qiskit.quantum_info import Pauli, Operator
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, QubitConverter

from numbers import Number

from VarSaw.term_grouping import *


def give_paulis_and_coeffs(hamiltonian, num_qubits):
    '''
    hamiltonian: A list containing all hamiltonian terms along with their weights
    num_qubits: The number of qubits in the hamiltonian
    '''
    paulis = []
    coeffs = []
   
    for idx, term in enumerate(hamiltonian):
       
        #the coefficient
        coeffs.append(term[0])
       
        #the pauli string
        pauli_string = num_qubits*'I'
        all_gates = term[1]
        #print(non_id_gates)
       
        for _, gate in enumerate(all_gates):
            pauli = gate[0]
            location = int(gate[1:])
            #print('location: ', location, 'pauli_string: ', pauli_string, 'pauli: ', pauli)
            pauli_string = pauli_string[0:location] + pauli + pauli_string[location+1:]
            #print(pauli_string, len(pauli_string))
       
        paulis.append(pauli_string)
   
    return coeffs, paulis

def hamiltonian_from_file(file, **kwargs):
    hamil = parseHamiltonian(file)
    max_length = 0
    for i in hamil:
        if int(i[1][-1][1]) + 1 > max_length:
            max_length = int(i[1][-1][1]) + 1
    n_qubits = max_length
    coeffs, paulis = give_paulis_and_coeffs(hamil, n_qubits)
    #HF_bitstring, energy_best = find_best_bitstring(coeffs, paulis)
    HF_bitstring, energy_best = "0"*n_qubits, "-"
    print(f"best bitstring {HF_bitstring}, with energy {energy_best}")
    return coeffs, paulis, HF_bitstring

def get_ref_energy(coeffs, paulis, return_groundstate=False):
    """
    Compute theoretical minimum energy.
    coeffs (Iterable[Float]): Pauli coefficients in Hamiltonian.
    paulis (Iterable[String]): Corresponding Pauli strings in Hamiltonian (same order as coeffs).
    return_groundstate (Bool): Whether to return groundstate.
    
    Returns:
    (Float) minimum energy (optionally also groundstate as array).
    """
    # the final operation
    final_op = None

    for ii, el in enumerate(paulis):
        if ii == 0:
            final_op = coeffs[ii]*Operator(Pauli(el))
        else:
            final_op += coeffs[ii]*Operator(Pauli(el))
   
    # compute the eigenvalues
    evals, evecs = eigh(final_op.data)
   
    # get the minimum eigenvalue
    min_eigenval = np.min(evals)
    if return_groundstate:
        return min_eigenval, evecs[:,0]
    else:
        return min_eigenval

def all1_hamiltonian(n_qubits=1, **kwargs):
    coeffs = np.array([1.]*n_qubits)
    paulis = []
    for i in range(n_qubits):
        paulis.append("I"*i + "Z" + "I"*(n_qubits-i-1))
    paulis = np.array(paulis)
    HF_bitstring = "1"*n_qubits
    return coeffs, paulis, HF_bitstring

def molecule(atom_string, new_num_orbitals=None, **kwargs):
    """
    Compute Hamiltonian for molecule in qubit encoding using Qiskit Nature.
    atom_string (String): string to describe molecule, passed to PySCFDriver.
    new_num_orbitals (Int): Number of orbitals in active space (if None, use default result from PySCFDriver).
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.

    Returns:
    (Iterable[Float], Iterable[String], String) (Pauli coefficients, Pauli strings, Hartree-Fock bitstring)
    """
    converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
    driver = PySCFDriver(
        atom=atom_string,
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )
    problem = driver.run()
    if new_num_orbitals is not None:
        num_electrons = (problem.num_alpha, problem.num_beta)
        transformer = ActiveSpaceTransformer(num_electrons, new_num_orbitals)
        problem = transformer.transform(problem)
    ferOp = problem.hamiltonian.second_q_op()
    qubitOp = converter.convert(ferOp, problem.num_particles)
    initial_state = HartreeFock(
        problem.num_spatial_orbitals,
        problem.num_particles,
        converter
    )
    bitstring = "".join(["1" if bit else "0" for bit in initial_state._bitstr])
    # need to reverse order bc of qiskit endianness
    paulis = [x[::-1] for x in qubitOp.primitive.paulis.to_labels()]
    # add the shift as extra I pauli
    paulis.append("I"*len(paulis[0]))
    paulis = np.array(paulis)
    coeffs = list(qubitOp.primitive.coeffs)
    # add the shift (nuclear repulsion)
    coeffs.append(problem.nuclear_repulsion_energy)
    coeffs = np.array(coeffs).real
    return coeffs, paulis, bitstring

def H6_hexagon(r):
    parts = [f"H {r*np.cos(n*np.pi/3):.4f} {r*np.sin(n*np.pi/3):.4f} 0" for n in range(6)]
    return "; ".join(parts)

def H6_linear(r):
    return "; ".join([f"H 0 0 {n*r}" for n in range(6)])

def H2O_linear(r):
    return f"H 0 0 0; O 0 0 {r}; H 0 0 {2*r}"

def ising_model_linear(N, Jx, h, periodic=False, HF_bitstring=None):
    """
    Constructs qubit Hamiltonian for linear Ising model.
    H = sum_{i=0...N-2} (Jx_i X_i X_{i+1} + sum_{i=0...N-1}  h_i Z_i

    N (Int): # sites/qubits.
    Jx (Float, Iterable[Float]): XX strength, either constant value or list (values for each pair of neighboring sites).
    h (Float, Iterable[Float]): Z self-energy, either constant value or list (values for each site).
    periodic: If periodic boundary conditions. If True, include term X_0 X_{N-1}.

    Returns:
    (Iterable[Float], Iterable[String], String) (Pauli coefficients, Pauli strings, "0"*N)
    """
    if isinstance(Jx, Number):
        if periodic:
            Jx = [Jx] * N
        else:
            Jx = [Jx] * (N-1)
    if isinstance(h, Number):
        h = [h] * N        
    if N > 1:
        assert len(Jx) == N if periodic else len(Jx) == N-1, "Jx has wrong length"
        assert len(h) == N, "h has wrong length"
    edges_weights = {(i, i+1): Jx[i] for i in range(N-1) if np.abs(Jx[i]) > 1e-12}
    if periodic and N > 2 and np.abs(Jx[N-1]) > 1e-12:
        edges_weights[(0, N-1)] = Jx[N-1]
    transverse_weights = {i: h[i] for i in range(N) if np.abs(h[i]) > 1e-12}
    return ising_model(edges_weights, transverse_weights, HF_bitstring)

def ising_model(edges_weights, transverse_weights, HF_bitstring=None):
    n_qubits = 0
    for e in edges_weights:
        n_qubits = max(n_qubits, max(e))
    for v in transverse_weights:
        n_qubits = max(n_qubits, v)
    n_qubits += 1
    coeffs = []
    paulis = []
    for e, w in edges_weights.items():
        s = "I"*e[0] + "X" + "I"*(e[1] - e[0] - 1) + "X" + "I"*(n_qubits - e[1] - 1)
        coeffs.append(w)
        paulis.append(s)
    for v, w in transverse_weights.items():
        s = "I"*v + "Z" + "I"*(n_qubits - v - 1)
        coeffs.append(w)
        paulis.append(s)
    if HF_bitstring is None:
        HF_bitstring = "0"*n_qubits
    return coeffs, paulis, HF_bitstring

def heisenberg_model(edges_weights, HF_bitstring=None):
    allowed_keys = ["X", "Y", "Z"]
    n_qubits = 0
    for e in edges_weights:
        n_qubits = max(n_qubits, max(e)) + 1
    coeffs = []
    paulis = []
    for e, key_weight in edges_weights.items():
        for key, w in key_weight.items():
            assert key in allowed_keys
            s = "I"*e[0] + key + "I"*(e[1] - e[0] - 1) + key + "I"*(n_qubits - e[1] - 1)
            coeffs.append(w)
            paulis.append(s)
    if HF_bitstring is None:
        HF_bitstring = "0"*n_qubits
    return coeffs, paulis, HF_bitstring

def XXZ_model_linear(N, Jxy, Jz, HF_bitstring=None):
    edges_weights = {
        (i, i+1): {
            "X": Jxy,
            "Y": Jxy,
            "Z": Jz
            } 
        for i in range(N-1)}
    return heisenberg_model(edges_weights, HF_bitstring)

def energy_of_bitstring(bs, coeffs, paulis):
    import re
    relevant_idx = []
    for i, P in enumerate(paulis):
        P = P.upper()
        if "X" in P or "Y" in P:
            continue
        else:
            relevant_idx.append(i)
    assert len(bs) == len(paulis[0])
    one_idx = [m.start() for m in re.finditer("1", bs)]
    energy = 0
    for i in relevant_idx:
        c = coeffs[i]
        P = paulis[i].upper()
        sign = 1
        for o_idx in one_idx:
            if P[o_idx] == "Z":
                sign *= -1
        energy += sign * c
    return energy

def find_best_bitstring(coeffs, paulis):
    import re
    relevant_idx = []
    for i, P in enumerate(paulis):
        P = P.upper()
        if "X" in P or "Y" in P:
            continue
        else:
            relevant_idx.append(i)
    n_qubits = len(paulis[0])
    energy_best = np.inf
    bs_best = ""
    for x in range(2**n_qubits):
        bs = bin(x)[2:].zfill(n_qubits)
        one_idx = [m.start() for m in re.finditer("1", bs)]
        energy = 0
        for i in relevant_idx:
            c = coeffs[i]
            P = paulis[i].upper()
            sign = 1
            for o_idx in one_idx:
                if P[o_idx] == "Z":
                    sign *= -1
            energy += sign * c
        if energy < energy_best:
            energy_best = energy
            bs_best = bs
    return bs_best, energy_best