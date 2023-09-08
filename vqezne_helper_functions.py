from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, execute, IBMQ, transpile
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.tools import job_monitor

from qiskit.circuit.library import EfficientSU2
from qiskit.circuit.library import RealAmplitudes

from qiskit.algorithms.optimizers import Optimizer, SLSQP, SPSA
from qiskit.opflow.gradients import GradientBase
from qiskit.quantum_info import Pauli, Operator
from qiskit.providers.models import BackendProperties

from qiskit.providers.fake_provider import FakeMumbai
from qiskit.compiler import transpile
import qiskit

import numpy as np
from skquant.opt import minimize
from scipy.stats import pearsonr
from numpy.linalg import eigh
import matplotlib.pyplot as plt
import pickle
import copy
import csv

from VarSaw.term_grouping import *
import VarSaw.Reconstruction_Functions as RF

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Sampler
from qiskit_aer.noise import NoiseModel

import mitiq
from mitiq import zne
from mitiq.zne.inference import RichardsonFactory
from mitiq.zne.inference import ExpFactory

# Contains functions for the VQE+ZNE experiments

# Given a counts dict from VQE measurement, give the expectation for the operator
# Assumes the necessary gates were applied to the VQE circuit to convert the 
# measurement from the operator basis to the all-Z basis.
def compute_expectations(all_counts):
    '''
    Args:
    all_counts: All the counts for which we want to compute expectations
    
    Returns:
    All the expectation values
    
    '''
    all_expectation_vals = []
    for idx, count in enumerate(all_counts): 
        sum_counts = sum(list(count.values()))
        exp_val = 0
        for el in count:
            
            # allot the sign to the element
            sign = 1
            if el.count('1')%2 == 1:
                sign = -1
            
            # add to expectation value
            exp_val += sign*(count[el]/sum_counts)
        
        all_expectation_vals.append(exp_val)
            
    return all_expectation_vals

# Given a list of Pauli oeprators and the corresponding coefficients, give the reference energy
def get_ref_energy(coeffs, paulis):
    '''
    Args:
    coeffs: The coeffs of the puali tensor products
    paulis: The pauli tensors
    '''   
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
    return min_eigenval

# Returns the Paulis and Coeffs dictionary for a particular hamiltonian
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
            location = int(gate[1])
            #print('location: ', location, 'pauli_string: ', pauli_string, 'pauli: ', pauli)
            pauli_string = pauli_string[0:location] + pauli + pauli_string[location+1:]
            #print(pauli_string, len(pauli_string))
        
        paulis.append(pauli_string)
    
    return coeffs, paulis


# Prepare a virtual VQE circuit with the given parameters, and the given hamiltonian operator
def vqe_circuit(n_qubits, parameters, hamiltonian):
    '''
    Args:
    n_qubits: The number of qubits in the circuit
    parameters: The parameters for the vqe circuit
    hamiltonian: The hamiltonian string whose expectation would be measured
    using this circuit
    
    Returns:
    The VQE circuit for the given Pauli tensor hamiltonian 
    '''
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    circuit = QuantumCircuit(qr, cr)
    
    #append the circuit with the state preparation ansatz
    circuit = quantum_state_preparation(circuit, parameters)
    
    #add the measurement operations
    for i, el in enumerate(hamiltonian):
        if el == 'I':
            #no measurement for identity
            continue
        elif el == 'Z':
            circuit.measure(qr[i], cr[i])
        elif el == 'X':
            circuit.u(np.pi/2, 0, np.pi, qr[i])
            circuit.measure(qr[i], cr[i])
        elif el == 'Y':
            circuit.u(np.pi/2, 0, np.pi/2, qr[i])
            circuit.measure(qr[i], cr[i])
    
    return circuit

# Given 
# 1) a list of digital ZNE scaled ansatz (without measurement operations) which are mapped to the hardware
# 2) A Pauli operator
# 3) The virtual - to - physcial mapping for the scaled ansayz

# Apply the required gates and measurement operations so that we can measure the expectation of the given
# Pauli operator
def apply_operator(all_scaled_ansatz, pauli_op, layout_dict):
    '''
    Args:
    all_scaled_ansatz: A list of ansatz scaled at different noise factors
    pauli_op: The Pauli operator whose expectation we want to obtain
    layout_dict: The mapping from virtual to physical done on the ansatz
    '''
    ansatz_with_measurement = []
    for idx, ansatz in enumerate(all_scaled_ansatz):
        
        # apply all the operations to change basis on the ansatz
        for pauli_idx, pauli in enumerate(pauli_op):
            if pauli == 'I':
                continue
            elif pauli == 'Z':
                ansatz.measure(layout_dict[pauli_idx], pauli_idx)
            elif pauli == 'X':
                ansatz.rz(np.pi/2, layout_dict[pauli_idx])
                ansatz.sx(layout_dict[pauli_idx])
                ansatz.rz(np.pi/2, layout_dict[pauli_idx])
                ansatz.measure(layout_dict[pauli_idx], pauli_idx)
            elif pauli == 'Y':
                ansatz.sx(layout_dict[pauli_idx])
                ansatz.rz(np.pi/2, layout_dict[pauli_idx])
                ansatz.measure(layout_dict[pauli_idx], pauli_idx)
        
        # add this circuit to the list of circuits with measurements
        ansatz_with_measurement.append(ansatz)
    
    return ansatz_with_measurement

# Performs the expectation calculation step for VQE. The flag 'zne_flag' controls whether
# we report a zero-noise-extrapolated value or just a noisy value
def compute_expectations_perfect_simulation(parameters, paulis, backend, scales, nm_scaling_factor, zne_flag = False):
    '''
    Args:
    parameters: The parameters for the VQE ansatz
    paulis: Pauli strings that make up the VQE hamiltonian
    backend: The backend on which the vqe is run
    scales: The different scales at which the noise has to be scaled
    nm_scaling_factor: The scale by which we should scale the noise in the backend
    
    Returns:
    A list of expectations for each circuit
    '''
    
    global scaled_exp_dict_for_diff_iterations
    global iter_num
    
    #the number of qubits
    n_qubits = len(paulis[0])
    
    #get the ansatz
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    circuit = QuantumCircuit(qr, cr)
    ansatz = quantum_state_preparation(circuit, parameters)
    
    # transpile the ansatz -- assuming all-to-all connectivity
    backend_basis_gates = backend.configuration().basis_gates
    transpiled_ansatz = transpile(ansatz, basis_gates = backend_basis_gates, optimization_level = 3, seed_transpiler = 0)
    
    # get noise extrapolated versions of the transpiled ansatz
    # add function to record state
    all_scaled_ansatz = []
    for scale_val in scales:
        scaled_ansatz_for_scale_val = mitiq.zne.scaling.folding.fold_all(transpiled_ansatz, scale_val)
        scaled_ansatz_for_scale_val.save_state()
        all_scaled_ansatz.append(scaled_ansatz_for_scale_val)
    
    # simulate the ansatz and compute expectations using
    # noisy simulator and ideally to get the expectation
    # value
    scaled_props_dict = alter_properties_dict(backend = backend, scaling_factor = nm_scaling_factor)
    scaled_props = BackendProperties.from_dict(scaled_props_dict)
    noise_model = NoiseModel.from_backend_properties(scaled_props)
    # simulator_statevector = Aer.get_backend('aer_simulator_statevector')
    simulator_dm = Aer.get_backend('aer_simulator_density_matrix')
    
    # simulate the ansatz and get the final statevector
    #scaled_statevecs_job = simulator_statevector.run(all_scaled_ansatz, noise_model = noise_model)
    #statevec_ideal_job = simulator_statevector.run(all_scaled_ansatz[0])
    scaled_dms_job = simulator_dm.run(all_scaled_ansatz, noise_model = noise_model)
    dm_ideal_job = simulator_dm.run(all_scaled_ansatz[0])
    
    # compute expectations and store them
    ideal_expectation_vals = [] # contains the ideal expectation values for each Pauli
    estimated_expectation_vals = []
    scaled_exps_dict = {}
    for pauli_idx, pauli_op in enumerate(paulis):
        
        pauli_op_data = Operator(Pauli(pauli_op)).data
        
        # get the different scaled expectation  
        # values for the given Pauli operator
        scaled_exp_vals_for_pauli_op = []
        
        for scale_idx, scale_val in enumerate(scales):
            #relevant_statevector_data = scaled_statevecs_job.result().get_statevector(scale_idx).data
            #exp_val = np.dot(relevant_statevector_data.conjugate(), np.matmul(pauli_op_data, relevant_statevector_data))
            
            relevant_circuit = all_scaled_ansatz[scale_idx]
            relevant_dm = scaled_dms_job.result().data(relevant_circuit)['density_matrix'].data
            exp_val = np.trace(np.matmul(relevant_dm, pauli_op_data))
            
            scaled_exp_vals_for_pauli_op.append(exp_val)
        
        #ideal_statevector_data = statevec_ideal_job.result().get_statevector(0).data
        #ideal_expectation_val = np.dot(ideal_statevector_data.conjugate(), np.matmul(pauli_op_data, ideal_statevector_data))
        
        ideal_dm_data = dm_ideal_job.result().data(all_scaled_ansatz[0])['density_matrix'].data
        ideal_expectation_val = np.trace(np.matmul(ideal_dm_data, pauli_op_data))
        
        scaled_exps_dict[pauli_op] = [ideal_expectation_val] + scaled_exp_vals_for_pauli_op
        ideal_expectation_vals.append(ideal_expectation_val)
        
        # if the zne_flag is false, then we use don't compute the estimated value
        # we simply pass the ideal value as the estimated one. If the zne_flag is
        # true, we actually compute the estimated value using Zero Noise Extrapolation
        
        estimated_expectation_val = scaled_exp_vals_for_pauli_op[0]
        if zne_flag:
            extraplation_factory = ExpFactory(scale_factors = scales)
            try:
                estimated_expectation_val = extrapolation_factory.extrapolate(scale_factors = scales,
                                                                             exp_values = scaled_exp_vals_for_pauli_op)
            except:
                # if we encounter an exception, then we just use the baseline noisy value
                estimated_expectation_val = scaled_exp_vals_for_pauli_op[0]
        
        # record the estimated value
        estimated_expectation_vals.append(estimated_expectation_val.real)
    
    # save the scaled expectations dict
    scaled_exp_dict_for_diff_iterations[iter_num] = scaled_exps_dict
    iter_num += 1
    
    return estimated_expectation_vals

# Given parameters, paulis, and coefficients, the function computes the VQE loss
# Calls the 'compute_expectations_perfect_simulation' function
def compute_loss_perfect_simulation(parameters, coeffs, zne_threshold, **kwargs):
    '''
    Args:
    parameters: The parameters for the VQE ansatz
    paulis: Pauli strings that make up the VQE hamiltonian
    coeffs: Coefficients corresponding to Paulis
    backend: The backend on which the vqe is run
    scales: The different scales at which the noise has to be scaled
    nm_scaling_factor: The scale by which we should scale the noise in the backend
    
    Returns:
    The loss for the entire VQE hamiltonian
    '''
    global iter_num
    
    # if the iteration number crosses a threshold, then start applying ZNE
    if iter_num < zne_threshold:
        expectations = compute_expectations_perfect_simulation(parameters, **kwargs)
    else:
        expectations = compute_expectations_perfect_simulation(parameters, zne_flag = True, **kwargs)
    
    loss = 0
    
    for i, el in enumerate(expectations):
        loss += coeffs[i]*el
    
    return loss

# Computes the loss, and saves it in the given files. Calls the 'compute_loss_perfect_simulation' function internally
def vqe_perfect_simulation(parameters, loss_filename = None, params_filename = None, **kwargs):
    '''
    Args:
    parameters: The parameters of the VQE ansatz
    paulis: The paulis tensor hamiltonians
    coeffs: The coefficients corresponding to each pauli tensor
    backend: The backend on which the vqe is run
    mode: Specifies if we have to run a noisy simulation or ideal simulation or run the circuit on a device
    shots: The number of shots for which each circuit is executed
    
    Returns:
    Loss for one iteration of the VQE
    '''
    
    #number of qubits in the VQE ansatz
    paulis = kwargs['paulis']
    n_qubits = len(paulis[0])
    
    #making sure that the number of elements in each pauli tensor is the same
    for i in paulis:
        assert len(i) == n_qubits
    
    loss =  compute_loss_perfect_simulation(parameters, **kwargs)
    print('Loss computed by VQE is: {}'.format(loss))
    
    # save the loss and parameters
    if not (loss_filename == None):
        with open(loss_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([loss])
    
    if not(params_filename == None):
        with open(params_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(parameters)

    return loss