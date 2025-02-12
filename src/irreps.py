from copy import copy

import numpy as np
import streamlit as st
import sympy as sp


def initialize_generators(D, symbols_list):
    """
    Initializes a set of generator matrices using symbolic variables.

    Parameters:
    -----------
    D : int
        The dimension of the generator matrices
        (each matrix will be of size DxD).

    symbols_list : list
        A list of symbolic variables that will be used to populate the
        matrices.It should have at least `2 * D^2` elements to construct the
        required matrices.

    Returns:
    --------
    generators : list of sympy.Matrix
        A list containing two DxD symbolic matrices.

    Notes:
    ------
    - The function sequentially populates each matrix with symbols from
    `symbols_list`.
    - The input `symbols_list` should have sufficient elements; otherwise,
    `pop()` will raise an `IndexError`.
    - Each generator is stored as a `sympy.Matrix` with complex entries.
    """

    generators = []
    symbols_copy = copy(symbols_list)  # Generates x0, x1, ..., x_{d^3-1})
    for _ in range(1, 3):
        rows = []
        for _ in range(D):
            cols = []
            for _ in range(D):
                cols.append(symbols_copy.pop(0))
            rows.append(cols)
        generators.append(sp.Matrix(rows, type=complex))
    return generators


def initialize_eigenstates(eigenstates):
    """
    Converts a given set of eigenstates into symbolic SymPy matrices.

    Parameters:
    -----------
    eigenstates : np.ndarray
        A NumPy array where each **column** represents an eigenstate.
        The shape should be (D, N), where:
        - `D` is the dimension of each eigenstate vector.
        - `N` is the number of eigenstates.

    Returns:
    --------
    symbolic_eigenstates : list of sympy.Matrix
        A list containing symbolic representations of the eigenstates as
        `sympy.Matrix` objects.
        Each matrix is treated as a complex-valued column vector.

    Notes:
    ------
    - Each eigenstate (a column in `eigenstates`) is converted into a `sympy.Matrix`
    of size `(D, 1)`.Assumes that `eigenstates` is well-formed and contains
    numeric values before conversion.
    - The function does **not** modify the original `eigenstates` array.
    """

    symbolic_eigenstates = []
    for j in range(eigenstates.shape[1]):
        symbolic_eigenstate = sp.Matrix(eigenstates[:, j], type=complex)
        symbolic_eigenstates.append(symbolic_eigenstate)
    return symbolic_eigenstates


# def form_matrix_irrep_so3(j:int) -> list[sp.Matrix]:

#     """

#     """
#     j = 1/2
#     D = int(2*j+1)
#     js = np.arange(j, -j-1, -1,dtype=complex)
#     J3 = sp.Matrix(np.diag(js))

#     # form eigenvectors of J3
#     eigenstates = np.eye(D, dtype=complex)

#     # form variables in the generators
#     num_vars = D**2*2  # Number of variables
#     # Generate symbols as a flattened list
#     symbols_list = [*sp.symbols(f'x:{num_vars}')]  # Generates x0, x1, ..., x_{d^3-1}
#     generators = initialize_generators(D,symbols_list)
#     symbolic_eigenstates = initialize_eigenstates(eigenstates)

#     # iterate over all matrice to be constructed
#     for i in range(D):
#         if i == 0:
#             prefactor = 2.0
#             # iterate over all columns, or rather pairs of columns

#             col_sol1 = sp.solve(prefactor*generators[i]*symbolic_eigenstates[0]-symbolic_eigenstates[1] ,symbols_list[:4],dict=True)
#             col_sol2 = sp.solve(prefactor*generators[i]*symbolic_eigenstates[1]-symbolic_eigenstates[0] ,symbols_list[:4],dict=True)
#             print(list(col_sol1[0].values()))
#             mat = [ list(col_sol1[0].values()),list(col_sol2[0].values())]
#             generators[i] = sp.Matrix(mat).T
#         else :
#             prefactor = 2.0j
#             # iterate over all columns, or rather pairs of columns
#             col_sol1 = sp.solve(prefactor*generators[i]*symbolic_eigenstates[0]+symbolic_eigenstates[1] ,symbols_list[4:],dict=True)
#             col_sol2 = sp.solve(prefactor*generators[i]*symbolic_eigenstates[1]-symbolic_eigenstates[0] ,symbols_list[4:],dict=True)
#             mat = [ list(col_sol1[0].values()),list(col_sol2[0].values())]
#             generators[i] = sp.Matrix(mat).T

#     return [*generators,J3]


def reassemble_mat_from_sols(D: int, sols: list[dict]) -> sp.Matrix:
    """
    Reconstructs a DxD SymPy matrix from a list of solution dictionaries.

    Parameters:
    -----------
    D : int
        The dimension of the square matrix (D x D).

    sols : list of dict
        A list where each dictionary represents a partial solution with
        symbolic keys (e.g., x0, x1, ...) and their corresponding values.

    Returns:
    --------
    sp.Matrix
        A SymPy matrix of size (D x D) with values sorted according to
        their symbolic variable indices.

    Notes:
    ------
    - The function merges all dictionaries in `sols` into one.
    - It extracts and sorts values based on the numerical index of their
      symbolic keys.
    - The values are then used to construct a D x D matrix.
    - Assumes the number of extracted values matches `D*D`; otherwise, an
    error occurs.

    Example:
    --------
    >>> sols = [
    ...     {sp.Symbol('x0'): 0.5, sp.Symbol('x1'): 0.0},
    ...     {sp.Symbol('x2'): 1.0, sp.Symbol('x3'): -0.5}
    ... ]
    >>> reassemble_mat_from_sols(2, sols)
    Matrix([
    [0.5, 0.0],
    [1.0, -0.5]
    ])
    """

    merged_dict = {key: value for d in sols for key, value in d.items()}
    sorted_dict = sorted(merged_dict.items(), key=lambda item: int(str(item[0])[1:]))
    sorted_vals = [val for (_, val) in sorted_dict]
    return sp.Matrix(D, D, sorted_vals)


def form_matrix_irrep_so3(j: int) -> list[sp.Matrix]:
    """
    Constructs the SO(3) irreducible representation (irrep) matrices for a
      given spin value j.

    Parameters:
    -----------
    j : int
        The total quantum number (j). It determines the
        dimension of the representation as D = 2j + 1.

    Returns:
    --------
    list[sp.Matrix]
        A list containing the SO(3) generators in matrix form: [J1, J2, J3].
        - J1 (J_x): The first generator of the SO(3) algebra.
        - J2 (J_y): The second generator of the SO(3) algebra.
        - J3 (J_z): The third generator of the SO(3) algebra (diagonal form).

    Notes:
    ------
    - The function first constructs the eigenstates of J3.
    - It initializes the generator matrices using symbolic variables.
    - It solves for the matrix elements by imposing the SO(3) commutation
    relations.
    - The solution is reassembled into matrices using
    `reassemble_mat_from_sols`.
    - The final output consists of the three operators J1, J2, and J3.

    Example:
    --------
    >>> import sympy as sp
    >>> J_matrices = form_matrix_irrep_so3(1)
    >>> for J in J_matrices:
    ...     sp.pprint(J)

    This will return the DxD matrices representing the SO(3) Lie algebra
    generators for any j in the standard representation.
    """

    D = int(2 * j + 1)
    js = np.arange(j, -j - 1, -1, dtype=complex)
    J3 = sp.Matrix(np.diag(js))

    # form eigenvectors of J3
    eigenstates = np.eye(D, dtype=complex)

    # form variables in the generators
    num_vars = D**2 * 2  # Number of variables
    # Generate symbols as a flattened list
    symbols_list = [*sp.symbols(f"x:{num_vars}")]  # Generates x0, x1, ..., x_{d^3-1}
    generators = initialize_generators(D, symbols_list)
    symbolic_eigenstates = initialize_eigenstates(eigenstates)
    null_ket = sp.Matrix(np.zeros(D, dtype=complex))

    for i, prefactor in enumerate([2.0, 2.0j]):
        # iterate over all columns, or rather pairs of columns
        sols = []
        for k, m in enumerate(js):
            # eig = symbolic_eigenstates[k]
            m_plus1_state, m_minus1_state = null_ket, null_ket
            # Check bounds before accessing elements
            # Ensure `i - 1` is valid before accessing it
            if k >= 0 and m + 1 <= j:
                prefactor_up = (j * (j + 1) - m * (m + 1)) ** (1 / 2)
                m_plus1_state = prefactor_up * symbolic_eigenstates[k - 1]

            # Ensure `i + 1` is valid before accessing it
            if k <= len(js) - 1 and m - 1 >= -j:
                prefactor_down = (j * (j + 1) - m * (m - 1)) ** (1 / 2)
                m_minus1_state = prefactor_down * symbolic_eigenstates[k + 1]

            rhs = (
                -m_plus1_state - m_minus1_state
                if prefactor == 2.0
                else -m_plus1_state + m_minus1_state
            )
            symbols = (
                symbols_list[: num_vars // 2]
                if prefactor == 2.0
                else symbols_list[num_vars // 2 :]
            )
            sol = sp.solve(
                prefactor * generators[i] * symbolic_eigenstates[k] + rhs,
                symbols,
                dict=True,
            )[0]
            sols.append(sol)
        J = reassemble_mat_from_sols(D, sols)
        generators[i] = J

    return generators + [J3]


# ------------- STREAMLIT UI -------------
st.title("SO(3) Irrep Matrix Generator")

# Input field
j_value = st.number_input(
    "Enter j value (integer or half-integer)", min_value=0.5, step=0.5
)

if st.button("Generate Matrices"):
    try:
        matrices = form_matrix_irrep_so3(j_value)

        st.write("### Generated Matrices:")
        for idx, matrix in enumerate(matrices):
            st.latex(f"J_{{{idx+1}}} = {sp.latex(matrix)}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
