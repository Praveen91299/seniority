def categorize_csf_state(csf_state, verbose=False):
    """
    Categorize CSF state into 4 types and extract spin-orbital indices:
    1. |HF> - Hartree-Fock (single determinant with coefficient 1.0)
    2. E_{ia}|HF> - Single excitation (two determinants with coefficients ±1/√2)
    3. E_{ia}E_{jb}|HF> - Double excitation (multiple determinants with various coefficients)
    4. Fourth type - Complex multi-determinant states
    
    Args:
        csf_state: CSF state in format [basis_states, indices, coefficients]
        
    Returns:
        tuple: (type_number, i, j, a, b) where type_number is 1-4 and i,j,a,b are spin-orbital indices
    """
    basis_states = csf_state[0]
    coefficients = csf_state[2]
    
    n_determinants = len(coefficients)
    
    # Type 1: |HF> - Single determinant with coefficient 1.0
    if n_determinants == 1 and abs(coefficients[0] - 1.0) < 1e-10:
        return (1, None, None, None, None)
    
    # Type 2: E_{ia}|HF> - Two determinants (regardless of coefficient pattern)
    elif n_determinants == 2:
        # Extract spin-orbital indices from the two determinants
        det1, det2 = basis_states[0], basis_states[1]
        # First check if it's a single excitation
        i_single, a_single = find_single_excitation_indices(det1, det2)
        if i_single is not None and a_single is not None:
            # It's a single excitation, return as (i, None, a, None)
            return (2, i_single, None, a_single, None)
        else:
            # It's a double excitation
            i, j, a, b = find_double_excitation_indices(det1, det2)
            if verbose:
                print(f"Double excitation indices found: i={i}, j={j}, a={a}, b={b}")
            return (2, i // 2, j // 2, a // 2, b // 2)
    
    # Type 3: E_{ia}E_{jb}|HF> - Multiple determinants with specific patterns
    elif n_determinants == 4:
        # Check for 4-determinant pattern with coefficients ±0.5
        coeffs_abs = [abs(c) for c in coefficients]
        if all(abs(c - 0.5) < 1e-6 for c in coeffs_abs):
            # Extract i, j, a, b from the 4 determinants
            i, j, a, b = find_double_excitation_indices_4det(basis_states)
            return (3, i, j, a, b)
        else:
            return (4, None, None, None, None)
    
    # Type 4: Complex multi-determinant states (6 determinants with complex coefficients)
    elif n_determinants > 4:
        # Check for 6-determinant pattern with coefficients involving √3
        # Main coefficients should be ±1/√3 (≈ ±0.57735027) and ±1/(2√3) (≈ ±0.28867513)
        # coeffs_abs = [abs(c) for c in coefficients]
        # has_main_coeff = any(abs(c - 0.57735027) < 1e-6 for c in coeffs_abs)
        # has_secondary_coeff = any(abs(c - 0.28867513) < 1e-6 for c in coeffs_abs)
        
        # if has_main_coeff and has_secondary_coeff:
        #     return (4, None, None, None, None)
        # else:
        # This is actually a double excitation with 6 determinants
        i, j, a, b = find_double_excitation_indices_6det(basis_states)
        return (4, i, j, a, b)
    
    else:
        return (0, None, None, None, None)  # Unknown type


def find_single_excitation_indices(det1, det2):
    """
    Find the occupied (i) and virtual (a) spin-orbital indices for a single excitation.
    
    Args:
        det1, det2: Two determinants (arrays of 0s and 1s)
        
    Returns:
        tuple: (i, a) where i is occupied and a is virtual
    """
    # Find where the determinants differ
    diff = det1 - det2
    
    # Find the occupied orbital (i) - where det1 has 1 and det2 has 0
    i_indices = [idx for idx, val in enumerate(diff) if val > 0.5]
    
    # Find the virtual orbital (a) - where det1 has 0 and det2 has 1
    a_indices = [idx for idx, val in enumerate(diff) if val < -0.5]
    
    if len(i_indices) == 1 and len(a_indices) == 1:
        return i_indices[0], a_indices[0]
    else:
        # This is actually a double excitation, not single
        return None, None


def find_double_excitation_indices(det1, det2):
    """
    Find the occupied (i,j) and virtual (a,b) spin-orbital indices for a double excitation.
    
    Args:
        det1, det2: Two determinants (arrays of 0s and 1s)
        
    Returns:
        tuple: (i, j, a, b) where i,j are occupied and a,b are virtual
    """
    # Find where the determinants differ
    diff = det1 - det2
    
    # Find the occupied orbitals (i,j) - where det1 has 1 and det2 has 0
    i_indices = [idx for idx, val in enumerate(diff) if val > 0.5]
    
    # Find the virtual orbitals (a,b) - where det1 has 0 and det2 has 1
    a_indices = [idx for idx, val in enumerate(diff) if val < -0.5]

    
    if len(i_indices) == 2 and len(a_indices) == 2:
        return i_indices[0], i_indices[1], a_indices[0], a_indices[1]
    else:
        return None, None, None, None


def find_double_excitation_indices_4det(basis_states):
    """
    Find the occupied (i,j) and virtual (a,b) spin-orbital indices for a double excitation
    from 4 determinants.

    For 4-determinant double excitations, find spin-orbitals that are NOT identical
    across all 4 determinants. These will be 2i, 2i+1, 2j, 2j+1, 2a, 2a+1, 2b, 2b+1.

    Args:
        basis_states: List of 4 determinants

    Returns:
        tuple: (i, j, a, b) where i,j are occupied and a,b are virtual (spatial orbital indices)
    """
    if len(basis_states) != 4:
        return None, None, None, None

    n_spin_orbitals = len(basis_states[0])
    changing_orbitals = []

    # Find spin-orbitals that are not identical across all 4 determinants
    for spin_idx in range(n_spin_orbitals):
        values = [det[spin_idx] for det in basis_states]
        if len(set(values)) > 1:  # Not all identical
            changing_orbitals.append(spin_idx)

    # Could have 4 or 8 changing spin-orbitals depending on whether both spins are involved
    if len(changing_orbitals) >= 4:
        # Group by spatial orbital (even/odd pairs)
        spatial_orbitals = {}
        for spin_idx in changing_orbitals:
            spatial_idx = spin_idx // 2
            if spatial_idx not in spatial_orbitals:
                spatial_orbitals[spatial_idx] = []
            spatial_orbitals[spatial_idx].append(spin_idx)

        # Should have at least 2 spatial orbitals involved
        if len(spatial_orbitals) >= 2:
            spatial_indices = sorted(spatial_orbitals.keys())

            # Use first determinant to determine occupied vs virtual
            det0 = basis_states[0]
            occupied_spatial = []
            virtual_spatial = []

            for spatial_idx in spatial_indices:
                # Check if this spatial orbital is occupied or virtual in det0
                # by checking any of its spin-orbitals
                spin_orbitals_for_spatial = spatial_orbitals[spatial_idx]
                # Check the first spin-orbital for this spatial orbital
                first_spin_idx = spin_orbitals_for_spatial[0]
                if det0[first_spin_idx] > 0.5:  # Occupied
                    occupied_spatial.append(spatial_idx)
                else:  # Virtual
                    virtual_spatial.append(spatial_idx)

            if len(occupied_spatial) >= 1 and len(virtual_spatial) >= 1:
                # Return the indices, padding with None if needed
                i = occupied_spatial[0] if len(occupied_spatial) > 0 else None
                j = occupied_spatial[1] if len(occupied_spatial) > 1 else None
                a = virtual_spatial[0] if len(virtual_spatial) > 0 else None
                b = virtual_spatial[1] if len(virtual_spatial) > 1 else None
                return i, j, a, b

    return None, None, None, None


def find_double_excitation_indices_6det(basis_states):
    """
    Find the occupied (i,j) and virtual (a,b) spatial orbital indices for a double excitation
    from 6 determinants.

    Args:
        basis_states: List of 6 determinants

    Returns:
        tuple: (i, j, a, b) where i,j are occupied and a,b are virtual (spatial orbital indices)
    """
    # For 6-determinant double excitations, analyze the pattern
    det0 = basis_states[0]

    # Find all unique spin-orbitals that change across all determinants
    all_changing_spin_orbitals = set()
    for det in basis_states[1:]:
        diff = det0 - det
        changing = [idx for idx, val in enumerate(diff) if abs(val) > 0.5]
        all_changing_spin_orbitals.update(changing)

    # Convert spin-orbitals to spatial orbitals
    changing_spatial_orbitals = set()
    for spin_idx in all_changing_spin_orbitals:
        spatial_idx = spin_idx // 2
        changing_spatial_orbitals.add(spatial_idx)

    changing_spatial_orbitals = sorted(list(changing_spatial_orbitals))

    # For a double excitation, we expect 4 changing spatial orbitals (i, j, a, b)
    if len(changing_spatial_orbitals) == 4:
        # Determine which are occupied in det0 (i,j) and which are virtual (a,b)
        # Check using the spin-up orbital (index 2*spatial_idx) of each spatial orbital
        occupied = []
        virtual = []

        for spatial_idx in changing_spatial_orbitals:
            spin_up_idx = 2 * spatial_idx
            if det0[spin_up_idx] > 0.5:  # Occupied
                occupied.append(spatial_idx)
            else:  # Virtual
                virtual.append(spatial_idx)

        if len(occupied) == 2 and len(virtual) == 2:
            return occupied[0], occupied[1], virtual[0], virtual[1]

    return None, None, None, None


def analyze_all_csf_states(list_list_refCSF):
    """
    Analyze all CSF states and categorize them.
    
    Args:
        list_list_refCSF: List of lists of CSF states
        
    Returns:
        dict: Summary of categories
    """
    type_names = {1: "|HF>", 2: "E_{ia}|HF>", 3: "E_{ia}E_{jb}|HF>", 4: "Fourth type", 0: "Unknown type"}
    categories = {1: 0, 2: 0, 3: 0, 4: 0, 0: 0}
    
    print("\n" + "="*80)
    print("CSF STATE CATEGORIZATION ANALYSIS")
    print("="*80)
    
    state_idx = 0
    for group_idx, csf_group in enumerate(list_list_refCSF):
        print(f"\nGroup {group_idx}:")
        for csf_idx, csf_state in enumerate(csf_group):
            type_num, i, j, a, b = categorize_csf_state(csf_state, verbose=True)
            categories[type_num] += 1
            
            basis_states = csf_state[0]
            coefficients = csf_state[2]
            
            print(f"  State {state_idx}: Type {type_num} - {type_names[type_num]}")
            print(f"    Determinants: {len(basis_states)}")
            print(f"    Coefficients: {coefficients}")
            
            # Print spin-orbital and spatial orbital indices if they exist
            if type_num == 2:  # Two determinants
                if i is not None and j is not None and a is not None and b is not None:
                    # Convert spin-orbital indices to spatial orbital indices
                    i_spatial = i // 2
                    j_spatial = j // 2
                    a_spatial = a // 2
                    b_spatial = b // 2
                    print(f"    Spin-orbitals: i={i}, j={j}, a={a}, b={b}")
                    print(f"    Spatial orbitals: i={i_spatial}, j={j_spatial}, a={a_spatial}, b={b_spatial}")
                else:
                    print(f"    Spin-orbitals: Could not extract (complex pattern)")
            elif type_num == 3:  # Four determinants
                if i is not None and j is not None and a is not None and b is not None:
                    # Convert spin-orbital indices to spatial orbital indices
                    i_spatial = i // 2
                    j_spatial = j // 2
                    a_spatial = a // 2
                    b_spatial = b // 2
                    print(f"    Spin-orbitals: i={i}, j={j}, a={a}, b={b}")
                    print(f"    Spatial orbitals: i={i_spatial}, j={j_spatial}, a={a_spatial}, b={b_spatial}")
                else:
                    print(f"    Spin-orbitals: Could not extract (complex pattern)")
            
            state_idx += 1
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for type_num, count in categories.items():
        if count > 0:
            print(f"Type {type_num} - {type_names[type_num]}: {count} states")
    
    return categories

