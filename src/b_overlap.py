def get_boundary_qubits(layout, coupling_map):
    """Get the boundary qubits of the layout."""

    coupling_map.make_symmetric()
    boundary_qubits = []
    for q1 in layout:
        neighbor_qubits = coupling_map.neighbors(q1)
        print(f'neighbour qubits = {[q1, [qubit for qubit in neighbor_qubits]]}')
        for q2 in neighbor_qubits:
            if q2 not in layout and q1 not in boundary_qubits:
                boundary_qubits.append(q1)
    return boundary_qubits 

def check_b_overlap(layout1, layout2, coupling_map, buffer_distance):
    """Check b-overlap between two layouts  """
    
    set1 = set(layout1)
    set2 = set(layout2)
    
    # Check for intersection
    print(len(set1 & set2))
    if len(set1 & set2) > 0:
        return True
    
    if(len(set1 & set2) == 0):
        print([set1, set2])
    
    distance = 0
    boundary1 = get_boundary_qubits(layout1, coupling_map)
    boundary2 = get_boundary_qubits(layout2, coupling_map)
    print([boundary1, boundary2])
    for q1 in boundary1:
        for q2 in boundary2:
            distance = coupling_map.distance(q1, q2)
            print([q1, q2, distance])
            if distance <= buffer_distance:
                return True
    return False