def get_boundary_qubits(layout, coupling_map):
    """Get the boundary qubits of the layout."""

    coupling_map.make_symmetric()
    boundary_qubits = []
    for q1 in layout:
        neighbor_qubits = coupling_map.neighbors(q1)
        for q2 in neighbor_qubits:
            if q2 not in layout and q1 not in boundary_qubits:
                boundary_qubits.append(q1)
    return boundary_qubits 

def get_connected_qubits(layout1, layout2, module1_qubits, module2_qubits):
    """Get the connected qubits between two layouts."""
    connected_qubits = []

    for idx1, q1 in enumerate(layout1):
        var = module1_qubits[idx1]

        if var in module2_qubits:
            idx2 = module2_qubits.index(var)
            q2 = layout2[idx2]

            connected_qubits.append((q1, q2))

    return connected_qubits

# def check_b_overlap(layout1, layout2, coupling_map, buffer_distance, module1_qubits, module2_qubits):
#     """Check b-overlap between two layouts and return the maximum I/O distance."""
    
#     set1 = set(layout1)
#     set2 = set(layout2)
    
#     # Check for intersection
#     if len(set1 & set2) > 0:
#         return True, 0
    
#     distance = 0
#     max_io_distance = 0
#     overlap = False
#     boundary1 = get_boundary_qubits(layout1, coupling_map)
#     boundary2 = get_boundary_qubits(layout2, coupling_map)
#     for q1 in boundary1:
#         for q2 in boundary2:
#             distance = coupling_map.distance(q1, q2)
#             if distance <= buffer_distance:
#                 overlap = True

#     connected_qubits = get_connected_qubits(layout1=layout1, layout2=layout2, module1_qubits=module1_qubits, module2_qubits=module2_qubits)
#     for (q1, q2) in connected_qubits:
#         distance = coupling_map.distance(q1, q2)
#         if distance > max_io_distance:
#             max_io_distance = distance

#     return overlap, max_io_distance

def check_b_overlap(layout1, layout2, coupling_map, buffer_distance):
    """Check b-overlap between two layouts. """
    
    set1 = set(layout1)
    set2 = set(layout2)
    
    # Check for intersection
    if len(set1 & set2) > 0:
        return True

    distance = 0
    boundary1 = get_boundary_qubits(layout1, coupling_map)
    boundary2 = get_boundary_qubits(layout2, coupling_map)
    for q1 in boundary1:
        for q2 in boundary2:
            distance = coupling_map.distance(q1, q2)
            if distance <= buffer_distance:
                return True
    return False