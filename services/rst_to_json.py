import numpy as np
import tempfile
from ansys.mapdl.reader import read_binary


def convert_rst_to_json(rst_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".rst") as temp_rst:
        rst_file.save(temp_rst.name)
        rst_path = temp_rst.name

    rst = read_binary(rst_path)

    # Natural frequencies
    try:
        natural_freqs = rst.time_values.tolist()
    except Exception:
        natural_freqs = []

    mode_node_ids = []
    mode_magnitudes = []

    # Extract mode shapes
    for i in range(rst.n_results):
        disp_tuple = rst.nodal_displacement(i)
        node_ids = disp_tuple[0].astype(int)
        disp_vectors = disp_tuple[1]
        mag = np.linalg.norm(disp_vectors, axis=1)

        eps = 1e-5
        mag = np.clip(mag, eps, None)
        mode_node_ids.append(node_ids)
        mode_magnitudes.append(mag)

    # Build master node list
    master_nodes = sorted(set().union(*[set(ids) for ids in mode_node_ids]))

    # Normalize mode shapes to same node order
    normalized_modes = []
    for ids, mags in zip(mode_node_ids, mode_magnitudes):
        map_dict = dict(zip(ids, mags))
        vect = [float(map_dict.get(node, 0.0)) for node in master_nodes]
        normalized_modes.append(vect)

    # ---- Extract node positions (FIXED HERE) ----
    # rst.mesh.nnum = array of node numbers
    # rst.mesh.nodes = coordinates in same order
    nnum = rst.mesh.nnum
    coords = rst.mesh.nodes

    # Build reverse lookup: node number → row index
    node_index = {int(num): idx for idx, num in enumerate(nnum)}

    node_positions = []
    for node in master_nodes:
        idx = node_index.get(node, None)
        if idx is not None:
            pos = coords[idx]
            node_positions.append([int(node), float(pos[0]), float(pos[1]), float(pos[2])])
        else:
            node_positions.append([int(node), 0.0, 0.0, 0.0])

    return {"PrimarySystemNaturalFrequencies": natural_freqs, "PrimarySystemModes": normalized_modes, "PrimarySystemNodePositions": node_positions}
