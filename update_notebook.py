import json
import sys

def modify_notebook(file_path):
    with open(file_path, 'r') as f:
        nb = json.load(f)

    new_source = [
        "# Extract ordered centerline points using MST (Minimum Spanning Tree) to eliminate branches.\n",
        "# This mathematically guarantees a strictly sequential, cycle-free, branch-free path!\n",
        "from scipy.sparse import csr_matrix\n",
        "from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path\n",
        "\n",
        "def extract_centerline_iterative(start_x, start_y):\n",
        "    \"\"\"\n",
        "    Extracts the longest path from the skeleton using MST.\n",
        "    This naturally eliminates all branches and handles loops perfectly\n",
        "    by breaking them at a single edge and returning the ordered sequence.\n",
        "    \"\"\"\n",
        "    ys, xs = np.where(centerline_dist > 0)\n",
        "    if len(xs) == 0:\n",
        "        return [], []\n",
        "        \n",
        "    points = list(zip(xs, ys))\n",
        "    point_to_idx = {pt: i for i, pt in enumerate(points)}\n",
        "    \n",
        "    row, col, data = [], [], []\n",
        "    for i, (x, y) in enumerate(points):\n",
        "        for dx in [-1, 0, 1]:\n",
        "            for dy in [-1, 0, 1]:\n",
        "                if dx == 0 and dy == 0:\n",
        "                    continue\n",
        "                nx, ny = x + dx, y + dy\n",
        "                if (nx, ny) in point_to_idx:\n",
        "                    j = point_to_idx[(nx, ny)]\n",
        "                    if i < j:  # undirected edges\n",
        "                        dist = np.hypot(dx, dy)\n",
        "                        row.append(i)\n",
        "                        col.append(j)\n",
        "                        data.append(dist)\n",
        "                        \n",
        "    # create symmetric adjacency matrix\n",
        "    row_sym = row + col\n",
        "    col_sym = col + row\n",
        "    data_sym = data + data\n",
        "    adj = csr_matrix((data_sym, (row_sym, col_sym)), shape=(len(points), len(points)))\n",
        "    \n",
        "    # 1. Minimum Spanning Tree\n",
        "    mst = minimum_spanning_tree(adj)\n",
        "    \n",
        "    # mst is a directed graph (upper triangular), make it undirected for shortest_path\n",
        "    mst_sym = mst.maximum(mst.T)\n",
        "    \n",
        "    # 2. Farthest node from an arbitrary node (node 0)\n",
        "    dist_matrix, predecessors = shortest_path(csgraph=mst_sym, directed=False, indices=0, return_predecessors=True)\n",
        "    dist_matrix[np.isinf(dist_matrix)] = -1\n",
        "    start_node = np.argmax(dist_matrix)\n",
        "    \n",
        "    # 3. Farthest node from start_node\n",
        "    dist_matrix, predecessors = shortest_path(csgraph=mst_sym, directed=False, indices=start_node, return_predecessors=True)\n",
        "    dist_matrix[np.isinf(dist_matrix)] = -1\n",
        "    end_node = np.argmax(dist_matrix)\n",
        "    \n",
        "    # 4. Reconstruct path from start_node to end_node\n",
        "    path_indices = []\n",
        "    curr = end_node\n",
        "    while curr != start_node:\n",
        "        if curr < 0:\n",
        "            break\n",
        "        path_indices.append(curr)\n",
        "        curr = predecessors[curr]\n",
        "    path_indices.append(start_node)\n",
        "    path_indices.reverse()\n",
        "    \n",
        "    # 5. Build points and widths arrays\n",
        "    ordered_points = []\n",
        "    ordered_widths = []\n",
        "    for idx in path_indices:\n",
        "        x, y = points[idx]\n",
        "        ordered_points.append(np.array([x, y]))\n",
        "        ordered_widths.append(np.array([centerline_dist[y, x], centerline_dist[y, x]]))\n",
        "        \n",
        "    return ordered_points, ordered_widths\n",
        "\n",
        "centerline_points, track_widths = extract_centerline_iterative(left_start_x, left_start_y)\n",
        "print(f\"Extracted {len(centerline_points)} centerline points\")\n",
        "\n",
        "if len(centerline_points) < 50:\n",
        "    print(\"WARNING: Very few centerline points extracted. Check THRESHOLD or map image quality.\")\n"
    ]

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "def extract_centerline_iterative" in source:
                cell['source'] = new_source
                print("Successfully updated the cell.")
                break

    with open(file_path, 'w') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    modify_notebook("/home/sedrica/Raceline-Optimization/map_converter.ipynb")
