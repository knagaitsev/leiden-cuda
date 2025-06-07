import cudf
import cugraph

# Step 1: Create the edge list as a cuDF DataFrame
edges = cudf.DataFrame({
    'src': [0, 1, 2, 3, 4, 5, 6, 7],
    'dst': [1, 2, 0, 4, 5, 6, 7, 4],
    'weight': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]
})

# Step 2: Create a cuGraph Graph and load the edges
G = cugraph.Graph()
G.from_cudf_edgelist(edges, source='src', destination='dst', edge_attr='weight')

# Step 3: Run the Leiden algorithm
partitions = cugraph.leiden(G)

# Step 4: Print community assignment
print(partitions)
