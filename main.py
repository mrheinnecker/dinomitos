import numpy as np
import tifffile
from scipy.ndimage import label, binary_erosion, center_of_mass
from scipy.spatial.distance import cdist
import pandas as pd
## load datasets


img_chrom_raw = tifffile.imread("/g/schwab/Marco/Chandni_mitodino/CellE_chromosomes.tif")
img_npc_raw = tifffile.imread("/g/schwab/Marco/Chandni_mitodino/CellE_NPCs.tif")


img_npc = img_npc_raw[800:900, 800:1000, 800:920]
img_chrom = img_chrom_raw[800:900, 800:1000, 800:920]


tifffile.imwrite('/g/schwab/Marco/Chandni_mitodino/crop_CellE_NPCs.tif', img_npc.astype(np.uint8))
tifffile.imwrite('/g/schwab/Marco/Chandni_mitodino/crop_CellE_chromosomes.tif', img_chrom.astype(np.uint8))

## check which voxel values are in to test
#unique, counts = np.unique(img_npc, return_counts=True)
#print("Voxel counts:", dict(zip(unique, counts)))


#unique, counts = np.unique(img_chrom, return_counts=True)
#print("Voxel counts:", dict(zip(unique, counts)))



## connected components
labeled, num_features = label(img_npc)

#unique, counts = np.unique(labeled, return_counts=True)
#print("Voxel counts:", dict(zip(unique, counts)))


output = np.zeros_like(img_npc)

## center of mass for each connected component

for i in range(1, num_features + 1):
    print(f"Processing object {i} of {num_features}...")
    com = center_of_mass(img_npc, labeled, i)
    # Round to nearest voxel coordinate
    com_voxel = tuple(map(int, np.round(com)))
    output[com_voxel] = 1




#tifffile.imwrite('/g/schwab/Marco/Chandni_mitodino/npc_centers.tif', output.astype(np.uint8))

#unique, counts = np.unique(output, return_counts=True)
#print("Voxel counts:", dict(zip(unique, counts)))


positions = np.argwhere(output == 1)

df_positions = pd.DataFrame(positions, columns=["z", "y", "x"])


source_positions = np.argwhere(output == 1)

# Get positions of target voxels (value > 0 in 'img_chrom')
target_positions = np.argwhere(img_chrom > 0)

# Prepare list to collect results
results = []

# Loop over each source voxel
for idx, source in enumerate(source_positions):
    # Compute all Euclidean distances to target voxels
    distances = np.linalg.norm(target_positions - source, axis=1)

    # Find closest voxel index and its distance
    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]
    closest_voxel = target_positions[min_idx]

    # Store result as a dictionary
    results.append({
        "id": idx,
        "source_z": source[0],
        "source_y": source[1],
        "source_x": source[2],
        "closest_dist": min_dist,
        "target_z": closest_voxel[0],
        "target_y": closest_voxel[1],
        "target_x": closest_voxel[2]
    })

# Convert to DataFrame
df_results = pd.DataFrame(results)

















target_positions = np.argwhere(img_chrom > 0)

# Loop over source positions
for i, source in enumerate(positions):
    # Compute Euclidean distances to all target voxels
    distances = np.linalg.norm(target_positions - source, axis=1)

    # Get the minimum distance
    min_distance = distances.min()

    print(f"Voxel {i} at {source} → closest chrom voxel: {min_distance:.2f}")



# Loop over source positions
for i, source in enumerate(positions):
    # Compute Euclidean distances to all target voxels
    distances = np.linalg.norm(target_positions - source, axis=1)

    # Get the minimum distance
    min_distance = distances.min()

    print(f"Voxel {i} at {source} → closest chrom voxel: {min_distance:.2f}")







# Create empty 3D volume with shape 20x20x20
vol = np.zeros((20, 20, 20), dtype=np.uint8)

# Set single voxel to 1 at position (10, 10, 10)
vol[10, 10, 10] = 1

# Set a 4x4x4 cube to value 2, starting at (5, 5, 5)
vol[5:9, 5:9, 5:9] = 2

# Sanity check: count voxel values
unique, counts = np.unique(vol, return_counts=True)
print("Voxel counts:", dict(zip(unique, counts)))



from scipy.spatial.distance import cdist

# Reference point
ref_point = np.array([[10, 10, 10]])

# Get coordinates of all voxels with value == 2
coords_obj = np.argwhere(vol == 2)

# Compute all Euclidean distances
dists = cdist(ref_point, coords_obj)

# Get min distance and location
min_dist = dists.min()
closest_voxel = coords_obj[dists.argmin()]

print(f"Minimum distance: {min_dist}")
print(f"Closest voxel with value 2: {closest_voxel}")

import tifffile

img_chrom = tifffile.imread("/g/schwab/Marco/Chandni_mitodino/CellE_chromosomes.tif")
img_npc = tifffile.imread("/g/schwab/Marco/Chandni_mitodino/CellE_NPCs.tif")

## check how many vocels have value=1

np.count_nonzero(img_npc == 1)

from scipy.ndimage import label, binary_erosion, center_of_mass

labeled, num_features = label(img_npc)

# Prepare empty output
output = np.zeros_like(img_npc)

# Structuring element (3x3x3 cube)
structure = np.ones((3,3,3), dtype=bool)


for i in range(1, num_features + 1):
    print(i)
    coords = np.argwhere(labeled == i)
    if coords.size > 0:
        random_voxel = coords[np.random.choice(len(coords))]
        output[tuple(random_voxel)] = 1









for i in range(1, num_features + 1):
    print(f"Processing object {i} of {num_features}...")
    com = center_of_mass(img_npc, labeled, i)
    # Round to nearest voxel coordinate
    com_voxel = tuple(map(int, np.round(com)))
    output[com_voxel] = 1



# Step 2: Loop over each object and erode it until 1 voxel remains
for i in range(1, num_features + 1):

    print(f"Processing object {i} of {num_features}...")

    obj = (labeled == i)
    eroded = obj.copy()

    while eroded.sum() > 1:
        next_eroded = binary_erosion(eroded, structure=structure)
        # Stop if it would be completely erased
        if next_eroded.sum() == 0:
            break
        eroded = next_eroded

    # Add the final voxel to the output
    output[eroded] = 1


unique, counts = img_npc.unique(img_npc, return_counts=True)
voxel_stats = dict(zip(unique, counts))
print(voxel_stats)





img_npc.max()
img_chrom.max()

# Find coords of point (e.g. [10,10,10])
ref_point = np.array([[10, 10, 10]])

# Get coordinates of all voxels == 2
coords_obj = np.argwhere(img == 2)

# Compute distances
dists = cdist(ref_point, coords_obj)
min_dist = dists.min()
closest_voxel = coords_obj[dists.argmin()]

print(f"Min dist: {min_dist}, Closest voxel: {closest_voxel}")










