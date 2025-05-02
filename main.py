import numpy as np
import tifffile
from scipy.ndimage import label, binary_erosion, center_of_mass
from scipy.spatial.distance import cdist
import pandas as pd
## load datasets


img_chrom_raw = tifffile.imread("/g/schwab/Marco/Chandni_mitodino/CellE_chromosomes.tif")
img_npc_raw = tifffile.imread("/g/schwab/Marco/Chandni_mitodino/CellE_NPCs.tif")


img_npc = img_npc_raw[0:900, 600:1500, 800:920]
img_chrom = img_chrom_raw[0:900, 600:1500, 800:920]


#tifffile.imwrite('/g/schwab/Marco/Chandni_mitodino/crop_CellE_NPCs.tif', img_npc.astype(np.uint8))
#tifffile.imwrite('/g/schwab/Marco/Chandni_mitodino/crop_CellE_chromosomes.tif', img_chrom.astype(np.uint8))

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
    print(f"Center of mass {i} of {num_features}...")
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
    print(f"Dinstance measurement {idx} of {len(source_positions)}...")
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

df_results.to_csv("/g/schwab/Marco/projects/dinomitos/distance_test.tsv", sep="\t", index=False)







