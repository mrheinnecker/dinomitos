import numpy as np
import tifffile
from scipy.ndimage import label, binary_erosion, center_of_mass
from scipy.spatial.distance import cdist
import pandas as pd
## load datasets


img_chrom_raw = tifffile.imread("/g/schwab/Marco/Chandni_mitodino/CellE_chromosomes.tif")
img_npc_raw = tifffile.imread("/g/schwab/Marco/Chandni_mitodino/CellE_NPCs.tif")


img_npc = img_npc_raw#[0:900, 600:1500, 800:920]
img_chrom = img_chrom_raw#[0:900, 600:1500, 800:920]


#tifffile.imwrite('/g/schwab/Marco/Chandni_mitodino/crop_CellE_NPCs.tif', img_npc.astype(np.uint8))
#tifffile.imwrite('/g/schwab/Marco/Chandni_mitodino/crop_CellE_chromosomes.tif', img_chrom.astype(np.uint8))

## check which voxel values are in to test
#unique, counts = np.unique(img_npc, return_counts=True)
#print("Voxel counts:", dict(zip(unique, counts)))


#unique, counts = np.unique(img_chrom, return_counts=True)
#print("Voxel counts:", dict(zip(unique, counts)))



## connected components
labeled, num_features = label(img_npc)






#object_ids = np.unique(labeled)
object_ids = np.arange(1,num_features+1)
#object_ids = object_ids[object_ids != 0]  # remove background (label 0)

com_results = []

for obj_id in object_ids:
    print(f"Center of mass {obj_id} of {max(object_ids)}...")
    # Get all voxel coordinates for this object
    coords = np.argwhere(labeled == obj_id)

    # Get bounding box around the object
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0) + 1  # +1 because slicing is exclusive

    # Crop a small subvolume around the object
    subvol = labeled[zmin:zmax, ymin:ymax, xmin:xmax]

    # Create a binary mask within this subvolume (where subvol == obj_id)
    mask = (subvol == obj_id)

    # Compute center of mass in the cropped subvolume
    local_com = center_of_mass(mask)

    # Translate local CoM back to global image coordinates
    global_com = (local_com[0] + zmin, local_com[1] + ymin, local_com[2] + xmin)

    # Store result
    com_results.append({
        "object_id": int(obj_id),
        "z": global_com[0],
        "y": global_com[1],
        "x": global_com[2]
    })


df_com = pd.DataFrame(com_results)


dist_results = []

for idx, row in df_com.iterrows():
    print(f"Distance measurement {idx + 1} of {len(df_com)}...")

    # Source position as array
    source = np.array([row["z"], row["y"], row["x"]])

    # Compute all distances to target positions
    distances = np.linalg.norm(target_positions - source, axis=1)

    # Find closest voxel
    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]
    closest_voxel = target_positions[min_idx]

    # Store result
    dist_results.append({
        "id": int(row["object_id"]),
        "source_z": source[0],
        "source_y": source[1],
        "source_x": source[2],
        "closest_dist": min_dist,
        "target_z": closest_voxel[0],
        "target_y": closest_voxel[1],
        "target_x": closest_voxel[2]
    })

# Create result DataFrame
df_results = pd.DataFrame(dist_results)



df_results.to_csv("/g/schwab/Marco/projects/dinomitos/distance_test.tsv", sep="\t", index=False)



