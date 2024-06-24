
# Map Generation

'''bash
cd map_gen
run LfSH_mapgen.py
'''

## Basic Algorithm

We utilized the mechanism used to generate the existing Barn Dataset.

1. Obtain information about the Global Path and the World from the existing Barn Dataset.
   - Note that it is also possible to use the path and world information from a customized dataset instead of the Barn Dataset.

2. Expand the space by the user-specified parameter perpendicular to the direction of the Global Path, assuming the remaining space as obstacles to create a narrow passage. (This is similar to the Hallucination technique of the existing LfH).

3. Further sampling is performed within the narrow passage, creating a more constrained environment compared to the existing LfH.

4. If there are environments where the mobile robot cannot pass, the free space is expanded around the sampled obstacles to make the environment more suitable for training.

## Additional Considerations

1. **File Path Settings**
   - `world_file = 'train_data/world_files/world_%d.world' % iteration`
   - `path_file = 'test_data/path_files/path_%d.npy' % iteration`
   - `grid_file = 'test_data/grid_files/grid_%d.npy' % iteration`
   
   Here, `path_file` and `grid_file` are the paths to the reference data, and `world_file` is the path to the folder you want to generate.

2. **Parameters**
   - `free_space`: The amount of free space to expand from the sampled obstacles.
   - `rand_fill_pct`: The degree to which obstacles occupy free space.
   - `smooth_iter`: The number of iterations for the obstacle generation algorithm.
     - The more iterations, the more concentrated the obstacles become, and smaller obstacles disappear.
   - `fill_threshold`: The minimum number of neighboring obstacles required to convert the current cell into an obstacle.
     - The smaller the value, the easier it is to create obstacles.
   - `clear_threshold`: The maximum number of neighboring obstacles required to convert the current cell into free space.
     - The larger the value, the easier it is to create free space.
