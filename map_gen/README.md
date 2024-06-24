
# Map Generation

```bash
cd map_gen
python LfSH_mapgen.py
```

## Basic Algorithm

We utilized the mechanism used to generate the existing BARN Dataset.

1. Obtain information about the Global Path and the World from the existing BARN Dataset.
   - Note that it is also possible to use the path and world information from a customized dataset instead of the BARN Dataset.

2. Expand the space by a user-specified parameter perpendicular to the direction of the Global Path, 
   assuming the remaining space as obstacles to create a narrow passage. 
   (This is similar to the hallucination technique of the existing LfH).

<div align="center">
   <img src="https://github.com/msjun23/RL-based-Navigation/assets/97781279/e81b9f8e-d248-46fd-ab03-09aa4e3a52f4" alt="image" width="300" height="300">
</div>

3. Further sampling is performed within the narrow passage,
   creating a more constrained environment compared to the existing LfH.
<div align="center">
   <img src="https://github.com/msjun23/RL-based-Navigation/assets/47807421/1b3a5dda-445f-4fd2-959e-67b31cbdaef2" alt="create_obstacle_samples" width="300" height="300">
</div>

4. If there are environments where the mobile robot cannot pass,
   the free space is expanded around the sampled obstacles to make the environment more suitable for training.

<div align="center">
   <img src="https://github.com/msjun23/RL-based-Navigation/assets/47807421/7f3775cd-d7ea-452c-b29c-9b425bbb02d2" alt="expand_free_space" width="300" height="300">
</div>

## Additional Considerations

1. **File Path Settings**
   - `world_file = 'train_data/world_files/world_%d.world' % iteration`
   - `path_file  = 'test_data/path_files/path_%d.npy'      % iteration`
   
   Here, `path_file` refers to the paths to the reference data,
     and `world_file` is the path to the folder you want to generate.

2. **Parameters**
   - `free_space`: The amount of free space to expand from the sampled obstacles.
   - `rand_fill_pct`: The degree to which obstacles occupy free space.
   - `smooth_iter`: The number of iterations for the obstacle generation algorithm.
     - The more iterations, the more concentrated the obstacles become, and smaller obstacles disappear.
   - `fill_threshold`: The minimum number of neighboring obstacles required to convert the current cell into an obstacle.
     - The smaller the value, the easier it is to create obstacles.
   - `clear_threshold`: The maximum number of neighboring obstacles required to convert the current cell into free space.
     - The larger the value, the easier it is to create free space.
