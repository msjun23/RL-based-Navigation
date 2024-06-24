# RL-based-Navigation
Reinforcement Learning based Robot Navigation at Constrained Environment: team project for MEU5053.01-00

# Environment setup
- We recommand to use conda env

- To train the model, [Singularity](https://docs.sylabs.io/guides/latest/admin-guide/installation.html#installation-on-linux) must be installed.

- To run inference (gazebo simulation), you need to install the [ROS melodic](https://wiki.ros.org/melodic/Installation/Ubuntu).

```bash
# Create conda venv
conda create -n jackal python=3.9
conda activate jackal

# Get required packages
mkdir -p jackal_ws/src
cd jackal_ws/src

git clone https://github.com/jackal/jackal.git --branch melodic-devel
git clone https://github.com/jackal/jackal_simulator.git --branch melodic-devel
git clone https://github.com/jackal/jackal_desktop.git --branch melodic-devel
git clone https://github.com/utexas-bwi/eband_local_planner.git

# Build
cd ../      # /home/dir/jackal_ws/
catkin_make
```

# Train
```bash
python3 train.py --config configs/e2e_default_TD3.yaml
```

# Inference
```bash
python3 run.py --world_idx 201
```
If you want to see the simulation, use --gui option when do the above command. In this step, gazebo simulation will be launched and should be connected to any display. (I.e., At the GPU server or docker envs it may not be done)

# Experiments Results


# QnA
1. What is the novelty of your approach compared to existing methods in terms of RL settings?

    A. Actually, we don't have the novelty in RL approach. We just use existing SoTA RL algorithms TD3 and SAC. But at the mobile robotics domain, we show and analyze how TD3 and SAC work with constrained environment. Also, we empirically imply that TD3 is more suitable than SAC in constrained environment.

2. Is your approach more akin to global path planning or local path planning?

    A. Our method is about learning local path planner. Global path planning is performed through rule based manner.

3. Why do you think RL works better than other methods, and to what extent?

    A. We set the LfH as our baseline and it learns local path planner in supervised manner. Empirically LfH shows lower performance. However, our RL based method achieves best performance.

4. Can it work for moving obstacles.

    A. Since it is based on RL, it seems likely to work with moving obstacles as well, but this needs to be tested. In the case of moving obstacles, the ground truth is not clear, so models trained in a supervised manner like LfH are expected to perform worse, and our model is expected to show an advantage.

5. How can you randomize the RL training for that for robustness?

    A. 
