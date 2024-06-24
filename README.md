# Learning based Robot Navigation System in Constrained Environments

This project was worked on 'Machine Learning and Programming(MEU5053.01-00)' class project

# Summery

- Our goal is to navigate mobile robot to its destination without encountering any collisions. 
At ICRA 2024, [the BARN challenge](https://cs.gmu.edu/~xiao/Research/BARN_Challenge/BARN_Challenge24.html) has been performing navigation missions on the simulated BARN dataset.
- [LFH(Learning from Hallucination)](https://cs.gmu.edu/~xiao/papers/hallucination.pdf) is a benchmark model applied in constrained environments, but there are mismatch with real-world environments and limited generalization.
    
    → Utilize RL models to generalize of the navigation model
    
    → Utilize sampling techniques to create a training dataset that allows the robot to learn a variety of constrained environments.
    
- Simulation results showed that our model outperformed existing models in navigating from a start to a goal location quickly and without collisions.

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

git clone <https://github.com/jackal/jackal.git> --branch melodic-devel
git clone <https://github.com/jackal/jackal_simulator.git> --branch melodic-devel
git clone <https://github.com/jackal/jackal_desktop.git> --branch melodic-devel
git clone <https://github.com/utexas-bwi/eband_local_planner.git>

# Build
cd ../      # /home/dir/jackal_ws/
catkin_make

```

# Training

```bash
# To apply TD3 model training
python3 train.py --config configs/e2e_default_TD3.yaml

# To apply SAC model training
python3 train.py --config configs/e2e_default_SAC.yaml
```

- The environment dataset used for training can be found at [rl-based-navigation/map_gen/](https://github.com/msjun23/RL-based-Navigation/tree/main/map_gen)
- Simulation training was conducted on a system with the following specifications: 

> CPU: Intel(R) Xeon(R) Gold 5320 CPU @ 2.20GHz
>
> RAM: 128GB
> 
> GPU: RTX 3090 24GB
- Our training logs can be found at [rl-based-navigation/logging/](https://github.com/msjun23/RL-based-Navigation/tree/main/logging)

# **Run Simulations**

```bash
python3 run.py --world_idx 201
```

If you want to see the simulation, use --gui option when do the above command. In this step, gazebo simulation will be launched and should be connected to any display. (I.e., At the GPU server or docker envs it may not be done)

A successful run should print the episode status (collided/succeeded/timeout) and the time cost in second:

> \>>>>>>>>>>>>>>>>>> Test finished! <<<<<<<<<<<<<<<<<<
>
> Navigation collided with time 11.2460 (s)

> \>>>>>>>>>>>>>>>>>> Test finished! <<<<<<<<<<<<<<<<<<
>
> Navigation succeeded with time 13.5330 (s)


> \>>>>>>>>>>>>>>>>>> Test finished! <<<<<<<<<<<<<<<<<<
>
>Navigation timeout with time 100.0000 (s)

Once the tests finish, run `python report_test.py --out_path /path/to/out/file` to report the test

```
python report_test.py
```

You should see the report as this:

> Avg Time: 39.5296, Avg Success: 0.8500, Avg Collision: 0.0560, Avg Timeout: 0.0940
> 

# Experiments Results

- Simulation tests were conducted in a ROS Gazebo environment using 50 different BARN datasets.
- We compared the [DWA(Dynamic Window Approach)](https://github.com/Daffan/the-barn-challenge/tree/main), [E-Band(Elastic Band)](https://github.com/Daffan/the-barn-challenge/tree/eband), and [LFH(Learning from Hallucination)](https://github.com/Daffan/the-barn-challenge/tree/LfH) models

| Method | Learning | Avg Time[s] ↓ | Avg Success[%] ↑ | Avg Collision[%] ↓ | Avg Timeout[%] ↓ |
| --- | --- | --- | --- | --- | --- |
| DWA | ⨉ | 31.03 | 95.40 | 3.00 | 1.60 |
| Eband | ⨉ | 24.55 | 88.6 | 3.8380 | 7.60 |
| LfH | ○ | 11.75 | 81.8 | 15.28 | 2.92 |
| Ours(SAC) | ○ | 9.02 | 86.67 | 13.00 | 0.33 |
| Ours(TD3) | ○ | 7.60 | 95.67 | 2.30 | 2.03 |

# Map Generation

If you want to generate Map, please go to [link](https://github.com/msjun23/RL-based-Navigation/tree/main/map_gen).

# QnA

1. **What is the novelty of your approach compared to existing methods in terms of RL settings**
    
    Our approach advances upon existing methods, such as the Learning from Hallucination (LfH) algorithm, which uses neural networks to simulate constrained environments for training. While LfH addresses safety and data insufficiency by generating hallucinated constraints, it lacks time optimality, limiting its real-world applicability. In contrast, our method employs reinforcement learning (RL) algorithms like TD3 and SAC, combined with advanced sampling techniques to create a diverse training dataset. This enables our model to generalize better and achieve long-term optimization by learning from a wide variety of scenarios, thereby improving its robustness and efficiency in real-world environments. For instance, our experiments show a 25% increase in navigation speed and a 15% reduction in collision rates compared to LfH.
    
2. **Is your approach more akin to global path planning or local path planning?**
    
    Our approach is more akin to local path planning. While global path planning typically involves determining an initial path from start to goal over a large area, our method focuses on real-time decision-making and adjustments to the path based on immediate sensor inputs and environmental changes. This allows the reinforcement learning model to continuously adapt the robot’s trajectory to navigate around obstacles and respond to dynamic environments effectively. This is particularly beneficial in constrained environments where real-time adaptability and responsiveness are crucial.
    
3. **Why do you think RL works better than other methods, and to what extent?**
    
    Reinforcement learning (RL) surpasses other methods because it allows the model to learn optimal navigation strategies through interaction with the environment, receiving continuous feedback to refine its behavior. Unlike traditional algorithms that may struggle with unforeseen changes or require extensive tuning, RL models can dynamically adapt to complex and changing environments. Our simulation results demonstrate that RL models, particularly TD3 and SAC, significantly outperform traditional approaches, such as LfH, in terms of navigation speed and success rate. This enhanced performance is attributed to the RL models' ability to optimize long-term goals and adjust to various constraints, thus making them more effective in real-world applications.
    
4. **Can it work for moving obstacles?**
    
    Currently, our approach does not explicitly consider moving obstacles in the training process. Our RL model is designed to generate collision-free paths in constrained environments, focusing on static obstacles. However, we have explored the challenge of navigating dynamic obstacles using the DynaBARN framework, which involves avoiding obstacles moving in a straight line without uncertainty. In this context, our model showed successful navigation in the easy and some intermediate levels but experienced collisions in most intermediate and difficult levels. The collisions at higher difficulty levels are likely due to the increased complexity and unpredictability of obstacle movements, which our current model isn't trained to handle. Based on these findings, we plan to refine our model to enhance its robustness in dynamic environments and improve its ability to navigate through more complex scenarios. We plan future improvements to include training the model with dynamic obstacle simulations to better anticipate and react to moving obstacles.
    
5. **How can you randomize the RL training for that for robustness?**
    
    To enhance robustness, we plan to introduce a wider range of variations during the RL training process. This includes randomizing the starting points and goals for the robot, as well as varying the number, size, and speed of obstacles. Additionally, we will introduce different obstacle movement patterns and randomize environmental features. By exposing the model to a diverse set of training scenarios, we aim to improve its ability to generalize to new and unseen environments. This comprehensive approach will help ensure that the RL model can handle a variety of real-world conditions more effectively, thus increasing its overall performance in dynamic and unpredictable environments.
