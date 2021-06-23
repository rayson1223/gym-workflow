# gym-workflow
This environment is build on top of [OpenAI Gym](https://gym.openai.com/). 
The purpose of this environment is to encourage other AI researchers to be participating research on Scientific Workflow field of research.
This repository has build an abstraction layer on top of the Pegasus(Workflow Management System) with Montage generator adopted from Pegasus Team. 
So that for every step of the running cycle, it will be generating and execute workflow once. 

## Prerequisite Dependencies
Before running this environment, please install Montage library first. 
You can find the installation manual [here](http://montage.ipac.caltech.edu/docs/download2.html). 
This repository will be using Montage version 5. 

## Installation
Install the dependencies : `pip install -e .`

*Note: We had compile WorkflowSim jar file in the `./libs/workflowsim/WorkflowSim-1.0.jar`.*
 
## Code Structure
### Montage Workflow Environment Versions
Currently there are multiple versions of gym env running on different scheme regarding different rewarding mechanism 
and configurations. 
Feel free to create more alternative versions of environment and test how the agent behave under such circumstances. 
Below are the current versions that I had code so far for experiment. 


<hr/>

**Preliminary Experiments Setup**

* Montage-v1
* Montage-v2
* Montage-v3
* Montage-v4
* Montage-v5
* Montage-v6
* Montage-v7
* Montage-v8 
* Montage-v9
* Montage-v10

<hr/>

**Experiment Setup**

* Montage-v11 - Pegasus Experiment 
* Montage-v12 - WorkflowSim Experiment

<hr/>

## Contact
Feel free to drop comments or any suggestion to: raysonlcp1223@gmail.com