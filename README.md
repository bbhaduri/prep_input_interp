# prep_input_interp
This repository replicates the task from [Schimel et. al. 2024](https://elifesciences.org/articles/89131#s3) using JAX and the [diffilqrax](https://github.com/ThomasMullen/diffilqrax/tree/main) repository created by Thomas Mullen. Original code for this task was written by Marine Schimel and can be found [here](https://github.com/marineschimel/why-prep-2).

### Installation
Clone this repository and its submodules using the following comand:
```
git clone --recursive https://github.com/bbhaduri/prep_input_interp.git
```

Create a conda environment in order to run any of the notebooks with Python 3.10
and activate it.
```
conda create -n <insert env_name> python=3.10
# After creating the environment
conda activate <insert env_name>
```

Add src folder and `diffilqrax` libraries to your environment:
```
# from prep_input_interp directory
pip install -e .

# Go to submodule
cd libs/diffilqrax-m1arm
pip install -e .
```

After doing the steps above, you will be able to run any of the folders in the `notebooks` folder.
