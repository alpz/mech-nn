
To run the code create a conda environment by using the env.yml file.
Examples can be run in the given Jupyter Notebooks.

For the Lorenz system ODE discovery run

```
PYTHONPATH=. python discovery/lorenz_ind.py
```

Plots and logs are saved in the `logs` directory. 

The discovery proceeds in a cycle of optimization and thresholding the basis weights.
The optimization runs for 400 epochs, which is a safe value but takes a few hours. A smaller value could possibly work.
The discovered ODE is fairly good after the first optimization step.