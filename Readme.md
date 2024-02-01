
To run the code create a conda environment by using the env.yml file.
Small test examples can be run in the given Jupyter Notebooks.
These include a simple fitting test of a damped sinusoidal wave, a solver comparison with scipy odeint, and 2-body trajectory predictions.

The smaller problems were tested with the dense Cholesky. Set the Cholesky solver in config.py

## Discovery

For the Lorenz system ODE discovery run

```
PYTHONPATH=. python discovery/lorenz_ind.py
```

Plots and logs are saved in the `logs` directory. 

The discovery proceeds in a cycle of optimization and thresholding the basis weights.
The optimization runs for 400 epochs, which is a safe value but takes a few hours. A smaller value could possibly work.
The discovered ODE is fairly good after the first optimization step.

## PDE
For the 1D KdV model, get the 50 second data using the scripts at https://github.com/brandstetter-johannes/LPSDA
Modify the data location in pde/kdv.py if needed.
Edit config.py and uncomment the lines for the conjugate gradient solver.

Run using
```
PYTHONPATH=. python pde/kdv.py
```

After training test rmse can be calculated by using the losses function in kdv.py. Pass the Pytorch checkpoint.

## N-body ephemerides

Get data using the astroquery script in the nbody directory.
Edit the data_ephermris.py file to set the data path if needed.
Set the conjugate gradient solver.

Run using
```
PYTHONPATH=. python pde/ephemeris_sys.py
```

Test trajectories can be generated using generate function by passing a checkpoint.