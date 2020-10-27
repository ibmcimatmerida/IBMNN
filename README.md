# IBMNN


Please cite the paper as: "An immersed boundary neural network for solving elliptic equations with singular forces on arbitrary domains (IBMNN)"

 

Requirements

·         Python 3.6

·         TensorFlow 1.14

·         MatLab/Octave

 

The code is organized as follows:

·         input: folder which contains the input data in format .mat from MatLab

·         output: folder which contains two MatLab functions to plot the results

·         Example1_A.py: Solve the Laplace equation in arbitrary domains with smooth solution

·         Example1_R.py: Solve the Laplace equation in rectangular domains with smooth solution

·         Example2_R.py: Solve the Laplace equation with mixed boundary conditions (Dirichlet and Neumann) in rectangular domains

·         Example3_R.py: Solve the elliptic equation with variable coefficients in rectangular domains

·         Example4_A.py: Solve the Poisson equation with a circular interface in arbitrary domains

·         Example4_R.py Solve the Poisson equation with a circular interface in rectangular domains

·         Example5_R.py: Solve the Poisson equation with a singular source term in rectangular domains

 

To run the code, open a command window, find the folder that contains the source code and call the python programs as follows:

 

>> python3 Example#_op.py

where:

·         #: a value between [1,5]

·         op: option A for arbitrary domains and option R for rectangular domains

 

Note: If your system has a GPU, then the programs are executed in the GPU automatically, otherwise the programs are executed in the CPU. If you want to force the execution only in the CPU, you should run the programs as follows:

>> CUDA_VISIBLE_DEVICES=-1 python3 Example#_op.py

There may be slight differences between the results, running the programs with GPU or without GPU due to the rounding errors in the floating point arithmetic. 

