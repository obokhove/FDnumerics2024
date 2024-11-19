## About
This folder contains materials related to the FEM presentation delivered by Yang on Friday 17 Nov 2023, as well as the updates made on 22 Nov 2023.
- :new: `FEM_Firedrake.pdf`: the *updated* slides where a new section is added to serve as the basis for the *new* approach regarding automated generation of weak formulations
- `Poissons_eq.py`: the associated [*Firedrake*](https://www.firedrakeproject.org/) code for solving the worked example
- `L2error.py`: the Python code for plotting the $L^2$ error versus mesh resolution (*Firedrake* not required)
- :new: `Poissons_eq_v2.py`: the *updated* code where the *new* approach is added and compared with the "old" approach. The weak formulation is generated automatically via *Firedrake*'s built-in `derivative()` method in the former, while in the latter, it was derived manually and implemented explicitly.
- :new: `L2error_v2.py`: the *updated* code for a comparison between the $L^2$ errors obtained from the two approaches.
- :new: Play with nCG (in line with CG, now 1, make it a variable nCG you set beforehand), such that nx*nCG=K and compare accuracy); look up CG-polynomials on the Firedrake website.

## How to run the *Firedrake* code from terminal?
Firedrake has been installed locally on your systems using an Apptainer (formerly Singularity) image that must be activated before you can access the Firedrake python interpreter.

1. Navigate to the Apptainer image folder from the terminal:
   ```
   cd /localhome/data/sw/firedrake-2023/
   ```
   
2. Activate the Apptainer image:
   ```
   apptainer shell -B ~/.cache:/home/firedrake/firedrake/.cache ./firedrake_latest.sif
   ```
   Now your terminal will be running inside the firedrake_latest container. You will only have access to modules that are installed inside the          container other modules including paraview will have to be run from another terminal.

3. Activate Firedrake:
   ```
   source /home/firedrake/firedrake/bin/activate
   ```

4. Navigate to the directory where you put the code (say `~/Desktop/FEM_course/`):
    ```
    cd Desktop/FEM_course/
    ```
          
5. Execute the code:
    ```
    python3 Poissons_eq.py
    ```
6. Check the results (be careful, please, when Anaconda and/or Fluent are open, these may interfere):
   - In the same directory you should find the `output.pvd` file which can be visualised using [*ParaView*](https://www.paraview.org/). Load paraview using the most recent working version:
     ```
     module load paraview-rhel8/5.11.2
     paraview
     ```
   - You can also find the mesh resolution and the corresponding $L^2$ error printed to the terminal window, e.g.
     ```
     Mesh resolution: Δx = 0.125; L2 error = 0.006213900940245384
     ```
7. Try different meshes (or order of the CG function space) and collect data. Put them into the `L2error.py` and do your own convergence analysis.

### Note
:eyes:  You may find the following warning when running the code in the *Firedrake* environment. It can be safely ignored.
        ```
        firedrake:WARNING OMP_NUM_THREADS is not set or is set to a value greater than 1, we suggest setting OMP_NUM_THREADS=1 to improve performance
        ```
        Otherwise, type `export OMP_NUM_THREADS=1` after activating *Firedrake* to bring more peace.


👀     If you want to export large volumes of data it is recommended to run your code within a personal 'not-backed-up' folder on the local machine         to avoid using your disk quota:
       ```
       mkdir /localhome/not-backed-up/yourusername/
       ```

👀     As an alternative approach to above, the image activation and code execution can be combined in a script. An example of this is provided in:
      ```
      /localhome/data/sw/firedrake-2023/proof-of-life.sh
      ```
      which can be run from within the folder using a terminal:
      ```
      ./proof-of-life.sh
      ```
      Feel free to modify the script for your own needs.
      
