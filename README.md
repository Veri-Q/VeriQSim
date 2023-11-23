# ApproximateQSIM: An Approximate Noisy Quantum Circuit Simulator

## Requirements ##

- [Python3.9.12](https://www.python.org/).
- Python libraries: 
    * [Tensornetwork](https://github.com/google/tensornetwork) for manipulating tensor networks.
    * [Numpy](https://numpy.org/) for linear algebra computations.
    * [Qiskit](https://qiskit.org/) for manipulating quantum circuits.

## Installation (for Linux) ##

We recommend the users to use [Conda](https://docs.conda.io/en/latest/) to configure the Python environment.

### Install with Conda (Miniconda) ###
1. Follow the instructions of [Miniconda Installation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install Miniconda.
2. Clone this repository and cd to it.
    ```bash
    git clone https://github.com/hmy98213/ApproximateQSIM.git && cd ApproximateQSIM
    ```
3. Use Conda to create a new Conda environment:
    ```bash
    conda create -n ApproximateQSIM python=3.9.12
    ```
4. Activate the above environment and use pip to install required libraries in `requirements.txt`.
    ```bash
    conda activate ApproximateQSIM
    pip install -r requirements.txt
    ```

## Noisy Quantum Circuit Simulation ##
Running the following command to simulate noisy quantum circuits:

```python
from my_cpu import *
tn.set_default_backend("pytorch")
folder_enum_test('<Benchmark_Folder>', '<Output_File>', error_num, enum)
```
- Benchmark_Folder: The folder containing the benchmark circuits.
- Output_File: The file to store the simulation results.
- error_num: The number of errors to be injected into the circuit.
- enum: The approximate level of approximate simulation.


