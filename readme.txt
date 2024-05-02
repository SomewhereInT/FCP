The document describes a team programming assignment aimed at simulating opinion dynamics through two different computational models, which include the implementation and testing of the Ising model and the Defuant model.


Github repo link: https://github.com/SomewhereInT/FCP


Dependencies

The following Python libraries are required to run the code:

- NumPy
- Matplotlib
- Argparse

You can install these dependencies using pip:

pip install numpy matplotlib argparse


How to Run:
1.	Ensure you have Python 3.x installed on your system. 
2.	Install the required dependencies by running the following command: pip install numpy matplotlib 3. 
3.	Navigate to the project directory and run the main script with the desired command-line arguments:  python3 assignment.py [options]

The option can be instead of the following flags:

Ising Model: 
- `-ising_model`: Run the Ising model with default parameters. 
- `-use_network N`: Use a network with N nodes for the Ising model simulation. 
- `-alpha [value]`: Choose an alpha value for societal tolerance (default: 1.0). 
- `-external [value]`: Choose a value of the external influence factor (default: 0.0). 
- `-test_ising`: Run test functions for the Ising model calculations.

Network Models: 
- `-network [value]`: Create and plot a random network with [value] nodes. Calculate the value of mean degree, mean clustering coefficient and mean path length.
- `-test_network`: Run test functions for network models.
- `-ring_network [value]`: Create and plot a ring network with [value] nodes. 
- `-small_world [value]`: Create and plot a small-world network with [value] nodes. 


Deffuant Model: - `-defuant`: Run the Deffuant model simulation. 
- `-beta [value]`: Choose the coupling parameter for the Deffuant model (default: 0.2). 
- `-threshold [value]`: Choose the opinion difference threshold for the Deffuant model (default: 0.2). 
- `-test_defuant`: Run test functions for the Deffuant model.


Example Usage

1. This should run the ising model with default parameters:
>>python3 assignment.py -ising_model

2. Run the Ising model simulation with an external influence factor of -0.1:
>>python3 assignment.py -ising_model -external -0.1

3. Run the Ising model simulation with no external influence but with a societal tolerance factor of 10:
>>python3 assignment.py -ising_model -alpha 10

4. Run the test functions associated with the model:
>>python3 assignment.py -test_ising

5. To visualize a ring network with 10 nodes, type the following in the terminal:
>>python3 assignment.py -ring_network 10

6. To visualize a small-world network with 20 nodes and a rewiring probability of 0.3, type the following in the terminal:
>>python3 assignment.py -small_world 20 -re_wire 0.3

7. Run defuant model with default parameters (beta = 0.2 and threshold = 0.2): 
>>python3 assignment.py -defuant 

8. Run defuant model with custom coupling parameter (logically valid range of beta: 0.0 ~ 0.5):
>>python3 assignment.py -defuant -beta 0.1

9. Run defuant model with custom threshold parameter:
>>python3 assignment.py -defuant -threshold 0.3

10. Run defuant model with custom coupling and threshold parameters:
>>python3 assignment.py -defuant -beta 0.1 -threshold 0.3

11. Run the test cases of defuant model:
>>python3 assignment.py -test_defuant

13. Run defuant model test case with interactive mode:
>>python3 assignment.py test_defuant()

14. Solve the ising model on a small world network of size 10.
>> python3 assignment.py -ising_model -use_network 10

15.This should run the random network with 10 nodes and provide the value of mean degree, mean clustering coefficient and mean path length:
>>python3 assignment.py -network 10

16.Run test functions for network models:
>>python3 assignment.py -test_network





