# TANDEMGAN
This is the code of the following paper.

"Chi Hong, Jiyue Huang, Robert Birke, and Lydia Y. Chen. "Exploring and Exploiting Data-Free Model Stealing." In Joint European Conference on Machine Learning and
Knowledge Discovery in Databases, pp. 20-35., 2023."

An example of running the algorithm is shown in "run.py".

# To run this file
The project is developed under python 3.8.10. If you use a newer version of Python, some dependency packages may fail to install.

- pip install -r requirements.txt
- python run.py

# Note
- ms_ee_attack.MSEEAttack is the class of the proposed algorithm.

# Expected Results
- You will see the target model accuracy is 93.09 %
- The accuracy of the substitute model will increaces with training over multiple epochs
- one training log is shown in "./example_run.log"
