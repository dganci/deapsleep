# DeapSleep
DeapSleep is a DEAP-based evolutionary computation toolkit for testing dropout in genetic algorithms (GAs). Random deactivation of individuals or decision variables 
might serve as diversity-induction mechanism, improving the algorithm’s ability to navigate rugged fitness landscapes with multiple peaks and valleys. 
The toolkit provides a GUI for easily optimize, plot and compare the results of optimization using a vanilla or dropout-based GA. DeapSleep is fully Dockerized, 
but using Docker is optional, although necessary for using the GUI.

DeapSleep includes the following features:
* Genetic Algorithm using DEAP implementation
* Single and multi-objective optimization (NSGA-II, NSGA-III)
* Hall of Fame (Hof) of the best found individuals
* Pymoo benchmark functions and corresponding internal .yaml configurations
* An interactive GUI for:
  * optimizing benchmark functions or custom problems using
    * vanilla GAs
    * individual dropout (IDrop), i.e. decision variables deactivation
    * population dropout (PDrop), i.e. complete individuals deactivation
    * both (I&PDrop)
  * plotting converge-to-target results and HoF statistics over n runs
  * comparing vanilla GAs vs dropout versions

**Disclaimer**: at the moment, DeapSleep does not allow to use custom problems.

## Installation

### 1. Docker installation (necessary for using the GUI)
Clone the repository
```bash
git clone https://github.com/dganci/deapsleep.git
cd deapsleep
```
Build the image
```bash
docker build -t deapsleep .
```
Run the container (and save the results in <YOUR_DIRECTORY>
```bash
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "<YOUR_DIRECTORY>:/app/results" deapsleep
```

### 2. Python installation
Clone the repository
```bash
git clone https://github.com/dganci/deapsleep.git
cd deapsleep
```
We recommend to use a virtual environment, e.g.
```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
```
Then
```bash
pip install -r requirements.txt
```
Examples of usage:
* for optimization
```bash
python3 -m deapsleep.main.optimize --config single.ackley -i --version='baseline_example' --n_runs=10 --ngen=200
```
```bash
python3 -m deapsleep.main.optimize --config single.rastrigin -i --version='I&PDrop_example' --n_runs=30 --ngen=1000 --n_var=10 --popD_rate=0.7 --indD_rate=0.2
```
* for plotting
```bash
python3 -m deapsleep.main.plot --problem griewank --version='baseline_example' --dirname="${PWD}/results"
```
* for comparison
```bash
python3 -m deapsleep.main.compare --config multi.zdt1 -i --version1='baseline_example' --version2='IDrop_example'
```

