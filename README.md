## Project: rubiks-cube 

This project is a collection of Python classes developed to explore algorithmic solutions to Rubik's Cube. The repository contains a set of notebooks to walk interested parties through my progress with the puzzle thus far, and highlight some features of the classes for modeling and displaying the cube contained herein. For a preliminary and complete explanation of my solution strategies thus far, please peruse the blog article referenced above.

___


### Installation:

The repository is setup for pipenv configuration management, thus you'll find a Pipfile rather than requirements.txt file.  
You may access installation instructions for pipenv here: [Pipenv Installation Instructions](https://pypi.org/project/pipenv/)

Once pipenv has been successfully installed, the following commands may be executes to install the rubiks-cube project:

```
$ git clone https://github.com/ajdonich/rubiks-cube
$ cd rubik-cube
$ pipenv install
$ pipenv --dev install
```

___


### Execution:

To understand the specifics of this project, please step through the interactive set of IPython/Jupyter notebooks provided. First, launch a notebook session from the command line using:

```
$ pipenv shell
$ jupyter lab
```

Then from within the notebook session that is launched in your browser, navigate to the notebooks directory. This directory contains this set of .ipynb files:

1. Nb1_Intro_Cube_View.ipynb
2. Nb2_Neural_Networks.ipynb
3. Nb3_CFOP_Algorithm.ipynb
4. Nb4_Cycles_Entropy.ipynb

Please step through these files, titled numerical with the suggested running order. Execute each cell of each notebook in turn, exploring the code execution output. You'll also find a number of markdown cells with further descriptive details. I hope you find the demonstrations interesting and enjoyable! 

