# orthogonal-augmentation
Scripts and data for the L4DC submission of the paper "Orthogonal projection-based regularization for efficient model augmentation".

## Installation
First, clone the repository, then open the project folder as
```
$ git clone https://github.com/AIMotionLab-SZTAKI/orthogonal-augmentation
$ cd orthogonal-augmentation/
```

It is recommended to use a virtual environment. E.g., on Linux/Bash run
```
$ python3 -m venv venv
$ source venv/bin/activate
```

Then finally, install the package and its dependencies
```
$ pip install -e .
```

## Usage
To train the augmented model, run
```
$ python3 scripts/orth_training.py
```
Training options can be chosen by editing the script. Furthermore, black-box models can be trained for benchmarking by running the `bb_ident.py` script.

## License
See the [LICENSE](/LICENSE.md) file for license rights and limitations (MIT).
