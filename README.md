# pyqumo - queueing models in Python

## Package status

Current version is pre-release development.
What will be changed prior to 1.0.0 release:

- C++ code will be moved closer to the modules using it
- merge `sim` and `cqumo.sim` packages
- reorganize `algorithms` and `fitting`
- move unstable code that is subject to changes into separate `experimental` package
- split large modules like `arrivals` and `randoms` into smaller modules (interface will be kept)
- Sphinx documentation
- GitHub actions for automatic testing

What will *hopefully* put into the release:

- Windows 10+ support

Expected release date: **01 june 2023**

If you need the package, please contact me directly via larioandr@gmail.com or post an issue.


## How to install for development

1. Install prerequisites

> **IMPORTNAT**: right now this package will not work under native Windows environment.
> To use it under Windows, please consider using WSL2 with some kind of Linux on board.

To compile the C++ extensions, `g++` and `gcc` need to be installed. Also the C++ sources
make use of `Python3.h`, so `python3-dev` or similiar package need to be installed. 
The following instructions relate to Ubuntu users, but something similiar may be done
under Mac or other Linux distros.

```bash
~> sudo apt install python3-dev
~> sudo apt install gcc g++
```

> If you are using WSL2 with Ubuntu, you will need to run these lines above.
> Please note, that it is better to be inside Linux partition (e.g. /home/username/),
> and NOT inside Windows partition (e.g. /mnt/Windows/...), since using Windows
> partition may lead to access rights problems.

2. Clone the repository (say, to `/home/username/pyqumo`) and go to its root:

```bash
~> git clone https://github.com/ipu69/pyqumo
```

3. Create a virtual environment using Python 3.8 and activate it.
To display a nice message when using the venv you can provide its name using `--prompt` key:

```bash
~/pyqumo> python3 -m venv .venv --prompt pyqumo
~/pyqumo> source .venv/bin/activate
```

4. Install the package in development mode:

```bash
~/pyquom> pip install -e .
```

5. Run tests:

```bash
~/pyqumo> pip install pytest
~/pyqumo> pytest
```

## Notes

To use `jupyter-lab` and enable progress bars, execute the following:

```bash
$ jupyter nbextension enable --py widgetsnbextension
$ jupyter labextension install @jupyter-widgets/jupyterlab-manager
