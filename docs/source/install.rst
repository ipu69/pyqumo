.. _install_guide:

=================
Installing pyqumo
=================


    **IMPORTANT**: right now, only development installation is possible and tested under Linux.
    Simple install using PyPI will be available soon.


    **COMPATIBILITY**: ``pyqumo`` will not work under native Windows environment.
    To use it under Windows, please consider using WSL2 with some kind of Linux on board.


Install prerequisites
=====================

To compile the C++ extensions, ``g++`` and ``gcc`` need to be installed. Also the C++ sources
make use of ``Python3.h``, so ``python3-dev`` or similar package need to be installed.
The following instructions relate to Ubuntu users, but something similar may be done
under Mac or other Linux distros.

    **COMPATIBILITY:** If you are using WSL2 with Ubuntu, you will need to run these lines above.
    Please note, that it is better to be inside Linux partition (e.g. /home/username/),
    and NOT inside Windows partition (e.g. /mnt/Windows/...), since using Windows
    partition may lead to access rights problems.

::

    sudo apt install python3-dev
    sudo apt install gcc g++

Install pyqumo
==============

Clone the repository (say, to `/home/username/pyqumo`) and go to its root:

::

    git clone https://github.com/ipu69/pyqumo


Create a virtual environment using Python 3.8 and activate it.
To display a nice message when using the venv you can provide its name using ``--prompt`` key:

::

    python3 -m venv .venv --prompt pyqumo
    source .venv/bin/activate


Install the package in development mode:

::

    pip install -e .

Run tests:

::

    pip install pytest
    pytest

Notes
=====

To use ``jupyter-lab`` and enable progress bars, execute the following:

::

    jupyter nbextension enable --py widgetsnbextension
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
