How to install HDNNP
====================

..  contents::
    :local:
    :depth: 2




Python installation
---------------------

| We recommend that you install python using pyenv,
  because non-sudo user can install any python version on any computer.
| We confirmed that this program works only with python 3.6.7.

..  code-block:: shell

    (on Linux)
    $ git clone https://github.com/yyuu/pyenv.git ~/.pyenv
    (on MacOS)
    $ brew install pyenv

    $ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
    $ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
    $ echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
    $ source ~/.bash_profile

    $ pyenv install 3.6.7



Get source code
---------------------

..  note::

    | This program is now under development, not uploaded to PyPI.
    | You have to get source code and install it manually.

..  code-block:: shell

    $ git clone https://github.com/ogura-edu/HDNNP.git

Install dependencies and this program
-------------------------------------

Via pipenv
^^^^^^^^^^^^^^^^^^^^^

..  code-block:: shell

    $ cd HDNNP/
    $ pyenv local 3.6.7
    $ pip install pipenv
    $ pipenv install --dev

    (activate)
    $ pipenv shell

    (for example:)
    (HDNNP) $ hdnnpy train

    (deactivate)
    (HDNNP) $ exit


Via anaconda
^^^^^^^^^^^^^^^^^^^^^

Anaconda also can be installed by pyenv.

..  code-block:: shell

    $ cd HDNNP/
    $ pyenv install anaconda3-xxx
    $ pyenv local anaconda3-xxx
    $ conda env create -n HDNNP --file condaenv.yaml

    (activate)
    $ conda activate HDNNP

    (for example:)
    (HDNNP) $ hdnnpy train

    (deactivate)
    (HDNNP) $ conda deactivate



Via raw pip
^^^^^^^^^^^^^^^^^^^^^

You can install all dependent packages manually.
The dependent packages are written in ``Pipfile``, ``condaenv.yaml`` or ``requirements.txt``.

..  code-block:: shell

    $ cd HDNNP/
    $ pip install PKG1 PKG2 ...
    $ pip install --editable .
