Requirements
============

This are the list of the modules/library used in this project with Python 3.7.


Installation with pip
---------------------


.. literalinclude:: ../requirements

Create a txt file contening the lines above and execute the following line to install all packages with pip.

.. code-block:: bash

	pip install -r requirements.txt


Installation with Anaconda
--------------------------


We encourage you to use `Anaconda <https://www.anaconda.com/>`_ to manage your packages and environments.


After installing Anaconda (`download <https://www.anaconda.com/distribution/#download-section>`_) in Python 3.7 version

Use the folling line to create a new environment named 'p3' with some tools already installed then the second line to activate it.

.. code-block:: bash

	conda create -n p3 python=3  anaconda
	conda activate p3

The library used for the neural network is TensorFlow, you have two possibilities to install and use it.

+ Use it with your CPU and install it with the line :

.. code-block:: bash

	conda install -c conda-forge tensorflow 

+ Use it with your GPU (faster) but you will need to install `CUDA <https://developer.nvidia.com/cuda-downloads>`_ before installing with this line :

.. code-block:: bash

	conda install -c anaconda tensorflow-gpu

Then you can install the others libraries with thoses lines :

.. code-block:: bash

	conda install -c rdkit rdkit
	conda install -c conda-forge keras
	conda install -c omnia cclib
	pip install pptree
	conda install -c anaconda joblib
	conda install -c conda-forge tensorboard


If the default framework used by keras is Theanos use the following line to switch to TensorFlow (print 'Using TensorFlow backend.' / 'Using Theanos backend.' when you launch the program) :

.. code-block:: bash

	 export KERAS_BACKEND='tensorflow'

Other libraries
---------------

If you are using 3D calculation (DFT,...), you will also need to install `OpenBabel <http://openbabel.org/wiki/Main_Page>`_ and `Gaussian <https://gaussian.com/>`_ or `NWChem <http://www.nwchem-sw.org/index.php/Main_Page>`_ (DFT coded for Gaussian in the project)

