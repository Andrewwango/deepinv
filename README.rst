.. image:: https://github.com/deepinv/deepinv/raw/main/docs/source/figures/deepinv_logolarge.png
   :width: 500px
   :alt: deepinv logo
   :align: center


|Test Status| |Docs Status| |Python Version| |Black| |codecov| |pip install| |discord| |colab|   


Introduction
------------
DeepInverse is an open-source PyTorch-based library for solving imaging inverse problems using deep learning. The goal of ``deepinv`` is to accelerate the development of deep learning based methods for imaging inverse problems, by combining popular learning-based reconstruction approaches in a common and simplified framework, standardizing forward imaging models and simplifying the creation of imaging datasets.

``deepinv`` features


* A large collection of `predefined imaging operators <https://deepinv.github.io/deepinv/user_guide/physics/physics.html>`_ (MRI, CT, deblurring, inpainting, etc.)
* `Training losses <https://deepinv.github.io/deepinv/user_guide/training/loss.html>`_ for inverse problems (self-supervised learning, regularization, etc.)
* Many `pretrained deep denoisers <https://deepinv.github.io/deepinv/user_guide/reconstruction/weights.html>`_ which can be used for `plug-and-play restoration <https://deepinv.github.io/deepinv/user_guide/reconstruction/iterative.html>`_
* A framework for `building datasets <https://deepinv.github.io/deepinv/user_guide/training/datasets.html>`_ for inverse problems
* Easy-to-build `unfolded architectures <https://deepinv.github.io/deepinv/user_guide/reconstruction/unfolded.html>`_ (ADMM, forward-backward, deep equilibrium, etc.)
* `Sampling algorithms <https://deepinv.github.io/deepinv/user_guide/reconstruction/sampling.html>`_ for uncertainty quantification (Langevin, diffusion, etc.)
* A large number of well-explained `examples <https://deepinv.github.io/deepinv/auto_examples/index.html>`_, from basics to state-of-the-art methods

.. image:: https://github.com/deepinv/deepinv/raw/main/docs/source/figures/deepinv_schematic.png
   :width: 1000px
   :alt: deepinv schematic
   :align: center


Documentation
-------------

Read the documentation and examples at `https://deepinv.github.io <https://deepinv.github.io>`_.

Install
-------

To install the latest stable release of ``deepinv``, you can simply do:

.. code-block:: bash

    pip install deepinv

    # or using uv for faster installation

    uv pip install deepinv

You can also install the latest version of ``deepinv`` directly from github:

.. code-block:: bash

    pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv

You can also install additional dependencies needed for some modules in deepinv.datasets and deepinv.models:

.. code-block:: bash

    pip install deepinv[dataset,denoisers]

    # or

    pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv[dataset,denoisers]

Since ``deepinv`` is under active development, you can update to the latest version easily using:

.. code-block:: bash

    pip install --upgrade --force-reinstall --no-deps git+https://github.com/deepinv/deepinv.git#egg=deepinv


Quickstart
----------
Try out the following plug-and-play image inpainting example:

.. code-block:: python

   import deepinv as dinv
   from deepinv.utils import load_example

   x = load_example("cameraman.png", img_size=512, grayscale=True, device='cpu')

   physics = dinv.physics.Inpainting((1, 512, 512), mask = 0.5, \
                                       noise_model=dinv.physics.GaussianNoise(sigma=0.01))

   data_fidelity = dinv.optim.data_fidelity.L2()
   prior = dinv.optim.prior.PnP(denoiser=dinv.models.MedianFilter())
   model = dinv.optim.optim_builder(iteration="HQS", prior=prior, data_fidelity=data_fidelity, \
                                    params_algo={"stepsize": 1.0, "g_param": 0.1})
   y = physics(x)
   x_hat = model(y, physics)
   dinv.utils.plot([x, y, x_hat], ["signal", "measurement", "estimate"], rescale_mode='clip')


Also try out `one of the examples <https://deepinv.github.io/deepinv/auto_examples/index.html>`_ to get started or check out our comprehensive `User Guide <https://deepinv.github.io/deepinv/user_guide.html>`_.

Contributing
------------

DeepInverse is a community-driven project and welcomes contributions of all forms.
We are building a comprehensive library of inverse problems and deep learning,
and we need your help to get there!

Please contribute to ``deepinv`` by forking the `main
repository <https://github.com/deepinv/deepinv/>`_ on GitHub,
then submit a "Pull Request" (PR). See our `contributing guide <https://deepinv.github.io/deepinv/contributing.html>`_
for full instructions on how to contribute.


Finding help
------------

If you have any questions or suggestions, please join the conversation in our
`Discord server <https://discord.gg/qBqY5jKw3p>`_. The recommended way to get in touch with the developers is to open an issue on the
`issue tracker <https://github.com/deepinv/deepinv/issues>`_.

Maintainers
-----------

View our current and former maintainers, and how to get in touch, at `MAINTAINERS <https://github.com/deepinv/deepinv/blob/main/MAINTAINERS.md>`_.


Citation
--------
If you use DeepInverse in your research, please cite the following paper (available on `arXiv <https://arxiv.org/abs/2505.20160>`_):


.. code-block:: bash

    @software{tachella2025deepinverse,
          title={DeepInverse: A Python package for solving imaging inverse problems with deep learning},
          author={Julián Tachella and Matthieu Terris and Samuel Hurault and Andrew Wang and Dongdong Chen and Minh-Hai Nguyen and Maxime Song and Thomas Davies and Leo Davy and Jonathan Dong and Paul Escande and Johannes Hertrich and Zhiyuan Hu and Tobías I. Liaudat and Nils Laurent and Brett Levac and Mathurin Massias and Thomas Moreau and Thibaut Modrzyk and Brayan Monroy and Sebastian Neumayer and Jérémy Scanvic and Florian Sarron and Victor Sechaud and Georg Schramm and Romain Vo and Pierre Weiss},
          year={2025},
          eprint={2505.20160},
          archivePrefix={arXiv},
          primaryClass={eess.IV},
          url={https://arxiv.org/abs/2505.20160},
    }


Star history
------------

.. image:: https://api.star-history.com/svg?repos=deepinv/deepinv&type=Date
   :alt: Star History Chart
   :target: https://www.star-history.com/#deepinv/deepinv&Date


.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. |Test Status| image:: https://github.com/deepinv/deepinv/actions/workflows/test.yml/badge.svg
   :target: https://github.com/deepinv/deepinv/actions/workflows/test.yml
.. |Docs Status| image:: https://github.com/deepinv/deepinv/actions/workflows/documentation.yml/badge.svg
   :target: https://github.com/deepinv/deepinv/actions/workflows/documentation.yml
.. |Python Version| image:: https://img.shields.io/badge/python-3.10%2B-blue
   :target: https://www.python.org/downloads/release/python-3100/
.. |codecov| image:: https://codecov.io/gh/deepinv/deepinv/branch/main/graph/badge.svg?token=77JRvUhQzh
   :target: https://codecov.io/gh/deepinv/deepinv
.. |pip install| image:: https://img.shields.io/pypi/dm/deepinv.svg?logo=pypi&label=pip%20install&color=fedcba
   :target: https://pypistats.org/packages/deepinv
.. |discord| image:: https://dcbadge.limes.pink/api/server/qBqY5jKw3p?style=flat
   :target: https://discord.gg/qBqY5jKw3p
.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1XhCO5S1dYN3eKm4NEkczzVU7ZLBuE42J
