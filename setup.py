#!/usr/bin/env python

from setuptools import setup

packages = ['embera',
            'embera.architectures',
            'embera.benchmark',
            'embera.composites',
            'embera.interfaces',
            'embera.preprocess',
            'embera.utilities',
            ]

install_requires = ['dimod>=0.8.0,<0.9.0',
                    'dwave_networkx>=0.8.0,<0.9.0',
                    'dwave-system>=0.8.0,<0.9.0',
                    'matplotlib>=3.1.0,<4.0.0',
                    'minorminer>=0.1.5,<0.2.0',
                    'networkx>=2.0,<3.0',
                    'numpy>=1.15.2,<2.00',
                    'pulp>=1.6.0,<2.0.0',
                    'scipy>=1.4.0,<2.0.0',
                    ]

setup(name='embera',
      version='0.0.1',
      description='Embedding Resources and Algorithms',
      long_description="Collection of minor-embedding methods and utilities "
      "to map unstructured binary quadratic problems to a structured sampler "
      "such as a D-Wave system.",
      author='Jose Pinilla',
      author_email='jpinilla@ece.ubc.ca',
      url='https://github.com/joseppinilla/embera',
      packages=packages,
      platforms='any',
      install_requires=install_requires,
      python_requires='>=3.6', # f-string support
      license='MIT'
     )
