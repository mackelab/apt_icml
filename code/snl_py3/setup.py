from setuptools import setup

setup(
    name='snl',
    version='0.1',
    description='Sequential neural likelihood',
    url='https://github.com/gpapamak/snl',
    author='original code from George Papamakarios',
    packages=['snl', 'snl.ml', 'snl.ml.models', 'snl.pdfs', 'snl.simulators', 'snl.util', 
              'snl.inference', 'snl.inference.diagnostics'],
    license='MIT',
    install_requires=['dill', 'lasagne==0.2.dev1', 'numpy', 'scipy', 'theano', 'tqdm'],
    dependency_links=[
        'https://github.com/Lasagne/Lasagne/archive/master.zip#egg=lasagne-0.2.dev1',
    ]
)
