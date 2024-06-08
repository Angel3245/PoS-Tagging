from setuptools import setup

setup(
   name='PoS Tagger',
   version='1.0',
   description='Develop a PoS tagger using the CoNLL-U datasets, which are widely used for training and evaluating PoS tagging models.',
   author='Jose Ángel Pérez Garrido',
   author_email='jpgarrido19@esei.uvigo.es',
   packages=['postagger'],
   install_requires=['wheel', 'bar', 'greek'], #external packages as dependencies
)