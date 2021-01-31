import logging

from hypothesis import settings

# turn off Numba logging
logging.getLogger('numba').setLevel(logging.INFO)


# set up profiles
settings.register_profile('large', max_examples=5000)
