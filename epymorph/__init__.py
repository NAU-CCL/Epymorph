"""epymorph initialization."""

from numpy import seterr

# set numpy errors to raise exceptions instead of warnings;
# useful for catching simulation errors
seterr(all="raise")
