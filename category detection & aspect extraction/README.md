# Aspect Based Sentiment Analysis

## Getting started

```
Python version 2.7.10
```

Install the required libraries and dependencies:

```
$ sudo apt-get install python-pip python-dev build-essentials libblas-dev liblapack-dev libatlas-base-dev gfortran
$ pip install numpy ad3 atlas decorator boto bz2file cvxopt cycler nltk httpretty logging path.py pexpect ptyprocess pyparsing pystruct python-dateutil pytz requests scipy Cython scikit-learn simplegeneric six smart-open threadpool traitlets sklearn
```

## Extra packages:

After installing nltk, get into python shell and download its content:

```
python -c "import nltk;nltk.download('all')"
```

## Run the projects:

For the constrained submission on the Restaurants domain, for the OTE task:

```
$ python ote_constrained_restaurants.py --train data/restaurants/train.xml --test data/restaurants/test.xml 
```

For the unconstrained submission on the Restaurants domain, for the OTE task:

```
$ python ote_unconstrained_restaurants.py --train data/restaurants/train.xml --test data/restaurants/test.xml 
```

For the constrained submission on the Restaurants domain, for the ACD task:

```
$ python acd_constrained_restaurants.py --train data/restaurants/train.xml --test data/restaurants/test.xml 
```

For the unconstrained submission on the Restaurants domain, for the ACD task:

```
$ python acd_unconstrained_restaurants.py --train data/restaurants/train.xml --test data/restaurants/test.xml 
```

For the constrained submission on the Laptops domain, for the ACD task:

```
$ python acd_constrained_laptops.py --train data/laptops/train.xml --test data/laptops/test.xml 
```

For the unconstrained submission on the Laptops domain, for the ACD task:

```
$ python acd_unconstrained_laptops.py --train data/laptops/train.xml --test data/laptops/test.xml 
```


