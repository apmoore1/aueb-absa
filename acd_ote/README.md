# Aspect Based Sentiment Analysis

## Getting started

```
Python version 2.7.10
```

## Download

```
$ git clone https://github.com/nlpaueb/aueb-absa.git
$ cd acd_ote
```

If you are using virtualenv:

```
$ virtualenv env
$ source env/bin/activate
```

## Install the required libraries and dependencies:

```
$ sudo apt-get install python-pip python-dev build-essentials libblas-dev liblapack-dev libatlas-base-dev gfortran
$ pip install -r requirements.txt
$ pip install pystruct
```

## Extra packages:

After installing nltk, get into python shell and download its content:

```
>> python -c "import nltk;nltk.download('punkt');nltk.download('averaged_perceptron_tagger')"
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


