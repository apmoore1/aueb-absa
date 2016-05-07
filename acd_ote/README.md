## Aspect Category Detection and Opinion Target Extraction

### Getting started

```
Python version 2.7.10
Operating System : Linux (Ubuntu)
```

### Download

```
$ git clone https://github.com/nlpaueb/aueb-absa.git
$ cd aueb-absa/acd_ote
```

If you are using virtualenv:

```
$ virtualenv env
$ source env/bin/activate
```

### Install the required libraries:

```
$ pip install -r requirements.txt
$ pip install pystruct
```

### Extra packages:

After installing nltk, get into python shell and download its content:

```
>> python -c "import nltk;nltk.download('punkt');nltk.download('averaged_perceptron_tagger')"
```

### Run the projects:

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

## Troubleshooting

Make sure that you have the following requirements installed:

```
$ sudo apt-get install python-pip python-dev build-essentials libblas-dev liblapack-dev libatlas-base-dev gfortran git
```

## Semeval 2016

Our results:

Aspect Category Detection
```
Restaurants (Unconstrained) 71.54% F1, ranked 4th/30
Restaurants (Constrained) 67.35% F1, ranked 4th/12
Laptops (Unconstrained) 49.10% F1, ranked 2nd/22
Laptops (Constrained) 45.63% F1, ranked 4th/9
```

Opinion Target Expression
```
Restaurants (Unconstrained) 70.44% F1, ranked 2nd/19
Restaurants (Constrained) 61.55% F1, ranked 6th/8
```

You can download the necessary data and tools from the Semeval's web page:

```
http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools
```
