## README.md ##
=========
### Dependencies: ###
Python 2.7.3 installed

### Basic Build [Linux]: ###
Download the folder category_detection and on your terminal type
```
 $ pip install -r requirements.txt
```

### Usage: ###
Depending on the domain (Restaurants, Laptops) and the type of submission (Constrained, Unconstrained) 
you can run a script in format
```
$ python acd_[typeOfSubmission]_[domain].py --train [train_file] --test [test_file]
```
, e.g.
```
$ python acd_unconstrained_laptops.py --train train.xml --test test.xml
```
