**Morphological (boolean)**

* Capital first letter
* All letters in capitals
* All letters in capitals, except the 1st one
* All letters in lowercase
* Only words
* Only digits
* Existence of full stop mark
* Existence of dash mark
* Existence of punctuation mark, except the full stop and the dash mark

**Lexicon based (one hot)**

* Part-of-Speech tags (of current token and of the 2 next and 2 previous ones surrounding it)
* Preffixes ans Suffixes of length 1, 2 and 3, found in the training data
* Aspect terms found more than once on the training data

**Word embeddings based features**

* Embedding of current token and of the 2 next and 2 previous ones surrounding it


