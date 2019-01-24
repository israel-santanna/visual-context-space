# visual-context-space
An experiment with object's context in the COCO dataset

## Usage

```bash
python3 main.py [-h] [-i INPUT] [-v VALIDATION] [-s SAVE_PATH] [-l LOAD_PATH] [-m MEN_PATH] [-c]
```
### Arguments:
  
* -h, --help
  
  Show help message and exit
  
* -i INPUT, --input INPUT
  
  File containing the COCO training data
                        
* -v VALIDATION, --validation VALIDATION
  
  File containing the COCO validation data
                        
* -s SAVE_PATH, --save_path SAVE_PATH
  
  Path where to save the Word2Vec trained data
                        
* -l LOAD_PATH, --load_path LOAD_PATH
  
  File containing the Word2Vec trained data
                        
* -m MEN_PATH, --men_path MEN_PATH
  
  File containing the MEN dataset for evaluation
                        
* -c, --console
  
  Show a console where you can query W2V for the objects more commonly in the same context


## Requirements

[Gensim](https://github.com/RaRe-Technologies/gensim)

[Pandas](https://pandas.pydata.org/)

[NumPy](http://www.numpy.org)

[COCO API](https://github.com/cocodataset/cocoapi)