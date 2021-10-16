# Fast Texture Synthesis using Tree Structured Vector Quantization
This is a Pytorch implementation of the paper titled "[Fast Texture Synthesis using Tree-structured Vector Quantization][tvsq]" by Li-Yi Wei and Marc Levoy (2000,SIGGRAPH).

## Requirements
The implementation was tested using the specified versions of the following packages: 
* Python 3.7.9
* Pytorch 1.9.0
* Pillow 8.1.0
* Numpy 1.19.2
* JupyterLab (if you want to run the notebook, demo.ipynb)

## Execution
Use the following command if you would like to use a constant neighborhood size:
````
python tvsq.py --in_path INPUT_TEXTURE_PATH --out_path SAVE_OUTPUT_TEXTURE_PATH [--n_levels NUM_LEVELS] --n_size NEIGHBORHOOD_SIZE  [--parent_size PARENT_SIZE] [--in_size INPUT_SIZE] [--out_size OUTPUT_SIZE]
````

Use the following command if you would like to use different neighborhood sizes per level:
````
python tvsq.py --in_path INPUT_TEXTURE_PATH --out_path SAVE_OUTPUT_TEXTURE_PATH [--n_levels NUM_LEVELS] --n_sizes [NEIGHBORHOOD_SIZES]  [--parent_sizes [PARENT_SIZES]] [--in_size INPUT_SIZE] [--out_size OUTPUT_SIZE]
````

Either n_size or n_sizes must be specified. Also, length of n_sizes and parent_sizes must be equal to n_levels.

For details on the arguments, run `python tvsq.py -h` or `python tvsq.py --h`. 

## References
* Wei, L. Y., & Levoy, M. (2000, July). Fast texture synthesis using tree-structured vector quantization. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques (pp. 479-488).

[tvsq]: https://graphics.stanford.edu/papers/texture-synthesis-sig00/texture.pdf