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
Run the following command:
````
python tvsq.py --in_path [path_to_input_texture] --out_path [path_to_save_output_texture] --n_levels [number_of_pyramid_levels] --n_size [neighborhood_size]
````

## References
* Wei, L. Y., & Levoy, M. (2000, July). Fast texture synthesis using tree-structured vector quantization. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques (pp. 479-488).

[tvsq]: https://graphics.stanford.edu/papers/texture-synthesis-sig00/texture.pdf