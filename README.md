# Fast Texture Synthesis using Tree Structured Vector Quantization
This is a Pytorch implementation of the paper titled "[Fast Texture Synthesis using Tree-structured Vector Quantization][tvsq]" by Li-Yi Wei and Marc Levoy (2000,SIGGRAPH).

[tvsq]: https://graphics.stanford.edu/papers/texture-synthesis-sig00/texture.pdf

## Requirements
The implementation was tested using the specified versions of the following packages: 
* Python 3.7.9
* Pytorch 1.9.0
* Pillow 8.1.0
* Numpy 1.19.2
* JupyterLab (if you want to run the notebook, demo.ipynb)

## Execution
Sample command if you would like to use a constant neighborhood size:
````
python tvsq.py --in_path texture1.jpg --out_path texture1_s.jpg --n_levels 4 --n_size 5 --parent_size 5  --in_size 64 --out_size 128
````

Sample command if you would like to use different neighborhood sizes per level:
````
python tvsq.py --in_path texture1.jpg --out_path texture1_s.jpg --n_levels 4 --n_sizes 3 5 7 9 --parent_sizes 3 5 7 9  --in_size 64 --out_size 128
````

Either `n_size` or `n_sizes` must be specified. Also, make sure `len(n_sizes)` and `len(parent_sizes)` are equal to `n_levels`.

For details on the arguments, run `python tvsq.py -h` or `python tvsq.py --h`. 

## References
* Wei, L. Y., & Levoy, M. (2000, July). Fast texture synthesis using tree-structured vector quantization. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques (pp. 479-488).


## Acknowledgements
- Some parts of the code were based on the implementation, [multi-resolution-texture-synthesis][mrts], by [anopara][user1].
[mrts]: https://github.com/anopara/multi-resolution-texture-synthesis#multi-resolution-texture-synthesis
[user1]: https://github.com/anopara
- Input texture image borrowed from [Texture Synthesis][ts] by [1iyiwei][user2]. 
[ts]: https://github.com/1iyiwei/texture. 
[user2]: https://github.com/1iyiwei