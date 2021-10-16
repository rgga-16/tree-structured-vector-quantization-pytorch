# Fast Texture Synthesis using Tree Structured Vector Quantization
This is a Pytorch implementation of the paper titled "[Fast Texture Synthesis using Tree-structured Vector Quantization][tvsq]" by Li-Yi Wei and Marc Levoy (2000,SIGGRAPH).

![Input texture](texture1.jpg?raw=true "Input texture")
![Synthetic texture](texture1_s.jpg?raw=true "Synthetic texture")

## Requirements
The implementation was tested using the specified versions of the following packages: 
* Python 3.7.9
* Pytorch 1.9.1
* Pillow 8.3.1
* Numpy 1.21.2
* tqdm 4.62.3

## Execution
Sample command if you would like to use a constant neighborhood size:
````
python tvsq.py --in_path texture1.jpg --out_path texture1_s.jpg --n_levels 4 --n_size 5 --parent_size 5  --in_size 64 --out_size 128
````

Sample command if you would like to use different neighborhood sizes per level:
````
python tvsq.py --in_path texture1.jpg --out_path texture1_s.jpg --n_levels 4 --n_sizes 3 5 7 9 --parent_sizes 3 5 7 9  --in_size 64 --out_size 128
````

Either `n_size` or `n_sizes` must be specified (same applies for `parent_size`). Also, make sure `len(n_sizes)` and `len(parent_sizes)` are equal to `n_levels`.

For details on the arguments, run `python tvsq.py -h` or `python tvsq.py --h`. 

## References
* Wei, L. Y., & Levoy, M. (2000, July). Fast texture synthesis using tree-structured vector quantization. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques (pp. 479-488).


## Acknowledgements
- Some parts of the code were based on the implementation, [multi-resolution-texture-synthesis][mrts], by [anopara][user1].


- Input texture image borrowed from [Texture Synthesis][ts] by [1iyiwei][user2]. 


[tvsq]: https://graphics.stanford.edu/papers/texture-synthesis-sig00/texture.pdf
[mrts]: https://github.com/anopara/multi-resolution-texture-synthesis#multi-resolution-texture-synthesis
[user1]: https://github.com/anopara
[ts]: https://github.com/1iyiwei/texture. 
[user2]: https://github.com/1iyiwei
