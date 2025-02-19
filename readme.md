### js-tinygrad
a tiny, no-dependency, no-WebGL, no-nonsense neural network cute code with bAcKproPaGation math for training and running models entirely in js. just for fun and learning. inspired by [micrograd](https://github.com/karpathy/micrograd).


### Usage
```js
const xs = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
]

const ys = [0, 1, 1, 0, 1, 0, 0, 1]

const n = new MLP(3, [4, 4, 1]);

n.train(xs, ys, 0.02, 1000, 'tanh');
```


### Example
```bash

...
loss 2.743117174117855
loss 0.9278638782737499
loss 0.5331272483131881
loss 0.0361078840714764
loss 0.010347189275518857
loss 0.0066721562754008595
loss 0.005135900905437655
loss 0.004313324610138
loss 0.0038125523633205263
loss 0.003325467685183328
...
results:
0.9843057630490994 expected 1
0.020020640843521776 expected 0
0.021857685609633953 expected 0
0.01985133603252693 expected 0
```
