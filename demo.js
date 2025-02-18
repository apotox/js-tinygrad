const {
    MLP
} = require('./tinygrad');

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

console.log(n.forward([1, 0, 0])[0].data, 'expected 1')
console.log(n.forward([1, 1, 0])[0].data, 'expected 0')
console.log(n.forward([1, 0, 1])[0].data, 'expected 0')
console.log(n.forward([0, 1, 1])[0].data, 'expected 0')
