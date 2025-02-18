
class Value {
    static New(data, _children = []) {
        const v = new Value();
        v.data = data;
        v._children = _children;
        v.grad = 0;
        v._backward = null;

        return v;
    }

    static Add(self, other) {
        if (!(other instanceof Value)) {
            other = Value.New(other);
        }
        const parent = Value.New(self.data + other.data, [self, other]);
        parent._backward = () => {
            self.grad += 1 * parent.grad;
            other.grad += 1 * parent.grad;
        }

        return parent;
    }

    static Mins(self, other) {
        if (!(other instanceof Value)) {
            other = Value.New(other);
        }

        return Value.Add(self, Value.Mul(other, -1));
    }

    static Mul(self, other) {
        if (!(other instanceof Value)) {
            other = Value.New(other);
        }
        const parent = Value.New(self.data * other.data, [self, other]);
        parent._backward = () => {
            self.grad += 1 * other.data * parent.grad;
            other.grad += 1 * self.data * parent.grad;
        }

        return parent;
    }

    static Tanh(self) {
        const parent = Value.New(Math.tanh(self.data), [self]);
        parent._backward = () => {
            self.grad += (1.0 - parent.data * parent.data) * parent.grad;
        }
        return parent;
    }

    static Relu(self) {
        const parent = Value.New(self.data < 0 ? 0 : self.data, [self]);
        parent._backward = () => {
            self.grad += (self.data < 0 ? 0 : 1) * parent.grad;
        }
        return parent;
    }

    static Sigmoid(self) {
        const parent = Value.New(1 / (1 + Math.exp(-self.data)), [self]);
        parent._backward = () => {
            self.grad += parent.data * (1 - parent.data) * parent.grad;
        }
        return parent;
    }

    static Pwr(self, k) {
        const parent = Value.New(Math.pow(self.data, k), [self]);
        parent._backward = () => {
            self.grad += k * Math.pow(self.data, k - 1) * parent.grad;
        }
        return parent;
    }


    static Backward(self) {
        self.grad = 1;
        let queue = [self];
        while (queue.length > 0) {
            const v = queue.pop();
            if (v._backward) {
                v._backward();
            }

            queue = [...queue, ...v._children];
        }
    }
}




class Neuron {
    constructor(nInputs) {
        this.W = new Array(nInputs).fill(0).map(() => Value.New(Math.random()));
        this.b = Value.New(0);
    }

    forward(inputs, activation = 'tanh') {
        let sum = Value.New(0);
        for (let i = 0; i < inputs.length; i++) {
            sum = Value.Add(sum, Value.Mul(this.W[i], inputs[i]));
        }
        const v = Value.Add(sum, this.b);
        switch (activation) {
            case 'tanh':
                return Value.Tanh(v);
            case 'relu':
                return Value.Relu(v);
            case 'sigmoid':
                return Value.Sigmoid(v);
            default:
                throw new Error('Invalid activation function');
        }
    }

    parameters() {
        return [...this.W, this.b];
    }
}

class Layer {
    constructor(nInputs, nOutput) {
        this.neurons = new Array(nOutput).fill(0).map(() => new Neuron(nInputs));
    }

    forward(inputs, act = 'tanh') {
        return this.neurons.map(neuron => neuron.forward(inputs, act));
    }

    parameters() {
        let arr = []
        for (let neuron of this.neurons) {
            arr = [...arr, ...neuron.parameters()];
        }
        return arr;
    }
}

class MLP {
    constructor(nInputs, nOutputs = [4, 1]) {
        this.nInputs = nInputs;
        this.layers = new Array(nOutputs.length).fill(0).map((_, i) => {
            return new Layer(i === 0 ? nInputs : nOutputs[i - 1], nOutputs[i]);
        });
    }

    static Load(str) {
        const data = JSON.parse(str);
        const nInputs = data.nInputs;

        const layersSizes = data.layers.map(layer => layer.neurons.length);
        const mlp = new MLP(nInputs, layersSizes);

        for (let i = 0; i < mlp.layers.length; i++) {
            for (let j = 0; j < mlp.layers[i].neurons.length; j++) {
                const neuron = mlp.layers[i].neurons[j];
                neuron.W = data.layers[i].neurons[j].W.map(w => Value.New(w));
                neuron.b = Value.New(data.layers[i].neurons[j].b);
            }
        }

        return mlp;
    }

    forward(inputs, activation = 'tanh') {
        let i = inputs;
        for (let layer of this.layers) {
            i = layer.forward(i, activation);
        }
        return i;
    }

    zeroGrad() {
        for (const param of this.parameters()) {
            param.grad = 0;
        }
    }

    parameters() {
        let arr = []
        for (let layer of this.layers) {
            arr = [...arr, ...layer.parameters()];
        }
        return arr;
    }

    toString() {
        let data = {
            nInputs: this.nInputs,
            layers: this.layers.map(layer => {
                return {
                    neurons: layer.neurons.map(neuron => {
                        return {
                            W: neuron.W.map(w => w.data),
                            b: neuron.b.data
                        }
                    })
                }
            })
        }

        return JSON.stringify(data);
    }

    saveToFile(path = 'model.json') {
        const fs = require('fs');
        const str = this.toString();
        fs.writeFileSync(path, str);
    }

    train(xs, ys, lr = 0.01, epochs = 100, activation = 'tanh') {
        for (let i = 0; i < epochs; i++) {
            this.zeroGrad();
            const ypreds = xs.map(x => this.forward(x, activation)[0]);
            const loss = ypreds.map((ypred, i) => Value.Pwr(Value.Mins(ypred, ys[i]), 2)).reduce((a, b) => Value.Add(a, b));

            Value.Backward(loss);

            // console log loss every 10% of epochs
            if (i % (epochs / 10) === 0) {
                console.log('loss',loss.data);
            }

            for (const param of this.parameters()) {
                param.data -= param.grad * lr;
            }
        }
    }
}

module.exports = {
    Value,
    Neuron,
    Layer,
    MLP
}
