# Group03
# Spiking Neural Network
- [Description](#description)
- [Group Members](#group-members)
- [Dependencies](#dependencies)
- [Repos Structure](#repos-structure)
- [Organization](#organization)
- [Main Structures](#main-structures)
- [Main Methods](#main-methods)
- [Usage Examples](#usage-examples)

## Description
This is a `Rust library` aiming to model a `Spiking Neural Network`. It is carried out for the `group03` related to the "Programmazione di Sistema" course of the Politecnico di Torino, a.y. 2022-2023.

The library provides support for implementing `Spiking Neural Network` models to be executed on spike datasets. It does not support the network's training phase but focuses solely on execution. Additionally, it allows for the introduction of a fault by simulating a `fault injection`, which can be exclusively `Stuck_At_0`, `Stuck_At_1`, or `Transient_Bit_Flip`. The primary goal is to assess the network's robustness by observing its behavior when simulating one of these faults.

## Group members
- Piergiuseppe Siragusa
- Vittorio Tabarè
- Giorgio Ferraro

## Dependencies
- `Rust` (version 1.75.0)
- `Cargo` (version 1.75.0)
- `rand`  (version 0.8.5)
- `bit`  (version 0.1)
- `zip`  (version 0.6.6)

No other particular dependencies are required.

## Repos Structure
The repository is structured as follows:
- `src/` contains the **source code** of the library
  + `snn/` contains the specific models implementations (`Lif Neuron`), the SNN generic implementation, the builder objects for the SNN.
- `tests/` contains the tests of the library

## Organization
The library is organized as follows:

- ### Builder
  The `Builder` module allows you to actually create the structure of the network with the corresponding layers, neurons per each layer, the corresponding weights between them and between neurons of the same layer and the fault. The structures of the network are allocated on the **Heap** (**Fitting with large networks**).

- ### Network
  The `Network` module allows you to actually execute the network on a given input. Receives the input as a dynamic vector of spikes and produces as output a dynamic vector of spikes too. The correctness of the input can be checked only at *run time*.

## Main structures
The library provides the following main structures:

- `LifNeuron` represents a neuron for the `Leaky Integrate and Fire` model, it can be used to build a `Layer` of neurons. 

```rust
pub struct LifNeuron {
    /* const fields */
    v_th:    f64,       /* threshold potential */
    v_rest:  f64,       /* resting potential */
    v_reset: f64,       /* reset potential */
    tau:     f64, 
    dt:      f64,       /* time interval between two consecutive instants */
    /* mutable fields */
    v_mem:   f64,       /* membrane potential */
    ts:      u64,       /* last instant in which receiving at least one spike */
}
```
For more information about the `Leaky Integrate and Fire` model, see [here](https://www.nature.com/articles/s41598-017-07418-y).

- `Layer` represents a layer of neurons, it can be used to build the `Network` of layers.
```rust
pub struct Layer<N: Neuron + Clone + Send + 'static, R: Configuration + Clone + Send + 'static> {
    neurons: Vec<N>,                /* neurons of the layer */
    weights: Vec<Vec<f64>>,         /* weights between the neurons of this layer and the previous one */
    intra_weights: Vec<Vec<f64>>,   /* weights between the neurons of this layer */
    prev_output_spikes: Vec<u8>     /* output spikes of the previous instant */
    configuration: R,               /* configuration for each layer */
}
```

- `SpikeEvent` represents an event of a neurons layer firing at a certain instant of time. 
It wraps the spikes flowing through the network
```rust
pub struct SpikeEvent {
    ts: u64,            /* discrete time instant */
    spikes: Vec<u8>,    /* vector of spikes in that instant (a 1/0 for each input neuron)  */
}
```

- `Conf` represent the configuration of the fault in the `Layer`.
```rust
pub struct Conf {
    components: Vec<Components>,    /* Vec of components to inject fault */
    failure: Failure,               /* kind of failure */
    index_neuron: usize,            /* index of neuron */
    done: bool,                     /* a boolean used to optimize code*/
}
```

- `Components` represent components that could break in `Conf`.
```rust
pub enum Components {
    /* List of possible fault components of LifNeuron */
    VTh, VRest, VReset,
    Tau, VMem, Ts, Dt,

    /* List of possible fault components of Layers*/
    Weights, IntraWeights,
    PrevSpikes,

    None,
}
```

- `Failure` represent the generic failure that could be applied in `Conf`, each one of these is a struct.
```rust
pub enum Failure {
    StuckAt0(StuckAt0),
    StuckAt1(StuckAt1),
    TransientBitFlip(TransientBitFlip),
    None,
}
```

- `StuckAt1`, `StuckAt0` and `TransientBitFlip` represent the failure `Failure`.
```rust
pub struct StuckAt1 {
    position: usize,
    value: u8,
}

pub struct StuckAt0 {
    position: usize,
    value: u8,
}

pub struct TransientBitFlip {
    position: usize,
    bit_changed: bool,
}
```

- `Network` represents a `Spiking Neural Network` composed by a vector of `Layer`s.
```rust
pub struct SNN<N: Neuron + Clone + 'static, R: Configuration + Clone + Send + 'static> {
    layers: Vec<Arc<Mutex<Layer<N>>>>
}
```

- `Builder` represents the builder for a `Network`
```rust
pub struct SnnBuilder<N: Neuron, R: Configuration> {
    params: SnnParams<N, R>,
}

pub struct SnnParams<N: Neuron, R: Configuration> {
    pub input_dimensions: usize,            /* dimension of the network input layer */
    pub neurons: Vec<Vec<N>>,               /* neurons per each layer */
    pub extra_weights: Vec<Vec<Vec<f64>>>,  /* (positive) weights between layers */
    pub intra_weights: Vec<Vec<Vec<f64>>>,  /* (negative) weights inside the same layer */
    pub num_layers: usize,                  /* number of layers */
    pub configuration: Vec<R>,              /* configuration for each layer */
}
```

## Main methods
The library provides the following main methods:
 - ### Builder Methods
 
- #### `SnnBuilder` methods:
   - **new()** method:
   
     ```rust
     pub fn new(input_dimension: usize) -> Self 
        ```
     
     creates a new `SnnBuilder`
   - **add_layer()** method:
   
     ```rust
     pub fn add_layer(self, neurons: Vec<N>, extra_weights: Vec<Vec<f64>>, intra_weights: Vec<Vec<f64>>, configuration: R) -> Self 
     ```
     
     adds a new `layer` to the SNN with the given `neurons`, `weights`, `intra_weights` and `configuration` passed as parameters

   - **build()** method:
   
     ```rust
     pub fn build(self) -> SNN<N, R>
     ```
     
     builds the `SNN` from the information collected so far by the `SnnBuilder`

 - ### Network Methods
   - #### `Network` method:
        - process() method:
        
            ```rust
             pub fn process(&mut self, spikes: &Vec<Vec<u8>>) -> Vec<Vec<u8>> 
            ```
          
            processes the input spikes passed as parameter and returns the output spikes of the network
   


## Usage examples
The following example shows how to *dynamically* create a `Spiking Neural Network` with 2 input neurons and 
a two layers of 3 `LifNeuron`s and failure using the `SnnBuilder`, and how to execute it on a given input of 3 instants per neuron.

```rust
use spiking_neural_network::snn::builder::SnnBuilder;
use spiking_neural_network::lif_neuron::LifNeuron;
use spiking_neural_network::failure::*;
use spiking_neural_network::snn::configuration::Configuration;

    let mut snn = SnnBuilder::<LifNeuron,Conf>::new(2)     /* input dimension of 2 */
        .add_layer(     /* first layer*/
            vec![   /* 3 LIF neurons */
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)
            ], 
            vec![   /* weights */
                vec![0.1, 0.2],     /* weigths from input layer to the 1st neuron */
                vec![0.3, 0.4],     /* weigths from input layer to the 2nd neuron */
                vec![0.5, 0.6]      /* weigths from input layer to the 3rd neuron */
            ], 
            vec![   /* intra-weights */
                vec![0.0, -0.1, -0.15],     /* weigths from the same layer to the 1st neuron */
                vec![-0.05, 0.0, -0.1],     /* weigths from the same layer to the 2nd neuron */
                vec![-0.15, -0.1, 0.0]      /* weigths from the same layer to the 3rd neuron */
            ],
            Conf::new(vec![], Failure::None, 0), /* no failure */
        ).add_layer(    /* second layer */
            vec![   /* 2 LIF neurons */
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)
            ],
            vec![   /* weights */
                vec![0.11, 0.29, 0.3],      /* weigths from previous layer to the 1st neuron */
                vec![0.33, 0.41, 0.57]      /* weigths from previous layer to the 2nd neuron */
            ],
            vec![   /* intra-weights */
                vec![0.0, -0.25],       /* weigths from the same layer to the 1st neuron */
                vec![-0.10, 0.0]        /* weigths from the same layer to the 2nd neuron */
            ],
            Conf::new(vec![Components::VMem], Failure::StuckAt0(StuckAt0::new(random_bit)), 5) /* stuck_at_0 failure */
        ).build();  /* create the network */
    
    /* process input spikes */
    let output_spikes = snn.process(&vec![
        vec![1,0,1],    /* 1st neuron input */
        vec![0,0,1]     /* 2nd neuron input */
    ]);
```
