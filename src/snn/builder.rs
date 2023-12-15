use std::sync::{Arc, Mutex};
use crate::neuron::Neuron;
use crate::snn::layer::Layer;
use crate::snn::network::SNN;
use crate::configuration::Configuration;
use crate::failure::{Conf, Failure, Stuck_at_0, Stuck_at_1, Transient_bit_flip};

/**
Object containing the configuration parameters describing the SNN architecture
 */
#[derive(Clone)]
pub struct SnnParams<N: Neuron, R: Configuration> {
    pub input_dimensions: usize,            /* dimension of the network input layer */
    pub neurons: Vec<Vec<N>>,               /* neurons per each layer */
    pub extra_weights: Vec<Vec<Vec<f64>>>,  /* (positive) weights between layers */
    pub intra_weights: Vec<Vec<Vec<f64>>>,  /* (negative) weights inside the same layer */
    pub num_layers: usize,                  /* number of layers */
    pub configuration: Vec<R>,              /* configuration for each layer */
}

/**
Object for the configuration and creation of the Spiking Neural Network.
It allows to configure the network by passing all the network parameters to one function.
 */
#[derive(Clone)]
pub struct SnnBuilder<N: Neuron, R:Configuration> {
    params: SnnParams<N,R>
}

impl<N: Neuron + Clone, R:Configuration + Clone> SnnBuilder<N,R> {
    pub fn new(input_dimension: usize) -> Self {
        let  config :Vec<R>= Vec::<R>::new();
        Self {
            params: SnnParams {
                input_dimensions: input_dimension,
                neurons: vec![],
                extra_weights: vec![],
                intra_weights: vec![],
                num_layers: 0,
                configuration: config,
            }
        }
    }

    pub fn get_params(&self) -> SnnParams<N,R> {
        self.params.clone()
    }

    /**
    It does all the checks related to the network's intra weights.
    - It checks that the number of neurons is equal to the number of rows of the intra weights matrix
    - It checks that the number of neurons is equal to the number of columns of the intra weights matrix
    - It checks that the intra weights' values are all negative and in the range [-1, 0]
     */
    fn check_intra_weights(&self, num_neurons: usize, weights: &Vec<Vec<f64>>)  {
        if num_neurons != weights.len() {
            panic!("The number of neurons must be equal to the number of rows of the intra weights matrix");
        }
        for row in weights {
            if num_neurons != row.len() {
                panic!("The number of neurons must be equal to the number of columns of the intra weights matrix");
            }
            for weight in row {
                if *weight > 0.0 {
                    panic!("The intra weights must be negative");
                }
            }
        }
    }

    /**
    It does all the checks related to the network's extra weights.
    - It checks that the number of neurons is equal to the number of rows of the extra weights matrix
    - It checks that the number of neurons is equal to the number of columns of the extra weights matrix
    - It checks that the number of columns of the extra weights matrix is equal to the number of neurons of the previous layer
    - It checks that the extra weights' values are all positive and in the range [0, 1]
     */

    fn check_weights(&self, num_neurons: usize, weights: &Vec<Vec<f64>>) {
        if num_neurons != weights.len() {
            panic!("The number of neurons must be equal to the number of rows of the weights matrix");
        }

        for row in weights {
            if self.params.num_layers == 0 {
                if row.len()!= self.params.input_dimensions {
                    panic!("The number of neurons must be equal to the number of columns of the weights matrix");
                }
            }
            else {
                if row.len() != self.params.neurons[self.params.num_layers - 1].len() {
                    panic!("The number of columns in the weights matrix must be equal to the number of neurons of the previous layer");
                }
            }
            for weight in row {
                if *weight < 0.0 {
                    panic!("The weights must be positive");
                }
            }
        }
    }

    /**
    It adds a new layer to the network specifying all the parameters requested.
     */
    pub fn add_layer( self, neurons: Vec<N>, extra_weights: Vec<Vec<f64>>, intra_weights: Vec<Vec<f64>>, configuration: R) -> Self {
        self.check_intra_weights(neurons.len(),&intra_weights);
        self.check_weights(neurons.len(),&extra_weights);

        let mut params = self.params;

        params.neurons.push(neurons);
        params.extra_weights.push(extra_weights);
        params.intra_weights.push(intra_weights);
        params.num_layers += 1;
        params.configuration.push(configuration);
        Self { params }
    }

    /**
    It adds a new layer to the network specifying all the parameters requested.
    - All neurons have the same parameters
     */
    pub fn add_layer_with_same_neurons( self, neuron: N, num_neurons: usize, extra_weights: Vec<Vec<f64>>, intra_weights: Vec<Vec<f64>>, configuration : R) -> Self {
        self.check_intra_weights(num_neurons,&intra_weights);
        self.check_weights(num_neurons,&extra_weights);

        let mut params = self.params;

        let mut neurons = Vec::with_capacity(num_neurons);

        for _i in 0..num_neurons {
            neurons.push(neuron.clone());
        }

        params.neurons.push(neurons);
        params.extra_weights.push(extra_weights);
        params.intra_weights.push(intra_weights);
        params.num_layers += 1;
        params.configuration.push(configuration);
        Self { params }
    }

    /**
    Create and initialize the whole dynamic Spiking Neural Network with the characteristics defined so far
    - If the network has no layers, the process panics
     */
    pub fn build(self) -> SNN<N,R> {

        if self.params.num_layers == 0 {
            panic!("The network must have at least one layer");
        }

        if  self.params.neurons.len() != self.params.extra_weights.len() ||
            self.params.neurons.len() != self.params.intra_weights.len() {
            /* it must not happen */
            panic!("Error: the number of neurons layers does not correspond to the number of weights layers")
        }

        let mut layers: Vec<Arc<Mutex<Layer<N,R>>>> = Vec::new();
        let mut neurons_iter = self.params.neurons.into_iter();
        let mut extra_weights_iter = self.params.extra_weights.into_iter();
        let mut intra_weights_iter = self.params.intra_weights.into_iter();
        let mut configuration_iter = self.params.configuration.into_iter();
        /* retrieve the Neurons, the extra weights and the intra weights for each layer */
        while let Some(layer_neurons) = neurons_iter.next() {
            let layer_extra_weights = extra_weights_iter.next().unwrap();
            let layer_intra_weights = intra_weights_iter.next().unwrap();
            let configuration = configuration_iter.next().unwrap().clone();
            /* create and save the new layer */
            let new_layer = Layer::new(layer_neurons, layer_extra_weights, layer_intra_weights,configuration);
            layers.push(Arc::new(Mutex::new(new_layer)));
        }

        SNN::new(layers)
    }
}