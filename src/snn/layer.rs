use crate::snn::neuron::Neuron;

#[derive(Debug)]
pub struct Layer<N: Neuron + Clone + Send + 'static> {
    neurons: Vec<N>,
    weights: Vec<Vec<f64>>,
    intra_weights: Vec<Vec<f64>>,
    prev_spikes: Vec<u8>
}

impl<N: Neuron + Clone + Send + 'static> Layer<N> {
    pub fn new(
        neurons: Vec<N>,
        weights: Vec<Vec<f64>>,
        intra_weights: Vec<Vec<f64>>,
    ) -> Self {
        let num_neurons = neurons.len();
        Self {
            neurons,
            weights,
            intra_weights,
            prev_spikes: vec![0; num_neurons]
        }
    }

    /* Getters  */
    pub fn get_number_neurons(&self) -> usize {
        self.neurons.len()
    }

    pub fn get_neurons(&self) -> Vec<N> { self.neurons.clone() }

    pub fn get_weights(&self) -> Vec<Vec<f64>> {
        self.weights.clone()
    }

    pub fn get_intra_weights(&self) -> Vec<Vec<f64>> {
        self.intra_weights.clone()
    }

    pub fn init(&mut self) {
        self.prev_spikes.clear();    /* reset prev_spikes */
        self.neurons.iter_mut().for_each(|neuron| neuron.init());  /* reset neurons */
    }
}