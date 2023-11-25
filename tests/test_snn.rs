use std::sync::{Arc, Mutex};
use spiking_neural_network::lif_neuron::LifNeuron;
use spiking_neural_network::layer::Layer;
use spiking_neural_network::spiking_n_n::SNN;

fn create_layer() -> Layer<LifNeuron> {
    let n = LifNeuron::new(0.1, 0.2, 0.3, 0.4, 0.5);
    let neurons = vec![n; 3];

    let weights = vec![
        vec![0.1, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
    ];

    let intra_weights = vec![
        vec![0.0, -0.1, -0.15],
        vec![-0.05, 0.0, -0.1],
        vec![-0.15, -0.1, 0.0],
    ];

    let l = Layer::new(neurons, weights, intra_weights);
    l
}
fn create_snn() -> SNN<LifNeuron> {
    let layer = create_layer();
    let mut layers: Vec<Arc<Mutex<Layer<LifNeuron>>>> = Vec::new();

    layers.push(Arc::new(Mutex::new(layer.clone())));
    layers.push(Arc::new(Mutex::new(layer.clone())));
    layers.push(Arc::new(Mutex::new(layer.clone())));

    SNN::new(layers)
}

#[test]
fn verify_get_number_layers() {
    let n = create_snn();
    assert_eq!(n.get_number_layers(), 3);
}

// #[test]
// fn verify_get_input_layer_dim(){
//     let n = create_snn();
//     assert_eq!(n.get_input_layer_dim(), 2);
// }