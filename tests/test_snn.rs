use std::sync::{Arc, Mutex};
use spiking_neural_network::lif_neuron::LifNeuron;
use spiking_neural_network::layer::Layer;
use spiking_neural_network::network::SNN;
use spiking_neural_network::failure::{Components, Conf, Failure, StuckAt0};

fn create_layer() -> Layer<LifNeuron, Conf> {
    let n = LifNeuron::new(0.76, 0.33, 0.14, 0.4, 0.05);
    let n2 = LifNeuron::new(0.88, 0.3, 0.1, 0.2, 0.05);
    let n3 = LifNeuron::new(0.9, 0.2, 0.05, 0.1, 0.05);
    let failure = Failure::StuckAt0(StuckAt0::new(0));
    let configuration = Conf::new(vec![Components::VMem, Components::VTh], failure, 1);
    let neurons = vec![n, n2, n3];

    let weights = vec![
        vec![0.1, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
    ];

    let intra_weights = vec![
        vec![0.0, -0.5, -0.15],
        vec![-0.05, 0.0, -0.2],
        vec![-0.35, -0.1, 0.0],
    ];

    let l = Layer::new(neurons, weights, intra_weights, configuration);
    l
}

fn create_snn() -> SNN<LifNeuron, Conf> {
    let layer = create_layer();
    let mut layers: Vec<Arc<Mutex<Layer<LifNeuron, Conf>>>> = Vec::new();

    layers.push(Arc::new(Mutex::new(layer.clone())));
    layers.push(Arc::new(Mutex::new(layer.clone())));
    layers.push(Arc::new(Mutex::new(layer.clone())));

    SNN::new(layers)
}

fn create_snn_1_layer() -> SNN<LifNeuron, Conf> {
    let layer = create_layer();
    let mut layers: Vec<Arc<Mutex<Layer<LifNeuron, Conf>>>> = Vec::new();

    layers.push(Arc::new(Mutex::new(layer.clone())));
    SNN::new(layers)
}

#[test]
fn verify_get_number_layers() {
    let n = create_snn();
    assert_eq!(n.get_number_layers(), 3);
}

#[test]
fn verify_get_input_layer_dim() {
    let n = create_snn();
    assert_eq!(n.get_input_layer_one_dim(), 2);
}

#[test]
fn verify_get_output_layer_dim() {
    let n = create_snn();
    assert_eq!(n.get_output_last_layer_dim(), 3);
}

#[test]
fn verify_output_last_layer_dim() {
    let mut n = create_snn_1_layer();
    let input_spikes: Vec<Vec<u8>> = vec![
        vec![0, 1, 1], /* 1st neuron input train of spikes */
        vec![1, 0, 1], /* 2nd neuron input train of spikes */
    ];
    assert_eq!(n.process(&input_spikes), vec![
        vec![0, 0, 0], /* 1st neuron input train of spikes */
        vec![0, 1, 0], /* 2nd neuron input train of spikes */
        vec![0, 1, 1], /* 3rd neuron input train of spikes */
    ]);
}

#[test]
fn verify_output_first_layer_dim() {
    let n = create_snn();
    assert_eq!(n.get_output_last_layer_dim(), 3);
}