use spiking_neural_network::lif_neuron::LifNeuron;
use spiking_neural_network::layer::Layer;

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

#[test]
fn verify_get_number_neurons() {
    let l = create_layer();

    assert_eq!(l.get_number_neurons(), 3);
}

#[test]
fn verify_get_neurons() {
    let l = create_layer();

    let n = LifNeuron::new(0.1, 0.2, 0.3, 0.4, 0.5);
    let neurons = vec![n; 3];

    assert_eq!(l.get_neurons(), neurons);
    assert_eq!(l.get_neurons().len(), 3);
}

#[test]
fn verify_get_weights() {
    let weights = vec![
        vec![0.1, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
    ];

    let l = create_layer();

    assert_eq!(l.get_weights(), weights);
    assert_eq!(l.get_weights().len(), 3);
}

#[test]
fn verify_get_intra_weights() {
    let l = create_layer();

    let intra_weights = vec![
        vec![0.0, -0.1, -0.15],
        vec![-0.05, 0.0, -0.1],
        vec![-0.15, -0.1, 0.0],
    ];

    assert_eq!(l.get_intra_weights(), intra_weights);
    assert_eq!(l.get_intra_weights().len(), 3);
}



