use spiking_neural_network::configuration::Configuration;
use spiking_neural_network::failure::{Conf, Failure, Stuck_at_0, Stuck_at_1, Transient_bit_flip};

use spiking_neural_network::lif_neuron::LifNeuron;
use spiking_neural_network::layer::Layer;
use spiking_neural_network::neuron::Neuron;

fn create_layer() -> Layer<LifNeuron, Conf> {
    let n = LifNeuron::new(0.9, 0.33, 0.14, 0.4);
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
    let failure = Failure::TransientBitFlip(Transient_bit_flip::new(15));
    let configuration = Conf::new(vec!["v_th".to_string()],failure);
    let l = Layer::new(neurons, weights, intra_weights,configuration);
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

    let n = LifNeuron::new(0.9, 0.33, 0.14, 0.4);
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

#[test]
fn verify_modification_bit_v_th() {
    let l = create_layer();
    for mut lif in l.get_neurons() {
        if l.get_configuration().get_vec_components().contains(&"v_th".to_string())  {
            let fail = l.get_configuration().get_failure();
            match fail {
                Failure::StuckAt0(stuck_at_0) => {
                    let mut vec_byte_original : Vec<u8> = Vec::new();
                    for byte in lif.get_v_th().to_ne_bytes() {
                        vec_byte_original.push(byte);
                    }
                    lif.modify_bits(&mut vec_byte_original,stuck_at_0.get_position(), stuck_at_0.get_valore());
                    lif.set_v_th(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                    assert_eq!(lif.get_v_th(), 0.65);//usare posizione 13
                }
                Failure::StuckAt1(stuck_at_1) => {
                    let mut vec_byte_original : Vec<u8> = Vec::new();
                    for byte in lif.get_v_th().to_ne_bytes() {
                        vec_byte_original.push(byte);
                    }
                    lif.modify_bits(&mut vec_byte_original,stuck_at_1.get_position(),stuck_at_1.get_valore());
                    lif.set_v_th(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                    assert_eq!(lif.get_v_th(), 0.9625);//usare posizione 15
                }
                Failure::TransientBitFlip(mut transient_bit) => {
                    if !transient_bit.get_bit_changed() {
                        let mut vec_byte_original : Vec<u8> = Vec::new();
                        let mut byte_original = 0u8;
                        let mut count = 0;
                        for byte in lif.get_v_th().to_ne_bytes() {
                            vec_byte_original.push(byte);
                            if count == transient_bit.get_position()/8 {
                                byte_original = byte;
                            }
                            count += 1;
                        }
                        let valor_bit =  (byte_original >> transient_bit.get_position()%8) & 1;
                        lif.modify_bits(&mut vec_byte_original, transient_bit.get_position(), valor_bit);
                        transient_bit.set_bit_changed(true);
                        lif.set_v_th(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                        assert_eq!(lif.get_v_th(), 0.9625);//usare posizione 15
                    }
                }
                Failure::None => {}
            }
        }
    }
}