use spiking_neural_network::failure::{Components, Conf, Failure, StuckAt0, StuckAt1};
use spiking_neural_network::lif_neuron::LifNeuron;
use spiking_neural_network::layer::{Layer, modify_bits};
use spiking_neural_network::neuron::Neuron;

fn create_layer(configuration: Conf) -> Layer<LifNeuron, Conf> {
    let n = LifNeuron::new(0.9, 0.33, 0.14, 0.4, 0.05);
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

    let l = Layer::new(neurons, weights, intra_weights, configuration);
    l
}

#[test]
fn verify_get_number_neurons() {
    let l = create_layer(Conf::new(vec![], Failure::None, 0));
    assert_eq!(l.get_number_neurons(), 3);
}

#[test]
fn verify_get_neurons() {
    let l = create_layer(Conf::new(vec![], Failure::None, 0));
    let n = LifNeuron::new(0.9, 0.33, 0.14, 0.4, 0.05);
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

    let l = create_layer(Conf::new(vec![], Failure::None, 0));

    assert_eq!(l.get_weights(), weights);
    assert_eq!(l.get_weights().len(), 3);
}

#[test]
fn verify_get_intra_weights() {
    let l = create_layer(Conf::new(vec![], Failure::None, 0));

    let intra_weights = vec![
        vec![0.0, -0.1, -0.15],
        vec![-0.05, 0.0, -0.1],
        vec![-0.15, -0.1, 0.0],
    ];

    assert_eq!(l.get_intra_weights(), intra_weights);
    assert_eq!(l.get_intra_weights().len(), 3);
}

#[test]
fn verify_modification_bit_v_th_stuck0() {
    let failure = Failure::StuckAt0(StuckAt0::new(13));
    let configuration = Conf::new(vec![Components::VMem], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();

    let vec_byte_original: Vec<_> = n.get_v_th().to_ne_bytes().to_vec().iter().rev().cloned().collect();
    let vec_byte_mod = modify_bits(failure, vec_byte_original.clone());
    n.set_v_th(f64::from_ne_bytes(vec_byte_mod.as_slice().try_into().unwrap()));
    assert_eq!(n.get_v_th(), 0.65);//use position 13
}

#[test]
fn verify_modification_bit_v_th_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(15));
    let configuration = Conf::new(vec![Components::VTh], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();

    let vec_byte_original: Vec<_> = n.get_v_th().to_ne_bytes().to_vec().iter().rev().cloned().collect();
    let vec_byte_mod = modify_bits(failure, vec_byte_original.clone());
    n.set_v_th(f64::from_ne_bytes(vec_byte_mod.as_slice().try_into().unwrap()));
    assert_eq!(n.get_v_th(), 0.9625);//use position 15
}

/*
#[test]
fn verify_modification_bit_v_th_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(15));
    let configuration = Conf::new(vec![Components::VTh], failure, 1);
    let l = create_layer(configuration.clone());

    match configuration.get_failure() {
        Failure::TransientBitFlip(mut transient_bit) => {
            if !transient_bit.get_bit_changed() {
                let mut neuron = l.get_neurons();
                let n = neuron.get_mut(0).unwrap();
                let mut vec_byte_original: Vec<_> = n.get_v_th().to_ne_bytes().iter().cloned().collect();
                let byte_original = vec_byte_original.get(transient_bit.get_position() as usize / 8).cloned().unwrap_or(0u8);
                let valor_bit = (byte_original >> (transient_bit.get_position() % 8)) & 1;
                modify_bits(&mut vec_byte_original, transient_bit.get_position() as u8 % 64u8, valor_bit);
                transient_bit.set_bit_changed(true);
                n.set_v_th(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                assert_eq!(n.get_v_th(), 0.9625);//use position 15
            }
        }
        _ => {}
    }
}

#[test]
fn verify_modification_bit_v_rest_stuck0() {
    let failure = Failure::StuckAt0(StuckAt0::new(13));
    let configuration = Conf::new(vec![Components::VRest], failure, 1);
    let l = create_layer(configuration.clone());

    match configuration.get_failure() {
        Failure::StuckAt0(s) => {
            let mut neuron = l.get_neurons();
            let n = neuron.get_mut(0).unwrap();
            let mut vec_byte_original: Vec<_> = n.get_v_rest().to_ne_bytes().iter().cloned().collect();
            modify_bits(&mut vec_byte_original, s.get_position() as u8 % 64u8, s.get_value());
            n.set_v_rest(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
            assert_eq!(n.get_v_rest(), 0.33);//use position 13
        }
        _ => {}
    }
}

#[test]
fn verify_modification_bit_v_rest_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(15));
    let configuration = Conf::new(vec![Components::VRest], failure, 1);
    let l = create_layer(configuration.clone());

    match configuration.get_failure() {
        Failure::StuckAt1(s) => {
            let mut neuron = l.get_neurons();
            let n = neuron.get_mut(0).unwrap();
            let mut vec_byte_original: Vec<_> = n.get_v_rest().to_ne_bytes().iter().cloned().collect();
            modify_bits(&mut vec_byte_original, s.get_position() as u8 % 64u8, s.get_value());
            n.set_v_rest(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
            assert_eq!(n.get_v_rest(), 0.36125);//use position 15
        }
        _ => {}
    }
}

#[test]
fn verify_modification_bit_v_rest_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(15));
    let configuration = Conf::new(vec![Components::VRest], failure, 1);
    let l = create_layer(configuration.clone());

    match configuration.get_failure() {
        Failure::TransientBitFlip(mut transient_bit) => {
            if !transient_bit.get_bit_changed() {
                let mut neuron = l.get_neurons();
                let n = neuron.get_mut(0).unwrap();
                let mut vec_byte_original: Vec<_> = n.get_v_rest().to_ne_bytes().iter().cloned().collect();
                let byte_original = vec_byte_original.get(transient_bit.get_position() as usize / 8).cloned().unwrap_or(0u8);
                let valor_bit = (byte_original >> (transient_bit.get_position() % 8)) & 1;
                modify_bits(&mut vec_byte_original, transient_bit.get_position() as u8 % 64u8, valor_bit);
                transient_bit.set_bit_changed(true);
                n.set_v_rest(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                assert_eq!(n.get_v_rest(), 0.36125);//use position 15
            }
        }
        _ => {}
    }
}

#[test]
fn verify_modification_bit_v_reset_stuck0() {
    let failure = Failure::StuckAt0(StuckAt0::new(13));
    let configuration = Conf::new(vec![Components::VRest], failure, 1);
    let l = create_layer(configuration.clone());

    match configuration.get_failure() {
        Failure::StuckAt0(s) => {
            let mut neuron = l.get_neurons();
            let n = neuron.get_mut(0).unwrap();
            let mut vec_byte_original: Vec<_> = n.get_v_reset().to_ne_bytes().iter().cloned().collect();
            modify_bits(&mut vec_byte_original, s.get_position() as u8 % 64u8, s.get_value());
            n.set_v_reset(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
            assert_eq!(n.get_v_reset(), 0.14);//use position 13
        }
        _ => {}
    }
}

#[test]
fn verify_modification_bit_v_reset_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(15));
    let configuration = Conf::new(vec![Components::VRest], failure, 1);
    let l = create_layer(configuration.clone());

    match configuration.get_failure() {
        Failure::StuckAt1(s) => {
            let mut neuron = l.get_neurons();
            let n = neuron.get_mut(0).unwrap();
            let mut vec_byte_original: Vec<_> = n.get_v_reset().to_ne_bytes().iter().cloned().collect();
            modify_bits(&mut vec_byte_original, s.get_position() as u8 % 64u8, s.get_value());
            n.set_v_reset(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
            assert_eq!(n.get_v_reset(), 0.155625);//use position 15
        }
        _ => {}
    }
}

#[test]
fn verify_modification_bit_v_reset_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(15));
    let configuration = Conf::new(vec![Components::VRest], failure, 1);
    let l = create_layer(configuration.clone());

    match configuration.get_failure() {
        Failure::TransientBitFlip(mut transient_bit) => {
            if !transient_bit.get_bit_changed() {
                let mut neuron = l.get_neurons();
                let n = neuron.get_mut(0).unwrap();
                let mut vec_byte_original: Vec<_> = n.get_v_reset().to_ne_bytes().iter().cloned().collect();
                let byte_original = vec_byte_original.get(transient_bit.get_position() as usize / 8).cloned().unwrap_or(0u8);
                let valor_bit = (byte_original >> (transient_bit.get_position() % 8)) & 1;
                modify_bits(&mut vec_byte_original, transient_bit.get_position() as u8 % 64u8, valor_bit);
                transient_bit.set_bit_changed(true);
                n.set_v_reset(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                assert_eq!(n.get_v_reset(), 0.14);//use position 15
            }
        }
        _ => {}
    }
}

#[test]
fn verify_modification_bit_tau_stuck0() {
    let failure = Failure::StuckAt0(StuckAt0::new(13));
    let configuration = Conf::new(vec![Components::Tau], failure, 1);
    let l = create_layer(configuration.clone());
//tau = 0.4 64-13 = 51
    match configuration.get_failure() {
        Failure::StuckAt0(s) => {
            let mut neuron = l.get_neurons();
            let n = neuron.get_mut(0).unwrap();
            let mut vec_byte_original: Vec<_> = n.get_tau().to_ne_bytes().iter().cloned().collect();
            modify_bits(&mut vec_byte_original, s.get_position() as u8 % 64u8, s.get_value());
            n.set_tau(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
            assert_eq!(n.get_tau(), 0.275);//use position 13
        }
        _ => {}
    }
}

#[test]
fn verify_modification_bit_tau_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(15));
    let configuration = Conf::new(vec![Components::Tau], failure, 1);
    let l = create_layer(configuration.clone());

    match configuration.get_failure() {
        Failure::StuckAt1(s) => {
            let mut neuron = l.get_neurons();
            let n = neuron.get_mut(0).unwrap();
            let mut vec_byte_original: Vec<_> = n.get_tau().to_ne_bytes().iter().cloned().collect();
            modify_bits(&mut vec_byte_original, s.get_position() as u8 % 64u8, s.get_value());
            n.set_tau(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
            assert_eq!(n.get_tau(), 0.43125);//use position 15
        }
        _ => {}
    }
}

#[test]
fn verify_modification_bit_tau_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(15));
    let configuration = Conf::new(vec![Components::Tau], failure, 1);
    let l = create_layer(configuration.clone());

    match configuration.get_failure() {
        Failure::TransientBitFlip(mut transient_bit) => {
            if !transient_bit.get_bit_changed() {
                let mut neuron = l.get_neurons();
                let n = neuron.get_mut(0).unwrap();
                let mut vec_byte_original: Vec<_> = n.get_tau().to_ne_bytes().iter().cloned().collect();
                let byte_original = vec_byte_original.get(transient_bit.get_position() as usize / 8).cloned().unwrap_or(0u8);
                let valor_bit = (byte_original >> (transient_bit.get_position() % 8)) & 1;
                modify_bits(&mut vec_byte_original, transient_bit.get_position() as u8 % 64u8, valor_bit);
                transient_bit.set_bit_changed(true);
                n.set_tau(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                assert_eq!(n.get_tau(), 0.43125);//use position 15
            }
        }
        _ => {}
    }
}


#[test]
fn verify_modification_bit_v_mem_stuck0() {
    let failure = Failure::StuckAt0(StuckAt0::new(13));
    let configuration = Conf::new(vec![Components::VMem], failure, 1);
    let l = create_layer(configuration.clone());

    match configuration.get_failure() {
        Failure::StuckAt0(s) => {
            let mut neuron = l.get_neurons();
            let n = neuron.get_mut(0).unwrap();
            let mut vec_byte_original: Vec<_> = n.get_v_mem().to_ne_bytes().iter().cloned().collect();
            modify_bits(&mut vec_byte_original, s.get_position() as u8 % 64u8, s.get_value());
            n.set_v_mem(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
            assert_eq!(n.get_v_mem(), 0.33);//use position 13
        }
        _ => {}
    }
}

#[test]
fn verify_modification_bit_v_mem_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(15));
    let configuration = Conf::new(vec![Components::VMem], failure, 1);
    let l = create_layer(configuration.clone());

    match configuration.get_failure() {
        Failure::StuckAt1(s) => {
            let mut neuron = l.get_neurons();
            let n = neuron.get_mut(0).unwrap();
            let mut vec_byte_original: Vec<_> = n.get_v_mem().to_ne_bytes().iter().cloned().collect();
            modify_bits(&mut vec_byte_original, s.get_position() as u8 % 64u8, s.get_value());
            n.set_v_mem(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
            assert_eq!(n.get_v_mem(), 0.36125);//use position 15
        }
        _ => {}
    }
}

#[test]
fn verify_modification_bit_v_mem_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(15));
    let configuration = Conf::new(vec![Components::VMem], failure, 1);
    let l = create_layer(configuration.clone());

    match configuration.get_failure() {
        Failure::TransientBitFlip(mut transient_bit) => {
            if !transient_bit.get_bit_changed() {
                let mut neuron = l.get_neurons();
                let n = neuron.get_mut(0).unwrap();
                let mut vec_byte_original: Vec<_> = n.get_v_mem().to_ne_bytes().iter().cloned().collect();
                let byte_original = vec_byte_original.get(transient_bit.get_position() as usize / 8).cloned().unwrap_or(0u8);
                let valor_bit = (byte_original >> (transient_bit.get_position() % 8)) & 1;
                modify_bits(&mut vec_byte_original, transient_bit.get_position() as u8 % 64u8, valor_bit);
                transient_bit.set_bit_changed(true);
                n.set_v_mem(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                assert_eq!(n.get_v_mem(), 0.36125);//use position 15
            }
        }
        _ => {}
    }
}


#[test]
fn verify_modification_bit_dt_stuck0() {
    let failure = Failure::StuckAt0(StuckAt0::new(13));
    let configuration = Conf::new(vec![Components::Dt], failure, 1);
    let l = create_layer(configuration.clone());

    match configuration.get_failure() {
        Failure::StuckAt0(s) => {
            let mut neuron = l.get_neurons();
            let n = neuron.get_mut(0).unwrap();
            let mut vec_byte_original: Vec<_> = n.get_dt().to_ne_bytes().iter().cloned().collect();
            modify_bits(&mut vec_byte_original, s.get_position() as u8 % 64u8, s.get_value());
            n.set_dt(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
            assert_eq!(n.get_dt(), 0.034375);//use position 13
        }
        _ => {}
    }
}

#[test]
fn verify_modification_bit_dt_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(15));
    let configuration = Conf::new(vec![Components::Dt], failure, 1);
    let l = create_layer(configuration.clone());

    match configuration.get_failure() {
        Failure::StuckAt1(s) => {
            let mut neuron = l.get_neurons();
            let n = neuron.get_mut(0).unwrap();
            let mut vec_byte_original: Vec<_> = n.get_dt().to_ne_bytes().iter().cloned().collect();
            modify_bits(&mut vec_byte_original, s.get_position() as u8 % 64u8, s.get_value());
            n.set_dt(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
            assert_eq!(n.get_dt(), 0.05390625);//use position 15
        }
        _ => {}
    }
}

#[test]
fn verify_modification_bit_dt_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(15));
    let configuration = Conf::new(vec![Components::Dt], failure, 1);
    let l = create_layer(configuration.clone());

    match configuration.get_failure() {
        Failure::TransientBitFlip(mut transient_bit) => {
            if !transient_bit.get_bit_changed() {
                let mut neuron = l.get_neurons();
                let n = neuron.get_mut(0).unwrap();
                let mut vec_byte_original: Vec<_> = n.get_dt().to_ne_bytes().iter().cloned().collect();
                let byte_original = vec_byte_original.get(transient_bit.get_position() as usize / 8).cloned().unwrap_or(0u8);
                let valor_bit = (byte_original >> (transient_bit.get_position() % 8)) & 1;
                modify_bits(&mut vec_byte_original, transient_bit.get_position() as u8 % 64u8, valor_bit);
                transient_bit.set_bit_changed(true);
                n.set_dt(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                assert_eq!(n.get_dt(), 0.05390625);//use position 15
            }
        }
        _ => {}
    }
}


#[test]
fn verify_modification_bit_ts_stuck0() {
    let failure = Failure::StuckAt0(StuckAt0::new(13));
    let configuration = Conf::new(vec![Components::Ts], failure, 1);
    let l = create_layer(configuration.clone());

    match configuration.get_failure() {
        Failure::StuckAt0(s) => {
            let mut neuron = l.get_neurons();
            let n = neuron.get_mut(0).unwrap();
            let mut vec_byte_original: Vec<_> = n.get_ts().to_ne_bytes().iter().cloned().collect();
            modify_bits(&mut vec_byte_original, s.get_position() as u8 % 64u8, s.get_value());
            n.set_ts(u64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
            assert_eq!(n.get_ts(), 0);//use position 13
        }
        _ => {}
    }
}

#[test]
fn verify_modification_bit_ts_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(15)); //2^(64-15) = 2^49 = 562949953421312
    let configuration = Conf::new(vec![Components::Ts], failure, 1);
    let l = create_layer(configuration.clone());

    match configuration.get_failure() {
        Failure::StuckAt1(s) => {
            let mut neuron = l.get_neurons();
            let n = neuron.get_mut(0).unwrap();
            let mut vec_byte_original: Vec<_> = n.get_ts().to_ne_bytes().iter().cloned().collect();
            modify_bits(&mut vec_byte_original, s.get_position() as u8 % 64u8, s.get_value());
            n.set_ts(u64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
            assert_eq!(n.get_ts(), 562949953421312);//use position 15
        }
        _ => {}
    }
}

#[test]
fn verify_modification_bit_ts_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(15));
    let configuration = Conf::new(vec![Components::Ts], failure, 1);
    let l = create_layer(configuration.clone());

    match configuration.get_failure() {
        Failure::TransientBitFlip(mut transient_bit) => {
            if !transient_bit.get_bit_changed() {
                let mut neuron = l.get_neurons();
                let n = neuron.get_mut(0).unwrap();
                let mut vec_byte_original: Vec<_> = n.get_ts().to_ne_bytes().iter().cloned().collect();
                let byte_original = vec_byte_original.get(transient_bit.get_position() as usize / 8).cloned().unwrap_or(0u8);
                let valor_bit = (byte_original >> (transient_bit.get_position() % 8)) & 1;
                modify_bits(&mut vec_byte_original, transient_bit.get_position() as u8 % 64u8, valor_bit);
                transient_bit.set_bit_changed(true);
                n.set_ts(u64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                assert_eq!(n.get_ts(), 0);//use position 15
            }
        }
        _ => {}
    }
}


#[test]
fn verify_modify_bit_extra_weights_stuck0() {
    let failure = Failure::StuckAt0(StuckAt0::new(65));
    let configuration = Conf::new(vec![Components::Weights], failure, 1);
    let mut l = create_layer(configuration.clone());
    match configuration.get_failure() {
        Failure::StuckAt0(s) => {
            let mut matrix = l.get_weights();
            let i = ((s.get_position() / 64) / matrix.len() as u32) as usize;
            let j = ((s.get_position() / 64) % matrix.len() as u32) as usize;
            let mut vec_byte = matrix[i][j].to_ne_bytes().iter().cloned().collect();
            modify_bits(&mut vec_byte, s.get_position() as u8 % 64u8, s.get_value());
            matrix[i][j] = f64::from_ne_bytes(vec_byte.as_slice().try_into().unwrap());
            l.set_weights(matrix);
            assert_eq!(l.get_weights(), vec![
                vec![0.1, 0.2],
                vec![0.3, 0.4],
                vec![0.5, 0.6],
            ]);//use position 65
        }
        _ => {}
    }
}

#[test]
fn verify_modify_bit_extra_weights_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(66));
    let configuration = Conf::new(vec![Components::Weights], failure, 1);
    let mut l = create_layer(configuration.clone());
    match configuration.get_failure() {
        Failure::StuckAt1(s) => {
            let mut matrix = l.get_weights();
            let i = ((s.get_position() / 64) / matrix.len() as u32) as usize;
            let j = ((s.get_position() / 64) % matrix.len() as u32) as usize;
            let mut vec_byte = matrix[i][j].to_ne_bytes().iter().cloned().collect();
            modify_bits(&mut vec_byte, s.get_position() as u8 % 64u8, s.get_value());
            matrix[i][j] = f64::from_ne_bytes(vec_byte.as_slice().try_into().unwrap());
            l.set_weights(matrix);
            assert_eq!(l.get_weights(), vec![
                vec![0.1, 3.595386269724632e307],
                vec![0.3, 0.4],
                vec![0.5, 0.6],
            ]);//use position 66
        }
        _ => {}
    }
}

#[test]
fn verify_modify_bit_extra_weights_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(65));
    let configuration = Conf::new(vec![Components::Weights], failure, 1);
    let mut l = create_layer(configuration.clone());
    match configuration.get_failure() {
        Failure::TransientBitFlip(mut t) => {
            if !t.get_bit_changed() {
                let mut matrix = l.get_weights();
                let i = ((t.get_position() / 64) / matrix.len() as u32) as usize;
                let j = ((t.get_position() / 64) % matrix.len() as u32) as usize;
                let mut vec_byte: Vec<_> = matrix[i][j].to_ne_bytes().iter().cloned().collect();

                let byte_original: u8 = vec_byte.get(t.get_position() as usize / 8).cloned().unwrap_or(0u8);
                let value_bit = (byte_original >> (t.get_position() % 8)) & 1;
                modify_bits(&mut vec_byte, t.get_position() as u8 % 64u8, value_bit);
                t.set_bit_changed(true);

                matrix[i][j] = f64::from_ne_bytes(vec_byte.as_slice().try_into().unwrap());
                l.set_weights(matrix);
                assert_eq!(l.get_weights(), vec![
                    vec![0.1, 0.2],
                    vec![0.3, 0.4],
                    vec![0.5, 0.6],
                ]);//use position 65
            }
        }
        _ => {}
    }
}


#[test]
fn verify_modify_bit_intra_weights_stuck0() {
    let failure = Failure::StuckAt0(StuckAt0::new(65));
    let configuration = Conf::new(vec![Components::Weights], failure, 1);
    let mut l = create_layer(configuration.clone());
    match configuration.get_failure() {
        Failure::StuckAt0(s) => {
            let mut matrix = l.get_intra_weights();
            let i = ((s.get_position() / 64) / matrix.len() as u32) as usize;
            let j = ((s.get_position() / 64) % matrix.len() as u32) as usize;
            let mut vec_byte = matrix[i][j].to_ne_bytes().iter().cloned().collect();
            modify_bits(&mut vec_byte, s.get_position() as u8 % 64u8, s.get_value());
            matrix[i][j] = f64::from_ne_bytes(vec_byte.as_slice().try_into().unwrap());
            l.set_intra_weights(matrix);
            assert_eq!(l.get_intra_weights(), vec![
                vec![0.0, 0.1, -0.15],
                vec![-0.05, 0.0, -0.1],
                vec![-0.15, -0.1, 0.0],
            ]);//use position 65
        }
        _ => {}
    }
}

#[test]
fn verify_modify_bit_intra_weights_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(66));
    let configuration = Conf::new(vec![Components::Weights], failure, 1);
    let mut l = create_layer(configuration.clone());
    match configuration.get_failure() {
        Failure::StuckAt1(s) => {
            let mut matrix = l.get_intra_weights();
            let i = ((s.get_position() / 64) / matrix.len() as u32) as usize;
            let j = ((s.get_position() / 64) % matrix.len() as u32) as usize;
            let mut vec_byte = matrix[i][j].to_ne_bytes().iter().cloned().collect();
            modify_bits(&mut vec_byte, s.get_position() as u8 % 64u8, s.get_value());
            matrix[i][j] = f64::from_ne_bytes(vec_byte.as_slice().try_into().unwrap());
            l.set_intra_weights(matrix);
            assert_eq!(l.get_intra_weights(), vec![
                vec![0.0, -1.797693134862316e307, -0.15],
                vec![-0.05, 0.0, -0.1],
                vec![-0.15, -0.1, 0.0],
            ]);//use position 66
        }
        _ => {}
    }
}

#[test]
fn verify_modify_bit_intra_weights_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(65));
    let configuration = Conf::new(vec![Components::Weights], failure, 1);
    let mut l = create_layer(configuration.clone());
    match configuration.get_failure() {
        Failure::TransientBitFlip(mut t) => {
            if !t.get_bit_changed() {
                let mut matrix = l.get_intra_weights();
                let i = ((t.get_position() / 64) / matrix.len() as u32) as usize;
                let j = ((t.get_position() / 64) % matrix.len() as u32) as usize;
                let mut vec_byte: Vec<_> = matrix[i][j].to_ne_bytes().iter().cloned().collect();

                let byte_original: u8 = vec_byte.get(t.get_position() as usize / 8).cloned().unwrap_or(0u8);
                let value_bit = (byte_original >> (t.get_position() % 8)) & 1;
                modify_bits(&mut vec_byte, t.get_position() as u8 % 64u8, value_bit);
                t.set_bit_changed(true);

                matrix[i][j] = f64::from_ne_bytes(vec_byte.as_slice().try_into().unwrap());
                l.set_intra_weights(matrix);
                assert_eq!(l.get_intra_weights(), vec![
                    vec![0.0, 0.1, -0.15],
                    vec![-0.05, 0.0, -0.1],
                    vec![-0.15, -0.1, 0.0],
                ]);//use position 65
            }
        }
        _ => {}
    }
}*/