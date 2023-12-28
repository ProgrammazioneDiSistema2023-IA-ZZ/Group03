use spiking_neural_network::failure::{Components, Conf, Failure, StuckAt0, StuckAt1, TransientBitFlip};
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

    let mut l = Layer::new(neurons, weights, intra_weights, configuration);

    l.set_prev_spikes(vec![1,1,0]);

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
    let failure = Failure::StuckAt0(StuckAt0::new(12));
    let configuration = Conf::new(vec![Components::VMem], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();

    let new_val = modify_bits(failure, n.get_v_th().to_bits());
    n.set_v_th(f64::from_bits(new_val));

    assert_eq!(n.get_v_th(), 0.65);//use position 12
}

#[test]
fn verify_modification_bit_v_th_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(14));
    let configuration = Conf::new(vec![Components::VTh], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();

    let new_val = modify_bits(failure, n.get_v_th().to_bits());
    n.set_v_th(f64::from_bits(new_val));
    assert_eq!(n.get_v_th(), 0.9625);//use position 14
}


#[test]
fn verify_modification_bit_v_th_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(14));
    let configuration = Conf::new(vec![Components::VTh], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();

    let new_val = modify_bits(failure, n.get_v_th().to_bits());
    n.set_v_th(f64::from_bits(new_val));
    assert_eq!(n.get_v_th(), 0.9625);//use position 14
}

#[test]
fn verify_modification_bit_v_rest_stuck0() {
    let failure = Failure::StuckAt0(StuckAt0::new(12));
    let configuration = Conf::new(vec![Components::VRest], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();

    let new_val = modify_bits(failure, n.get_v_rest().to_bits());
    n.set_v_rest(f64::from_bits(new_val));
    assert_eq!(n.get_v_rest(), 0.33);//use position 12
}

#[test]
fn verify_modification_bit_v_rest_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(14));
    let configuration = Conf::new(vec![Components::VRest], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();

    let new_val = modify_bits(failure, n.get_v_rest().to_bits());
    n.set_v_rest(f64::from_bits(new_val));
    assert_eq!(n.get_v_rest(), 0.36125);//use position 12
}


#[test]
fn verify_modification_bit_v_rest_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(14));
    let configuration = Conf::new(vec![Components::VRest], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();

    let new_val = modify_bits(failure, n.get_v_rest().to_bits());
    n.set_v_rest(f64::from_bits(new_val));
    assert_eq!(n.get_v_rest(), 0.36125);//use position 12
}

#[test]
fn verify_modification_bit_v_reset_stuck0() {
    let failure = Failure::StuckAt0(StuckAt0::new(12));
    let configuration = Conf::new(vec![Components::VReset], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();

    let new_val = modify_bits(failure, n.get_v_reset().to_bits());
    n.set_v_rest(f64::from_bits(new_val));
    assert_eq!(n.get_v_reset(), 0.14);//use position 12
}

#[test]
fn verify_modification_bit_v_reset_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(14));
    let configuration = Conf::new(vec![Components::VReset], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();


    let new_val = modify_bits(failure, n.get_v_reset().to_bits());
    n.set_v_reset(f64::from_bits(new_val));
    assert_eq!(n.get_v_reset(), 0.155625);//use position 15
}


#[test]
fn verify_modification_bit_v_reset_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(14));
    let configuration = Conf::new(vec![Components::VReset], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();

    let new_val = modify_bits(failure, n.get_v_reset().to_bits());
    n.set_v_reset(f64::from_bits(new_val));
    assert_eq!(n.get_v_reset(), 0.155625);//use position 15
}

#[test]
fn verify_modification_bit_tau_stuck0() {
    let failure = Failure::StuckAt0(StuckAt0::new(12));
    let configuration = Conf::new(vec![Components::Tau], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();


    let new_val = modify_bits(failure, n.get_tau().to_bits());
    n.set_tau(f64::from_bits(new_val));
    assert_eq!(n.get_tau(), 0.275);//use position 12
}

#[test]
fn verify_modification_bit_tau_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(14));
    let configuration = Conf::new(vec![Components::Tau], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();


    let new_val = modify_bits(failure, n.get_tau().to_bits());
    n.set_tau(f64::from_bits(new_val));
    assert_eq!(n.get_tau(), 0.43125);//use position 15
}


#[test]
fn verify_modification_bit_tau_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(14));
    let configuration = Conf::new(vec![Components::Tau], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();

    let new_val = modify_bits(failure, n.get_tau().to_bits());
    n.set_tau(f64::from_bits(new_val));
    assert_eq!(n.get_tau(), 0.43125);//use position 15
}


#[test]
fn verify_modification_bit_v_mem_stuck0() {
    let failure = Failure::StuckAt0(StuckAt0::new(12));
    let configuration = Conf::new(vec![Components::VMem], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();

    let new_val = modify_bits(failure, n.get_v_mem().to_bits());
    n.set_v_mem(f64::from_bits(new_val));
    assert_eq!(n.get_v_mem(), 0.33);//use position 12
}

#[test]
fn verify_modification_bit_v_mem_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(14));
    let configuration = Conf::new(vec![Components::VMem], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();

    let new_val = modify_bits(failure, n.get_v_mem().to_bits());
    n.set_v_mem(f64::from_bits(new_val));
    assert_eq!(n.get_v_mem(), 0.36125);//use position 12
}


#[test]
fn verify_modification_bit_v_mem_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(14));
    let configuration = Conf::new(vec![Components::VMem], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();

    let new_val = modify_bits(failure, n.get_v_mem().to_bits());
    n.set_v_mem(f64::from_bits(new_val));
    assert_eq!(n.get_v_mem(), 0.36125);//use position 12
}

#[test]
fn verify_modification_bit_dt_stuck0() {
    let failure = Failure::StuckAt0(StuckAt0::new(12));
    let configuration = Conf::new(vec![Components::Dt], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();


    let new_val = modify_bits(failure, n.get_dt().to_bits());
    n.set_dt(f64::from_bits(new_val));
    assert_eq!(n.get_dt(), 0.034375);//use position 12
}

#[test]
fn verify_modification_bit_dt_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(14));
    let configuration = Conf::new(vec![Components::Dt], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();

    let new_val = modify_bits(failure, n.get_dt().to_bits());
    n.set_dt(f64::from_bits(new_val));
    assert_eq!(n.get_dt(), 0.05390625);//use position 15
}


#[test]
fn verify_modification_bit_dt_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(14));
    let configuration = Conf::new(vec![Components::Dt], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();

    let new_val = modify_bits(failure, n.get_dt().to_bits());
    n.set_dt(f64::from_bits(new_val));
    assert_eq!(n.get_dt(), 0.05390625);//use position 15
}

#[test]
fn verify_modification_bit_ts_stuck0() {
    let failure = Failure::StuckAt0(StuckAt0::new(12));
    let configuration = Conf::new(vec![Components::Ts], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();


    let new_val = modify_bits(failure, n.get_ts());
    n.set_ts(new_val);
    assert_eq!(n.get_ts(), 0);//use position 12
}

#[test]
fn verify_modification_bit_ts_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(14));
    let configuration = Conf::new(vec![Components::Ts], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();

    let new_val = modify_bits(failure, n.get_ts());
    n.set_ts(new_val);
    assert_eq!(n.get_ts(), 562949953421312);//use position 15
}


#[test]
fn verify_modification_bit_ts_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(14));
    let configuration = Conf::new(vec![Components::Ts], failure.clone(), 1);
    let l = create_layer(configuration.clone());

    let mut neuron = l.get_neurons();
    let n = neuron.get_mut(0).unwrap();


    let new_val = modify_bits(failure, n.get_ts());
    n.set_ts(new_val);
    assert_eq!(n.get_ts(), 562949953421312);//use position 15
}

#[test]
fn verify_modification_bit_intra_weights_stuck0() {
    let failure = Failure::StuckAt0(StuckAt0::new(12));
    let configuration = Conf::new(vec![Components::IntraWeights], failure.clone(), 1);
    let mut l = create_layer(configuration.clone());

    let mut matrix = l.get_intra_weights();
    let i = (failure.get_position().unwrap() / 64) / matrix.len();
    let j = (failure.get_position().unwrap() / 64) % matrix.len();

    let new_val = modify_bits(failure, matrix[i][j].to_bits());
    matrix[i][j] = f64::from_bits(new_val);
    l.set_intra_weights(matrix);
    assert_eq!(l.get_intra_weights(), vec![
        [0.0, -0.1, -0.15],
        [-0.05, 0.0, -0.1],
        [-0.15, -0.1, 0.0],
    ]);
}


#[test]
fn verify_modification_bit_intra_weights_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(14));
    let configuration = Conf::new(vec![Components::IntraWeights], failure.clone(), 1);
    let mut l = create_layer(configuration.clone());

    let mut matrix = l.get_intra_weights();
    let i = (failure.get_position().unwrap() / 64) / matrix.len();
    let j = (failure.get_position().unwrap() / 64) % matrix.len();

    let new_val = modify_bits(failure, matrix[i][j].to_bits());
    matrix[i][j] = f64::from_bits(new_val);
    l.set_intra_weights(matrix);
    assert_eq!(l.get_intra_weights(), vec![
        [2.781342323134e-309, -0.1, -0.15],
        [-0.05, 0.0, -0.1],
        [-0.15, -0.1, 0.0]
    ]);
}

#[test]
fn verify_modification_bit_intra_weights_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(14));
    let configuration = Conf::new(vec![Components::IntraWeights], failure.clone(), 1);
    let mut l = create_layer(configuration.clone());

    let mut matrix = l.get_intra_weights();
    let i = (failure.get_position().unwrap() / 64) / matrix.len();
    let j = (failure.get_position().unwrap() / 64) % matrix.len();


    let new_val = modify_bits(failure, matrix[i][j].to_bits());
    matrix[i][j] = f64::from_bits(new_val);
    l.set_intra_weights(matrix);
    assert_eq!(l.get_intra_weights(), vec![
        [2.781342323134e-309, -0.1, -0.15],
        [-0.05, 0.0, -0.1],
        [-0.15, -0.1, 0.0]
    ]);
}


#[test]
fn verify_modification_bit_extra_weights_stuck0() {
    let failure = Failure::StuckAt0(StuckAt0::new(12));
    let configuration = Conf::new(vec![Components::Weights], failure.clone(), 1);
    let mut l = create_layer(configuration.clone());

    let mut matrix = l.get_weights();
    let i = (failure.get_position().unwrap() / 64) / matrix.len();
    let j = (failure.get_position().unwrap() / 64) % matrix.len();


    let new_val = modify_bits(failure, matrix[i][j].to_bits());
    matrix[i][j] = f64::from_bits(new_val);
    l.set_intra_weights(matrix);
    assert_eq!(l.get_intra_weights(), vec![
        vec![0.06875, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
    ]);
}

#[test]
fn verify_modification_bit_extra_weights_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(14));
    let configuration = Conf::new(vec![Components::Weights], failure.clone(), 1);
    let mut l = create_layer(configuration.clone());

    let mut matrix = l.get_weights();
    let i = (failure.get_position().unwrap() / 64) / matrix.len();
    let j = (failure.get_position().unwrap() / 64) % matrix.len();


    let new_val = modify_bits(failure, matrix[i][j].to_bits());
    matrix[i][j] = f64::from_bits(new_val);
    l.set_intra_weights(matrix);
    assert_eq!(l.get_intra_weights(), vec![
        [0.1078125, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]);
}

#[test]
fn verify_modification_bit_extra_weights_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(14));
    let configuration = Conf::new(vec![Components::Weights], failure.clone(), 1);
    let mut l = create_layer(configuration.clone());

    let mut matrix = l.get_weights();
    let i = (failure.get_position().unwrap() / 64) / matrix.len();
    let j = (failure.get_position().unwrap() / 64) % matrix.len();


    let new_val = modify_bits(failure, matrix[i][j].to_bits());
    matrix[i][j] = f64::from_bits(new_val);
    l.set_intra_weights(matrix);
    assert_eq!(l.get_intra_weights(), vec![
        [0.1078125, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]);
}


#[test]
fn verify_modification_bit_prev_spikes_stuck0() {
    let failure = Failure::StuckAt0(StuckAt0::new(12));
    let configuration = Conf::new(vec![Components::Weights], failure.clone(), 1);
    let mut l = create_layer(configuration.clone());

    let modified_prev_spikes = l.fault_prev_spikes(&failure);

    l.set_prev_spikes(modified_prev_spikes);

    assert_eq!(l.get_prev_spikes(), vec![0,1,0]);
}


#[test]
fn verify_modification_bit_prev_spikes_stuck1() {
    let failure = Failure::StuckAt1(StuckAt1::new(14));
    let configuration = Conf::new(vec![Components::Weights], failure.clone(), 1);
    let mut l = create_layer(configuration.clone());

    let modified_prev_spikes = l.fault_prev_spikes(&failure);

    l.set_prev_spikes(modified_prev_spikes);

    assert_eq!(l.get_prev_spikes(), vec![1,1,1]);
}

#[test]
fn verify_modification_bit_prev_spikes_transient() {
    let failure = Failure::TransientBitFlip(TransientBitFlip::new(14));
    let configuration = Conf::new(vec![Components::Weights], failure.clone(), 1);
    let mut l = create_layer(configuration.clone());

    let modified_prev_spikes = l.fault_prev_spikes(&failure);

    l.set_prev_spikes(modified_prev_spikes);

    assert_eq!(l.get_prev_spikes(), vec![1,1,1]);
}


