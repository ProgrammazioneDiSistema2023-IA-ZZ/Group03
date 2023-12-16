use spiking_neural_network::lif_neuron::LifNeuron;
use spiking_neural_network::neuron::Neuron;
#[test]
fn verify_init() {
    let rest = 0.2;
    let mut n = LifNeuron::new(0.1, rest, 0.3, 0.4);

    n.init();

    assert_eq!(n.get_v_mem(), rest);
    assert_eq!(n.get_ts(), 0u64);
}

#[test]
fn verify_get_v_mem_one() {
    let extra_sum = 5.3;
    let t = 2u64;
    let v_reset = 0.3;

    let mut n = LifNeuron::new(0.1, 0.2, v_reset, 0.4);

    assert_eq!(n.calculate_v_mem(t,extra_sum),1);
    assert_eq!(n.get_v_mem(),v_reset);

}

#[test]
fn verify_get_v_mem_zero() {
    let extra_sum = 5.3;
    let t = 2u64;
    let v_reset = 0.3;

    let mut n = LifNeuron::new(20.2, 0.2, v_reset, 0.4);

    assert_eq!(n.calculate_v_mem(t,extra_sum),0);

}

#[test]
fn verify_value_neuron() {
    let v_reset = 0.3;
    let mut n = LifNeuron::new(20.2, 0.2, v_reset, 0.4);
    let campo = n.get_v_th();
    n.set_v_th(campo);
    assert_eq!( n.get_v_th() ,campo);

}