use std::sync::mpsc::{Receiver, Sender};
use crate::snn::neuron::Neuron;
use crate::snn::spike_event::SpikeEvent;
use crate::snn::configuration::Configuration;
use crate::failure::{Components, Failure};
use bit::BitIndex;

#[derive(Debug)]
pub struct Layer<N: Neuron + Clone + Send + 'static, R: Configuration + Clone + Send + 'static> {
    neurons: Vec<N>,
    weights: Vec<Vec<f64>>,
    intra_weights: Vec<Vec<f64>>,
    prev_spikes: Vec<u8>,
    configuration: R,
}

impl<N: Neuron + Clone + Send + 'static, R: Configuration + Clone + Send + 'static> Layer<N, R> {
    pub fn new(
        neurons: Vec<N>,
        weights: Vec<Vec<f64>>,
        intra_weights: Vec<Vec<f64>>,
        configuration: R,
    ) -> Self {
        let num_neurons = neurons.len();
        Self {
            neurons,
            weights,
            intra_weights,
            prev_spikes: vec![0; num_neurons],
            configuration,
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

    pub fn get_prev_spikes(&self) -> Vec<u8> { self.prev_spikes.clone() }

    pub fn get_configuration(&self) -> R { self.configuration.clone() }

    pub fn set_intra_weights(&mut self, val: Vec<Vec<f64>>) { self.intra_weights = val }

    pub fn set_weights(&mut self, val: Vec<Vec<f64>>) { self.weights = val }

    fn generate_faults(&mut self) {
        /* If there is at least one component to fail, search the selected component to keep it broken */
        for component in self.configuration.get_vec_components() {
            /* Get index of neuron and info of failure from configuration field */
            let index = self.configuration.get_index_neuron();
            let failure = self.configuration.get_failure();

            /* Get neuron from vec of neurons */
            let neuron = self.neurons.get_mut(index).unwrap();

            match component {
                Components::VTh => {
                    let new_val = modify_bits(failure, neuron.get_v_th().to_bits());
                    neuron.set_v_th(f64::from_bits(new_val));
                }
                Components::VRest => {
                    let new_val = modify_bits(failure, neuron.get_v_rest().to_bits());
                    neuron.set_v_rest(f64::from_bits(new_val));
                }
                Components::VReset => {
                    let new_val = modify_bits(failure, neuron.get_v_reset().to_bits());
                    neuron.set_v_reset(f64::from_bits(new_val));
                }
                Components::Tau => {
                    let new_val = modify_bits(failure, neuron.get_tau().to_bits());
                    neuron.set_tau(f64::from_bits(new_val));
                }
                Components::VMem => {
                    let new_val = modify_bits(failure, neuron.get_v_mem().to_bits());
                    neuron.set_v_mem(f64::from_bits(new_val));
                }
                Components::Ts => {
                    let new_val = modify_bits(failure, neuron.get_ts());
                    neuron.set_ts(new_val);
                }
                Components::Dt => {
                    let new_val = modify_bits(failure, neuron.get_dt().to_bits());
                    neuron.set_dt(f64::from_bits(new_val));
                }
                Components::IntraWeights => {
                    let mut matrix = self.intra_weights.clone();
                    let i = (failure.get_position().unwrap() / 64) / matrix.len();
                    let j = (failure.get_position().unwrap() / 64) % matrix.len();

                    let new_val = modify_bits(failure, matrix[i][j].to_bits());
                    matrix[i][j] = f64::from_ne_bytes(new_val.to_ne_bytes());
                    self.intra_weights = matrix;
                }
                Components::Weights => {
                    let mut matrix = self.weights.clone();
                    let i = (failure.get_position().unwrap() / 64) / matrix.len();
                    let j = (failure.get_position().unwrap() / 64) % matrix.len();

                    let new_val = modify_bits(failure, matrix[i][j].to_bits());
                    matrix[i][j] = f64::from_ne_bytes(new_val.to_ne_bytes());
                    self.weights = matrix;
                }
                Components::PrevSpikes => {
                    /*TODO*/
                    /*
                    let mut vec = self.prev_spikes.clone();

                    let i = (failure.get_position().unwrap() / 8) / vec.len();

                    let new_val = modify_bits(failure, vec[i] as u64);
                    vec[i] = u8::from_ne_bytes(new_val.to_ne_bytes());
                    self.prev_spikes = vec;
                    */
                }
                Components::None => {}
            }
        }
    }

    fn generate_spike(&mut self, input_spike_event: &SpikeEvent, instant: u64, output_spikes: &mut Vec<u8>, at_least_one_spike: &mut bool) {
        /* Generate FAULTS according to the configuration */
        self.generate_faults();

        /* For each neuron compute the sums of intra weights, extra weights and v_mem*/
        for (index, neuron) in self.neurons.iter_mut().enumerate() {
            let events = input_spike_event.get_spikes();

            /* compute extra weighted sum */
            let mut extra_weighted_sum = 0f64;
            let extra_weights_pairs = self.weights[index].iter().zip(events.iter());
            for (weight, spike) in extra_weights_pairs {
                if *spike != 0 {
                    extra_weighted_sum += *weight;
                }
            }

            /* compute intra weighted sum */
            let prev_events = self.prev_spikes.clone();
            let mut intra_weighted_sum = 0f64;
            let intra_weights_pairs = self.intra_weights[index].iter().zip(prev_events.iter());
            for (i, (weight, spike)) in intra_weights_pairs.enumerate() {
                /* skip the reflexive link */
                if i != index && *spike != 0 {
                    intra_weighted_sum += *weight;
                }
            }

            let neuron_spike = neuron.calculate_v_mem(instant, extra_weighted_sum + intra_weighted_sum);
            output_spikes.push(neuron_spike);
            if neuron_spike == 1u8 {
                *at_least_one_spike = true;
            }
        }
    }

    pub fn process(&mut self, layer_input_rc: Receiver<SpikeEvent>, layer_output_tx: Sender<SpikeEvent>) {
        /* initialize data structures, so that the SNN can be reused */
        self.init();

        /* listen to SpikeEvent(s) coming from the previous layer and process them */
        while let Ok(input_spike_event) = layer_input_rc.recv() {
            /* time instant of the input spike */
            let instant = input_spike_event.get_ts();

            /*flag to manage not firing layer*/
            let mut at_least_one_spike = false;

            let mut output_spikes = Vec::<u8>::with_capacity(self.neurons.len());

            /*this function compute the v_mem for all the neurons considering the input spikes
            and if there is configuration field it provides to apply the fault */
            self.generate_spike(&input_spike_event, instant, &mut output_spikes, &mut at_least_one_spike);

            /* save output spikes for later */
            self.prev_spikes = output_spikes.clone();

            /* check if at least one neuron fired - if not, not send any spike */
            if !at_least_one_spike {
                continue;
            }

            /* at least one neuron fired -> send output spikes to the next layer */
            let output_spike_event = SpikeEvent::new(instant, output_spikes);

            layer_output_tx.send(output_spike_event)
                .unwrap_or_else(|_| panic!("Unexpected error sending input spike event t={}", instant))
        }
    }

    pub fn init(&mut self) {
        self.prev_spikes.clear();    /* reset prev_spikes */
        self.neurons.iter_mut().for_each(|neuron| neuron.init());  /* reset neurons */
    }
}

pub fn modify_bits(failure: Failure, mut val: u64) -> u64 {

    let position = failure.get_position().unwrap();
    if position >= 64 {
        return val;
        //position = position%64;
    }

    /* Match the type of failure -> StuckAt0 / StuckAt1 / TransientBitFlip */
    match failure {
        Failure::StuckAt0(_s) => {
            val.set_bit(63-position, false);
            val
        }
        Failure::StuckAt1(_s) => {
            val.set_bit(63-position, true);
            val
        }
        Failure::TransientBitFlip(mut t) => {
            if !t.get_bit_changed() {
                let old_bit = val.bit(63-position);
                val.set_bit(63-position, !old_bit);
                t.set_bit_changed(true);
            }
            val
        }
        _ => { val }
    }
}

impl<N: Neuron + Clone + Send + 'static, R: Configuration + Clone + Send + 'static> Clone for Layer<N, R> {
    fn clone(&self) -> Self {
        Self {
            neurons: self.neurons.clone(),
            weights: self.weights.clone(),
            intra_weights: self.intra_weights.clone(),
            prev_spikes: self.prev_spikes.clone(),
            configuration: self.configuration.clone(),
        }
    }
}