use std::sync::mpsc::{Receiver, Sender};
use crate::snn::neuron::Neuron;
use crate::snn::spike_event::SpikeEvent;

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

    pub fn process(&mut self, layer_input_rc: Receiver<SpikeEvent>, layer_output_tx: Sender<SpikeEvent>) {
        /* initialize data structures, so that the SNN can be reused */
        self.init();

        /* listen to SpikeEvent(s) coming from the previous layer and process them */
        while let Ok(input_spike_event) = layer_input_rc.recv() {
            let instant = input_spike_event.get_ts();    /* time instant of the input spike */
            let mut output_spikes = Vec::<u8>::with_capacity(self.neurons.len());
            let mut at_least_one_spike = false;

            /*
                for each neuron compute the intra and the extra weighted sums,
                then retrieve the output spike
            */
            for (index, neuron) in self.neurons.iter_mut().enumerate() {
                let mut extra_weighted_sum = 0f64;
                let mut intra_weighted_sum = 0f64;

                /* compute extra weighted sum */
                let events = input_spike_event.get_spikes();
                let extra_weights_pairs =
                    self.weights[index].iter().zip(events.iter());

                for (weight, spike) in extra_weights_pairs {
                    if *spike != 0 {
                        extra_weighted_sum += *weight;
                    }
                }

                /* compute intra weighted sum
                   (intra_weights[index] contains the weights of the links to the current neuron) */
                let intra_weights_pairs =
                    self.intra_weights[index].iter().zip(self.prev_spikes.iter());

                for (i, (weight, spike)) in intra_weights_pairs.enumerate() {
                    /* ignore the reflexive link */
                    if i != index && *spike != 0 {
                        intra_weighted_sum += *weight;
                    }
                }

                /* compute membrane potential and determine if the Neuron fires or not */
                let neuron_spike = neuron.calculate_v_mem(instant, extra_weighted_sum, intra_weighted_sum);
                output_spikes.push(neuron_spike);

                if !at_least_one_spike && neuron_spike == 1u8 {
                    at_least_one_spike = true;
                }
            }

            /* save output spikes for later */
            self.prev_spikes = output_spikes.clone();

            /* check if at least one neuron fired - if not, not send any spike */
            if !at_least_one_spike {
                continue;
            }
            /* at least one neuron fired -> send output spikes to the next layer */

            let output_spike_event = SpikeEvent::new(instant, output_spikes);

            layer_output_tx.send(output_spike_event)
                .expect(&format!("Unexpected error sending input spike event t={}", instant));
        }

        /*
            we don't need to drop the sender, because it will be
            automatically dropped when the layer goes out of scope
        */
    }


    pub fn init(&mut self) {
        self.prev_spikes.clear();    /* reset prev_spikes */
        self.neurons.iter_mut().for_each(|neuron| neuron.init());  /* reset neurons */
    }
}

impl<N: Neuron + Clone + Send + 'static> Clone for Layer<N> {
    fn clone(&self) -> Self {
        Self {
            neurons: self.neurons.clone(),
            weights: self.weights.clone(),
            intra_weights: self.intra_weights.clone(),
            prev_spikes: self.prev_spikes.clone()
        }
    }
}