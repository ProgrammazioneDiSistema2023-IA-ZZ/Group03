use crate::neuron::Neuron;
use crate::snn::layer::Layer;
use crate::spike_event::SpikeEvent;
use crate::configuration::Configuration;
use std::slice::IterMut;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::channel;
use std::thread;
use std::thread::JoinHandle;

#[derive(Debug, Clone)]
pub struct SNN<N: Neuron + Clone + 'static, R: Configuration + Clone + Send + 'static> {
    layers: Vec<Arc<Mutex<Layer<N, R>>>>,
}

impl<N: Neuron + Clone, R: Configuration + Clone + Send + 'static> SNN<N, R> {
    pub fn new(layers: Vec<Arc<Mutex<Layer<N, R>>>>) -> Self {
        Self { layers }
    }

    pub fn get_number_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn get_input_layer_one_dim(&self) -> usize {
        let first_layer = self.layers[0].lock().unwrap();
        first_layer.get_weights().first().unwrap().len()
    }

    pub fn get_output_last_layer_dim(&self) -> usize {
        let last_layer = self.layers.last().unwrap().lock().unwrap();
        last_layer.get_number_neurons()
    }

    pub fn get_layers(&self) -> Vec<Layer<N, R>> {
        //self.layers.iter().map(|layer| layer.lock().unwrap().clone()).collect()
        self.layers.iter().map(|layer| layer.lock().unwrap().clone()).collect()
    }

    /**
    Process input spikes through the Spiking Neural Network and generate corresponding output spikes.
    The 'spikes' variable comprises an array for each neuron in the input layer, with each array
    containing the same number of spikes, reflecting the input duration. (Spikes is represented
    as a matrix, with one row for each input neuron and one column for each time instant.)
    This approach examines user input during the runtime.
     */
    pub fn process(&mut self, spikes: &Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        // * check and compute the spikes duration *
        let spikes_duration = self.spikes_duration(spikes);

        let input_layer_dimension = self.get_input_layer_one_dim();
        let output_layer_dimension = self.get_output_last_layer_dim();

        // * encode spikes into SpikeEvent(s) *
        let input_spike_events =
            SNN::<N, R>::encode_spikes(input_layer_dimension, spikes, spikes_duration);

        // * process input *
        let output_spike_events = self.process_events(input_spike_events);

        // * decode output into array shape *
        let decoded_output = SNN::<N, R>::decode_spikes(output_layer_dimension,
                                                        output_spike_events, spikes_duration);

        decoded_output
    }

    /**
    This function checks if each vector passed in 'spikes' has the same number of spikes.
    If yes, it returns the duration, otherwise it triggers an error
     */
    fn spikes_duration(&self, spikes: &Vec<Vec<u8>>) -> usize {
        // compute length of the first Vec (0 if it does not exist)
        let spikes_duration = spikes.get(0)
            .unwrap_or(&Vec::new())
            .len();

        for neuron_spikes in spikes {
            if neuron_spikes.len() != spikes_duration {
                panic!("The number of spikes duration must be equal for each neuron");
            }
        }
        spikes_duration
    }

    /**
    This function encodes the received input spikes in a Vec of **SpikeEvent** to process them.
     */
    fn encode_spikes(input_layer_dimension: usize, spikes: &Vec<Vec<u8>>, spikes_duration: usize) -> Vec<SpikeEvent> {
        let mut spike_events = Vec::<SpikeEvent>::new();
        if spikes.len() != input_layer_dimension {
            panic!("Error: number of input spikes is not coherent with the input layer dimension, \
                    'spikes' must have a Vec for each neuron");
        }
        for t in 0..spikes_duration {
            let mut t_spikes = Vec::<u8>::new();
            /* retrieve the input spikes for each neuron */
            for (in_neuron_index, spike) in spikes.iter().enumerate() {
                /* check for 0 or 1 only */
                if spike[t] != 0 && spike[t] != 1 {
                    panic!("Error: input spike must be 0 or 1 at for N={} at t={}", in_neuron_index, t);
                }
                t_spikes.push(spike[t]);
            }
            let t_spike_event = SpikeEvent::new(t as u64, t_spikes);
            spike_events.push(t_spike_event);
        }
        spike_events
    }

    /**
    This function decodes a Vec of SpikeEvents and returns an output spikes matrix of 0/1
     */
    fn decode_spikes(output_layer_dimension: usize, spikes: Vec<SpikeEvent>, spikes_duration: usize) -> Vec<Vec<u8>> {
        let mut raw_spikes = vec![vec![0; spikes_duration]; output_layer_dimension];
        for spike_event in spikes {
            for (out_neuron_index, spike) in spike_event.get_spikes().into_iter().enumerate() {
                raw_spikes[out_neuron_index][spike_event.get_ts() as usize] = spike;
            }
        }

        raw_spikes
    }

    fn process_events(&mut self, spikes: Vec<SpikeEvent>) -> Vec<SpikeEvent> {
        let mut threads = Vec::<JoinHandle<()>>::new();

        /* create channel to feed the (first layer of the) network */
        let (net_input_tx, mut layer_rc) = channel::<SpikeEvent>();

        /* create input TX and output RC for each layer and spawn layers' threads */
        for layer_ref in self {

            /* create channel to feed the next layer */
            let (layer_tx, next_layer_rc) = channel::<SpikeEvent>();

            let layer_ref_cloned = layer_ref.clone();

            let thread = thread::spawn(move || {
                /* retrieve layer */
                let mut layer = layer_ref_cloned.lock().unwrap();

                /* execute layer task */
                layer.process(layer_rc, layer_tx);
            });

            /* push the new thread into pool of threads */
            threads.push(thread);

            /* update external rc, to pass it to the next layer */
            layer_rc = next_layer_rc;
        }

        let net_output_rc = layer_rc;

        /* fire input SpikeEvents into *net_input_tx* */
        for spike_event in spikes {
            /* * check if there is at least 1 spike, otherwise skip to the next instant * */
            if spike_event.get_spikes().iter().all(|spike| *spike == 0u8) {
                continue;   /* (process only *effective* spike events) */
            }

            let instant = spike_event.get_ts();

            net_input_tx.send(spike_event)
                .unwrap_or_else(|_| panic!("Unexpected error sending input spike event t={}", instant))
        }

        drop(net_input_tx); /* drop input tx, to make all the threads terminate */

        /* get output SpikeEvents from *net_output* rc */
        let mut output_events = Vec::<SpikeEvent>::new();

        while let Ok(spike_event) = net_output_rc.recv() {
            output_events.push(spike_event);
        }

        /* waiting for threads to terminate */
        for thread in threads {
            thread.join().unwrap();
        }

        output_events
    }
}

impl<'a, N: Neuron + Clone + 'static, R: Configuration + Clone + Send + 'static> IntoIterator for &'a mut SNN<N, R> {
    type Item = &'a mut Arc<Mutex<Layer<N, R>>>;
    type IntoIter = IterMut<'a, Arc<Mutex<Layer<N, R>>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.layers.iter_mut()
    }
}
