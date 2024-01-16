use std::ffi::OsStr;
use std::fs::File;
use std::io::Write;
use spiking_neural_network::failure::*;
use spiking_neural_network::lif_neuron::LifNeuron;
use spiking_neural_network::snn::builder::SnnBuilder;
use std::process::{Command};
use crate::{get_file_name, read_multiple_input_spikes,
            build_neurons, build_intra_weights, get_current_dir,
            read_extra_weights, N_NEURONS, N_INPUTS, N_INSTANTS};

pub fn demo() {
    let path = get_current_dir();

    /* build parameters of the network */
    let input_spikes: Vec<Vec<Vec<u8>>> = read_multiple_input_spikes(&path);
    let neurons: Vec<LifNeuron> = build_neurons(&path);
    let extra_weights: Vec<Vec<f64>> = read_extra_weights(&path);
    let intra_weights: Vec<Vec<f64>> = build_intra_weights();

    /* run demo over snn with fault configuration */
    let conf_fault = Conf::new(
        vec![Components::VRest], Failure::TransientBitFlip(TransientBitFlip::new(0)), 10);

    let file_name = get_file_name(&conf_fault);
    let path_output1 = format!("{path}/simulation/configurations/{file_name}");
    let mut output_file1 = File::create(path_output1).expect("Something went wrong opening the file outputCounters.txt!");

    let mut snn = SnnBuilder::new(N_INPUTS)
        .add_layer(neurons.clone(), extra_weights.clone(), intra_weights.clone(), conf_fault.clone())
        .build();

    let output_spikes = snn.process(&input_spikes[0]);
    write_output(&mut output_file1, output_spikes);


    /* run demo over snn with NO fault */
    let conf_no_fault = Conf::new(vec![], Failure::None, 0);

    let file_name = get_file_name(&conf_no_fault);
    let path_output2 = format!("{path}/simulation/configurations/{file_name}");
    let mut output_file2 = File::create(path_output2).expect("Something went wrong opening the file outputCounters.txt!");

    let mut snn = SnnBuilder::new(N_INPUTS)
        .add_layer(neurons.clone(), extra_weights.clone(), intra_weights.clone(), conf_no_fault.clone())
        .build();

    let output_spikes = snn.process(&input_spikes[0]);
    write_output(&mut output_file2, output_spikes);


    /* when all simulations are finished run the python script to print logs file */
    let string = format!("{path}/simulation/runDemo.py");
    let path_py = OsStr::new(&string);
    Command::new("python").arg(path_py).output()
        .expect("Error during execution of the command");
}

fn write_output(output_file: &mut File, output_spikes: Vec<Vec<u8>>) {
    let mut neurons_sum = vec![0u32; 400];
    for k in 0..N_NEURONS {
        for j in 0..N_INSTANTS {
            neurons_sum[k] += output_spikes[k][j] as u32;
        }
    }
    for n in 0..N_NEURONS {
        output_file.write_all(format!("{}\n", neurons_sum[n]).as_bytes()).expect("Something went wrong writing into the file outputCounters.txt!");
    }
}