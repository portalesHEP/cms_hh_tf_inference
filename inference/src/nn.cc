#include "cms_hh_tf_inference/inference/interface/nn.hh"

NN::NN(std::string root_name, unsigned int n_threads, bool verbose=false) {
    /* Basic class to load and apply saved TF models */

    _verbose = verbose;
    _n_threads = n_threads;
    if (_verbose) std::cout << "Loading model\n";
    assert(NN::load_model(root_name));
    if (_verbose) std::cout << "Model loaded\n";
}

NN::~NN() {
    delete _model;
}

bool NN::load_model(std::string root_name) {
    /* Load TF model from specified protocol buffer file */

    if (!boost::filesystem::exists(root_name + ".pb")) {
        throw std::invalid_argument("File: " + root_name + ".pb not found");
        return false;
    }
    if (_verbose) std::cout << "Loading protobuf...";
    _model = tensorflow::loadGraphDef(root_name + ".pb");
    if (_verbose) std::cout << "protobuf loaded\n";

     _input_name  = _model->node(0).name();
     _output_name = _model->node(_model->node_size()-1).name();

    if (_verbose) std::cout << "Using " << _input_name << " as input and " << _output_name << " as output\n";
    return true;
}

float NN::predict(tensorflow::Tensor input) {
    /* Pass features through network and return class prediction */

    if (_verbose) std::cout << "Launching TF session... ";
    tensorflow::Session* session = tensorflow::createSession(_model, _n_threads);
    if (_verbose) std::cout << "TF session launched\n";

    if (_verbose) std::cout << "Running model:\n";
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::run(session, {{_input_name, input}}, {_output_name}, &outputs);
    float pred = outputs[0].matrix<float>()(0,0);
    if (_verbose) std::cout << "Event evaulated, class prediction is: " << pred << "\n";
    tensorflow::closeSession(session);

    if (pred < 0.0 || pred > 1.0) {
        throw std::out_of_range("Model prediction of " + std::to_string(pred) + " not within range of [0,1]");
        return -1;
    }
    return pred;
}