#include "../include/nn.hh"

NN::NN(std::string root_name, unsigned int n_threads, bool verbose=false) {
    /* Basic class to load and apply saved TF models */

    _verbose = verbose;
    _n_threads = n_threads;
    assert(NN::load_model(root_name));
}

NN::~NN() {
    delete _model;
}

bool NN::load_model(std::string root_name) {
    /* Load TF model from specified protocol buffer file */

    if (!std::experimental::filesystem::exists(root_name + ".pb")) {
        throw std::invalid_argument("File: " + root_name + ".pb not found");
        return false;
    }
    // if (!std::experimental::filesystem::exists(root_name + "_info.txt")) {
    //     throw std::invalid_argument("File: " + root_name + "_info.txt not found");
    //     return false;
    // }

    // if (_verbose) std::cout << "Both required files found\n";
    // std::string line;
    // std::ifstream infile(root_name + "_info.txt");
    // while (std::getline(infile, line)) {
    //     std::istringstream iss(line);
    //     std::string arg, val;
    //     if (!(iss >> arg >> val)) break; // error
    //     if (arg == "input_name") {
    //         if (_verbose) std::cout << "Input name " << val << "\n";
    //         _input_name = val;
    //     } else if (arg == "output_name") {
    //         if (_verbose) std::cout << "Output name " << val << "\n";
    //         _output_name = val;
    //     }
    // }
    // infile.close();
    if (_verbose) std::cout << "Loading model\n";
    _model = tensorflow::loadGraphDef(root_name + "_model.pb", _n_threads);
    if (_verbose) std::cout << "Model loaded\n";

     _input_name  = _model->node(0).name();
     _output_name = _model->node(_model->node_size()-1).name();

    if (_verbose) {
        for (int i = 0; i < _model->node_size(); i++) std::cout << "Tensor " << i << " name " <<  _model->node(i).name() << "\n";
    }
    return true;
}

float NN::predict(tensorflow::Tensor input) {
    /* Pass features through network and return class prediction */

    if (_verbose) std::cout << "Launching TF session\n";
    tensorflow::Session* session = tensorflow::createSession(_model, _n_threads);
    if (_verbose) std::cout << "TF session launched\n";

    if (_verbose) std::cout << "Running model\n";
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