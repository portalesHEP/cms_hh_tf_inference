#include "nn.hh"

NN::NN(std::string root_name, unsigned int n_threads, bool verbose=false) {
    /* Basic class to load and apply saved TF models */

    _verbose = verbose;
    _n_threads = n_threads;
    assert(NN::load_model(root_name));
}

NN::~NN() {
    delete model;
}

bool NN::load_model(std::string root_name) {
    /* Load TF model from specified protocol buffer file */

    if (!std::filesystem::exists(root_name + "_model.pb") {
        throw std::invalid_argument("File: " + root_name + "_model.pb not found");
        return false;
    }

    if (!std::filesystem::exists(root_name + "_info.txt") {
        throw std::invalid_argument("File: " + root_name + "_info.txt not found");
        return false;
    }

    if (_verbose) std::cout << "Both required filed found\n";
    std::string line;
    std::ifstream argfile(root_name + "_info.txt");
    while (std::getline(argfile, line)) {
        std::istringstream iss(line);
        std::string arg, val;
        if (!(iss >> arg >> val)) break; // error
        if (arg == "input_sz") {
            if (_verbose) std::cout << "Input size " << val << "\n";
            _input_sz = static_cast<unsigned int>(val);
        } else if (arg == "input_name") {
            if (_verbose) std::cout << "Input name " << val << "\n";
            _input_name = val;
        } else if (arg == "output_name") {
            if (_verbose) std::cout << "Output name " << val << "\n";
            _output_name = val;
        }
    }
    argfile.close();
    if (_verbose) std::cout << "Loading model\n";
    _model = tensorflow::loadMetaGraph(root_name + "_model.pb", _n_threads);
    if (_verbose) std::cout << "Model loaded\n";
    return true;
}

float NN::predict(std::vector<float> input) {
    if (input.size() > _input_sz) {
        throw std::length_error("NN expects input of length " << _input_sz << " but received input of length " << input.size());
        return -1.0;
    }

    if (_verbose) std::cout << "Building input tensor\n";
    tensor_input = tensorflow::Tensor(tensorflow::DT_FLOAT, {1, _input_sz});
    for (unsigned int i = 0; i < input.size(); i++) tensor_input.matrix<float>()(0, static_cast<Eigen::Index>(i)) = static_cast<float>(input[i]);
    if (_verbose) std::cout << "Input tensor built\n";

    if (_verbose) std::cout << "Launching TF session\n";
    session = tensorflow::createSession(graphDef, _n_threads);
    if (_verbose) std::cout << "TF session launched\n";

    if (_verbose) std::cout << "Running model\n";
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::run(session, {{_input_name, tensor_input}}, {_output_name}, &outputs);
    pred = outputs[0].matrix<float>()(0, 0);
    if (_verbose) std::cout << "Event evaulated, class prediction is: " << pred << "\n";
    tensorflow::closeSession(session);

    if (pred >= 0.0 && pred <= 1.0) {
        throw std::out_of_range("Model prediction of " << pred << " not within range of [0,1]");
        return -1;
    }
    return pred
}