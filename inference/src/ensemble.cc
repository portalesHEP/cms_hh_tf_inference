#include "inference/include/ensemble.hh"

Ensemble::Ensemble(std::string root_name, unsigned int n_threads, bool verbose=false) {
    /* Class to handle loading and inference of ensembling of models */

    _verbose = verbose;
    _n_threads = n_threads;
    assert(Ensemble::load_ensemble(root_name));
}

Ensemble::~Ensemble() {
    _models.clear();
}

bool Ensemble::load_ensemble(std::string root_name) {
    /* Load ensemble of models and settings */

    if (!boost::filesystem::exists(root_name + "model_weights.txt")) {
        throw std::invalid_argument("File: " + root_name + "model_weights.txt not found");
        return false;
    }

    // Load models and weights
    if (_verbose) std::cout << "Required file found\n";
    std::string line;
    std::ifstream infile(root_name + "model_weights.txt");
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string name, weight;
        if (!(iss >> name >> weight)) break; // error
        if (_verbose) std::cout << "Loading model " << name << " with weight " << weight << "\n";
        _models.push_back(NN(root_name + name, _n_threads, _verbose));
        _weights.push_back(std::stof(weight));
    }
    infile.close();
    _n_models = _models.size();
    if (_verbose) std::cout << _n_models << " loaded\n";

    // Renormalise weights
    float sum = 0.0;
    for (float i : _weights) sum += i;
    if (_verbose) std::cout << "Weight sum " << sum << "\n";
    for (unsigned int i = 0; i < _n_models; i++) _weights[i] /= sum;
    if (_verbose) {
        float sum = 0.0;
        for (float i : _weights) sum += i;
        std::cout << "Weight sum after renormalisation " << sum << "\n";
    }
    return true;
}

float Ensemble::predict(tensorflow::Tensor input) {
    /* Pass features through all network and return weighted class prediction */

    if (_verbose) std::cout << "Predicting input\n";
    float pred = 0.0;
    float tmp;
    for (unsigned int i = 0; i < _n_models; i++) {
        tmp = _models[i].predict(input);
        if (_verbose) std::cout << "Model " << i << " predicts " << tmp << "\n";
        pred +=_weights[i]*tmp;
    }
    if (_verbose) std::cout << "Weighted prediction " << pred << "\n";
    return pred;
}