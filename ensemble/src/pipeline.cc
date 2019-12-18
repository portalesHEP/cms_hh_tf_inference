#include "pipeline.hh"

Pipeline::Pipeline(Preproc preproc, Ensemble ensemble, bool verbose) {
    _preproc = preproc;
    _ensemble = ensemble;
    _verbose = verbose;
}

Pipeline::~Pipeline() {
    delete _preproc;
    delete _ensemble;
}

float Pipeline::predict(std::vector<float> input) {
    if (_verbose) std::cout << "Preprocessing input\n";
    std::vector<float> x = _preproc(input);
    if (_verbose) std::cout << "Input processed\n";
    if (x.size() != input.size()) throw std::length_error("Something went wrong in preprocessing")
    if (_verbose) std::cout << "Predicting input\n";
    float y = _ensemble.predict(x);
    if (_verbose) std::cout << "Prediction is " << y << "\n";
    return y;
}