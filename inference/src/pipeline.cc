#include "interface/include/pipeline.hh"

Pipeline::Pipeline(Preproc* preproc, Ensemble* ensemble, bool verbose) {
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
    tensorflow::Tensor x = _preproc->process(input);
    if (_verbose) std::cout << "Input processed, predicting input\n";
    float y = _ensemble->predict(x);
    if (_verbose) std::cout << "Prediction is " << y << "\n";
    return y;
}