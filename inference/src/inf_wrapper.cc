#include "inf_wrapper.hh"

InfWrapper::InfWrapper(std::string root_name, unsigned int n_threads, bool verbose) {
    _verbose = verbose;
    assert(InfWrapper::load_pipeline(root_name + "_0", n_threads));
    assert(InfWrapper::load_pipeline(root_name + "_1", n_threads));
}

InfWrapper::~InfWrapper() {
    _pipes.clear();
}

InfWrapper::load_pipeline(std::string root_name, unsigned int n_threads) {
    _pipes.push_back(Pipeline(Preproc(root_name + "_preproc.txt", _verbose),
                              Ensemble(root_name, n_threads, _verbose),
                              _verbose))
}

float InfWrapper::predict(std::vector<float> input, unsigned long int event_id) {
    if (event_id % 2 == 0) {
        if (_verbose) std::cout << "Even Event ID, passing through pipeline 0\n";
        return _pipes[0].predict(input);
    } else {
        if (_verbose) std::cout << "Odd Event ID, passing through pipeline 1\n";
        return _pipes[1].predict(input);
    }
}