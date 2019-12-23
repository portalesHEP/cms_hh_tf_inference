#include "../include/preproc.hh"

Preproc::Preproc(std::string file_name, bool verbose=false) {
    /* Class to handle preprocessing of inpute features */

    _verbose = verbose;
    assert(Preproc::load_preproc(file_name));
}

Preproc::~Preproc() {}

bool Preproc::load_preproc(std::string file_name) {
    /* Load preproc settings */
    if (!boost::filesystem::exists(file_name)) {
        throw std::invalid_argument("File: " + file_name + " not found");
        return false;
    }

    // Load models and weights
    if (_verbose) std::cout << "Required file found\n";
    std::string line;
    std::ifstream infile(file_name);
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string mean, stdev;
        if (!(iss >> mean >> stdev)) break; // error
        if (_verbose) std::cout << "Mean " << mean << " stdev " << stdev << "\n";
        _means.push_back(std::stof(mean));
        _stdevs.push_back(std::stof(stdev));
    }
    infile.close();
    if (_verbose) std::cout << _means.size() << " means & stdevs loaded\n";
    return true;
}

tensorflow::Tensor Preproc::process(std::vector<float> input) {
    tensorflow::Tensor output(tensorflow::DT_FLOAT, {1, static_cast<long long int>(input.size())});
    float val;
    for (unsigned int i = 0; i < input.size(); i++) {
        val = input[i];
        if (i < _means.size()) { // Rescale continuous inputs, leave categoricals
            if (std::isnan(val)) {
                val = 0.0;
            } else {
                val -= _means[i];
                val /= _stdevs[i];
            }
        }
        output.matrix<float>()(0,static_cast<Eigen::Index>(i)) = val;
    }
    return output;
}