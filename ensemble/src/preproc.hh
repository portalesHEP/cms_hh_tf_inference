 
#ifndef PREPROC_HH_
#define PREPROC_HH_

// C++
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <experimental/filesystem>

class Preproc {
	/* Class to handle preprocessing of inpute features*/

private:
    std::vector<float> _means, _stdevs;
    bool _verbose;

public:
    // Methods
	Preproc(std::string, bool);
	~Preproc();
	std::vector<float> process(std::vector<float>);
    bool load_preproc(std::string);
};

#endif /* PREPROC_HH_ */