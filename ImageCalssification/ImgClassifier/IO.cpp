#include "IO.h"
#include "csv.h"
#include <iostream>
#include <fstream> 

IO::IO() {
}

IO::IO(const IO& orig) {
}

IO::~IO() {
}

std::unordered_map<int, int> IO::read_train_csv(std::string filename) {
	std::unordered_map<int, int> data;
	io::CSVReader<3> in(filename);
	in.read_header(io::ignore_extra_column, "image_id", "image_url", "image_label");
	std::string url; 
	int identifier;
	int label;
	while (in.read_row(identifier, url, label)) {
		data.insert({identifier, label});
	}
	return data;
}
std::vector<int> IO::read_test_csv(std::string filename) {
	std::vector<int> data;
	io::CSVReader<2> in(filename);
	in.read_header(io::ignore_extra_column, "image_id", "image_url");
	std::string url; 
	int identifier;
	while (in.read_row(identifier, url)) {
		data.push_back(identifier);
	}
	return data;
}
std::vector<float> IO::feature_indices(std::string filename){
	
	std::vector<float> indices;
	std::ifstream ind_file;
	ind_file.open(filename);
	int ind = 0;
	while(!ind_file.eof()){
		ind_file >> ind;
		indices.push_back(ind);
	}
	ind_file.close();
	return indices;
	
}	



