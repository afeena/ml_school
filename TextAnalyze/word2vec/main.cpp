#include <cstdlib>
#include <iostream>
#include "word2vec.h"
using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {


	Word2Vec<std::string> model;
	std::string file_name = "/home/mlstudent1/w2v_model.txt";
	model.load_text(file_name);

	std::string word = argv[1];
	auto most_similar = model.most_similar(std::vector<std::string>{word}, std::vector<std::string>(), 10);
	size_t i = 0;
	for (auto& v : most_similar) {
		if (v.second != 0) {
			std::cout << v.first << " " << v.second << std::endl;
		}
	}
	return 0;
}

