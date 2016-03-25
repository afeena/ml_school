#include "DB.h"
#include <exception>

DB::DB() {
	this->connection.connect("mlschool", "localhost", "mlstudent1", "b3e462d2b0");
	if (!connection.connected()) {
		throw std::exception();
	}
}

void DB::save(std::vector<int> identificators, std::vector<int> labels) {
	if (identificators.size() != labels.size()) {
		std::cout << "can't save";
		return;
	}
	mysqlpp::Query ins_query = connection.query();
	auto i_it = identificators.begin();
	auto l_it = labels.begin();
	while (i_it != identificators.end() && l_it != labels.end()) {
		ins_query << "REPLACE INTO mlstudent1" <<
			" VALUES (" << (*i_it) << "," << (*l_it) << ");";
		ins_query.execute();
		i_it++;
		l_it++;
	}

}

DB::~DB() {
}

